import os
import re
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS


def env_int(name: str, default: int) -> int:
    """Read a positive integer from env, falling back to default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"⚠️ Ignoring invalid {name}={value!r}; using {default}")
        return default
    return parsed if parsed > 0 else default


def env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean-ish flag from env."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class AlphaZeroTrainer:
    def __init__(self):
        # CPU is now the safe default for AlphaZero/MCTS. Override with AZ_DEVICE=auto, mps, cuda, or cpu.
        self.cpu_threads = env_int("AZ_TORCH_THREADS", 20)
        self.cpu_interop_threads = env_int("AZ_TORCH_INTEROP_THREADS", 1)
        torch.set_num_threads(self.cpu_threads)
        try:
            torch.set_num_interop_threads(self.cpu_interop_threads)
        except RuntimeError as exc:
            print(f"⚠️ Could not set interop threads after torch initialization: {exc}")

        self.device = self._select_device()
        print(f"Using device: {self.device}")
        print(f"Torch CPU threads: {torch.get_num_threads()}")
        print(f"Torch CPU interop threads: {torch.get_num_interop_threads()}")

        self.model = WallzNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Fast diagnostic defaults. Override without editing code, for example:
        # AZ_EPOCHS=50 AZ_GAMES_PER_EPOCH=10 AZ_MCTS_SIMULATIONS=25 python scripts/train_alphazero.py
        self.epochs = env_int("AZ_EPOCHS", 10)
        self.games_per_epoch = env_int("AZ_GAMES_PER_EPOCH", 2)
        self.mcts_simulations = env_int("AZ_MCTS_SIMULATIONS", 5)
        self.batch_size = env_int("AZ_BATCH_SIZE", 64)
        self.save_every = env_int("AZ_SAVE_EVERY", 1)
        self.max_steps_per_game = env_int("AZ_MAX_STEPS_PER_GAME", 200)
        self.show_progress = env_flag("AZ_PROGRESS", True)
        self.replay_buffer = deque(maxlen=10000)

        self.checkpoint_dir = ROOT_DIR / "wallz_v2" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = self._load_latest_checkpoint() + 1
        self.current_epoch = self.start_epoch - 1

        print(
            "Config -> "
            f"epochs={self.epochs}, games_per_epoch={self.games_per_epoch}, "
            f"mcts_simulations={self.mcts_simulations}, batch_size={self.batch_size}, "
            f"save_every={self.save_every}, max_steps_per_game={self.max_steps_per_game}, "
            f"device={self.device}, torch_threads={torch.get_num_threads()}, "
            f"progress={self.show_progress}"
        )

    def _select_device(self) -> torch.device:
        requested = os.getenv("AZ_DEVICE", "cpu").strip().lower()

        if requested == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            print("⚠️ AZ_DEVICE=mps requested, but MPS is unavailable. Falling back to CPU.")
            return torch.device("cpu")

        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("⚠️ AZ_DEVICE=cuda requested, but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")

        if requested != "cpu":
            print(f"⚠️ Unknown AZ_DEVICE={requested!r}; using CPU.")
        return torch.device("cpu")

    def _checkpoint_epoch(self, path: Path):
        match = re.fullmatch(r"alphazero_epoch_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else None

    def _load_latest_checkpoint(self) -> int:
        checkpoints = []
        for path in self.checkpoint_dir.glob("alphazero_epoch_*.pt"):
            epoch = self._checkpoint_epoch(path)
            if epoch is not None:
                checkpoints.append((epoch, path))

        if not checkpoints:
            print("No AlphaZero checkpoint found. Starting from scratch.")
            return 0

        epoch, path = max(checkpoints, key=lambda item: item[0])
        print(f"♻️ Loading AlphaZero checkpoint: {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Resuming after epoch {epoch}.")
        return epoch

    def save_checkpoint(self, epoch: int, interrupted: bool = False):
        if interrupted:
            path = self.checkpoint_dir / f"alphazero_interrupt_epoch_{epoch}.pt"
        else:
            path = self.checkpoint_dir / f"alphazero_epoch_{epoch}.pt"

        torch.save(self.model.state_dict(), path)
        latest_path = self.checkpoint_dir / "alphazero_latest.pt"
        torch.save(self.model.state_dict(), latest_path)
        tqdm.write(f"💾 Saved checkpoint to {path}")
        tqdm.write(f"💾 Updated latest checkpoint at {latest_path}")

    def self_play(self):
        """Generates training data by having the network play against itself using MCTS."""
        self.model.eval()
        game_iter = range(self.games_per_epoch)
        if self.show_progress:
            game_iter = tqdm(
                game_iter,
                total=self.games_per_epoch,
                desc="Self-play games",
                unit="game",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            print(f"\n🎮 Generating {self.games_per_epoch} self-play games...")

        completed_games = 0
        skipped_games = 0
        total_steps = 0

        for game in game_iter:
            env = WallzEnv()
            mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
            game_history = []

            terminal = False
            reward = 0.0
            step = 0

            while not terminal and step < self.max_steps_per_game:
                # Use temperature=1.0 for the first 15 moves to encourage exploration, then 0 to play strict best moves
                temp = 1.0 if step < 15 else 0.0

                # MCTS thinking
                action_probs = mcts.get_action_prob(env, temperature=temp)

                # Store state and target policy (from MCTS)
                game_history.append((env.get_observation(), action_probs, env.current_player))

                # Sample action
                if temp == 0:
                    action = np.argmax(action_probs)
                else:
                    action = np.random.choice(len(action_probs), p=action_probs)

                _, reward, terminal, _ = env.step(action)
                step += 1

                if self.show_progress and step % 5 == 0:
                    game_iter.set_postfix(
                        game=game + 1,
                        step=step,
                        replay=len(self.replay_buffer),
                        refresh=False,
                    )

            if not terminal:
                skipped_games += 1
                message = f"⚠️ Game {game + 1}/{self.games_per_epoch} hit max_steps={self.max_steps_per_game}; skipping it."
                if self.show_progress:
                    tqdm.write(message)
                else:
                    print(message)
                continue

            # Game over, assign final values to the history buffer
            winner = 1 if (reward == 1.0 and env.current_player == 2) else 2

            for obs, probs, player in game_history:
                # Value is +1 if this player won, -1 if they lost
                z = 1.0 if player == winner else -1.0
                self.replay_buffer.append((obs, probs, z))

            completed_games += 1
            total_steps += step
            if self.show_progress:
                game_iter.set_postfix(
                    game=game + 1,
                    steps=step,
                    winner=f"P{winner}",
                    replay=len(self.replay_buffer),
                    refresh=True,
                )
            else:
                print(f"Game {game + 1}/{self.games_per_epoch} complete (Steps: {step}). Winner: P{winner}")

        avg_steps = total_steps / completed_games if completed_games else 0.0
        return {
            "completed_games": completed_games,
            "skipped_games": skipped_games,
            "avg_steps": avg_steps,
            "replay_buffer": len(self.replay_buffer),
        }

    def train_network(self):
        """Trains the Neural Network using the experiences gathered by MCTS."""
        if len(self.replay_buffer) < self.batch_size:
            message = f"Replay buffer too small: {len(self.replay_buffer)}/{self.batch_size}. Skipping network update."
            if self.show_progress:
                tqdm.write(message)
            else:
                print(message)
            return None

        if self.show_progress:
            tqdm.write("🧠 Training Neural Network...")
        else:
            print("\n🧠 Training Neural Network...")
        self.model.train()

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        prob_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
        value_batch = torch.FloatTensor(np.array([x[2] for x in batch]).astype(np.float32)).unsqueeze(1).to(self.device)

        # We don't mask actions here because MCTS prob_batch already has 0s for illegal moves
        logits, values = self.model(state_batch)

        # Policy Loss: Cross Entropy between NN logits and MCTS probabilities
        policy_loss = -torch.sum(prob_batch * F.log_softmax(logits, dim=1), dim=1).mean()

        # Value Loss: Mean Squared Error between NN prediction and actual game outcome
        value_loss = F.mse_loss(values, value_batch)

        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        losses = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }
        message = (
            f"Loss -> Policy: {losses['policy_loss']:.4f} | "
            f"Value: {losses['value_loss']:.4f} | Total: {losses['total_loss']:.4f}"
        )
        if self.show_progress:
            tqdm.write(message)
        else:
            print(message)
        return losses

    def learn(self):
        final_epoch = self.start_epoch + self.epochs - 1
        epoch_iter = range(self.start_epoch, final_epoch + 1)
        if self.show_progress:
            epoch_iter = tqdm(
                epoch_iter,
                total=self.epochs,
                desc="AlphaZero epochs",
                unit="epoch",
                dynamic_ncols=True,
            )

        for epoch in epoch_iter:
            self.current_epoch = epoch
            if not self.show_progress:
                print(f"\n{'=' * 40}\n AlphaZero Epoch {epoch}/{final_epoch}\n{'=' * 40}")

            stats = self.self_play()
            losses = self.train_network()

            if self.show_progress:
                postfix = {
                    "epoch": f"{epoch}/{final_epoch}",
                    "games": stats["completed_games"],
                    "avg_steps": f"{stats['avg_steps']:.1f}",
                    "replay": stats["replay_buffer"],
                }
                if losses is not None:
                    postfix["loss"] = f"{losses['total_loss']:.3f}"
                epoch_iter.set_postfix(postfix, refresh=True)

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)


if __name__ == '__main__':
    trainer = AlphaZeroTrainer()
    try:
        trainer.learn()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving interrupt checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, interrupted=True)
        raise
