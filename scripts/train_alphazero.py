import copy
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
import concurrent.futures
import multiprocessing as mp

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / "wallz_v2"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS, invert_action_array, flip_action_array_horizontal, flip_obs_horizontal, get_canonical_obs

def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default

def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}

def _worker_play_game(args):
    state_dict, config, game_idx = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WallzNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    env = WallzEnv()
    mcts = MCTS(model, num_simulations=config['mcts_simulations'])
    game_history = []
    seen_states = {}

    def state_key(e):
        return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())

    seen_states[state_key(env)] = 1
    terminal = False
    reward = 0.0
    step = 0

    while not terminal and step < config['max_steps']:
        if np.random.rand() < 0.25:
            mcts.num_simulations = config['mcts_simulations']
        else:
            mcts.num_simulations = max(2, config['mcts_simulations'] // 5)
            
        temp = 1.0 if step < config['temp_moves'] else 0.0
        action_probs = mcts.get_action_prob(env, temperature=temp)
        
        obs = env.get_observation()
        mask = env.get_legal_action_mask()
        
        if env.current_player == 2:
            c_obs = get_canonical_obs(obs)
            c_mask = invert_action_array(mask)
            c_probs = invert_action_array(action_probs)
            
            game_history.append((c_obs, c_mask, c_probs, env.current_player))
            game_history.append((flip_obs_horizontal(c_obs), flip_action_array_horizontal(c_mask), flip_action_array_horizontal(c_probs), env.current_player))
        else:
            game_history.append((obs, mask, action_probs, env.current_player))
            game_history.append((flip_obs_horizontal(obs), flip_action_array_horizontal(mask), flip_action_array_horizontal(action_probs), env.current_player))

        legal_mask = env.get_legal_action_mask()
        legal_actions = np.flatnonzero(legal_mask)
        probs = np.zeros(209)
        probs[legal_actions] = action_probs[legal_actions]
        
        for act in legal_actions:
            saved_p1, saved_p2, saved_cp = env.p1_pos, env.p2_pos, env.current_player
            saved_wl = env.walls_left.copy()
            saved_hw, saved_vw = env.h_walls.copy(), env.v_walls.copy()
            
            env.step(int(act))
            if seen_states.get(state_key(env), 0) >= config['rep_limit'] - 1:
                probs[act] = 0.0
                
            env.p1_pos, env.p2_pos, env.current_player = saved_p1, saved_p2, saved_cp
            env.walls_left = saved_wl
            env.h_walls, env.v_walls = saved_hw, saved_vw
                
        total_prob = probs.sum()
        if total_prob <= 0:
            probs[legal_actions] = 1.0 / len(legal_actions)
            probs /= probs.sum()
        else:
            probs /= total_prob

        if temp == 0:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        _, reward, terminal, _ = env.step(action)
        step += 1
        key = state_key(env)
        seen_states[key] = seen_states.get(key, 0) + 1

        if not terminal and seen_states[key] >= config['rep_limit']:
            break

    winner = None
    is_tiebreaker = False
    base_reward = 0.0

    if terminal and reward == 1.0:
        winner = 1 if env.current_player == 2 else 2
        base_reward = 1.0
    else:
        p1_dist = env._get_bfs_distance(env.p1_pos, 0)
        p2_dist = env._get_bfs_distance(env.p2_pos, 8)
        
        if p1_dist < p2_dist:
            winner = 1
            is_tiebreaker = True
        elif p2_dist < p1_dist:
            winner = 2
            is_tiebreaker = True
            
        if winner is not None:
            diff = abs(p1_dist - p2_dist)
            base_reward = min(1.0, diff * 0.15)

    processed = []
    total_steps_in_game = len(game_history)
    for current_step_idx, (obs, mask, p, player) in enumerate(game_history):
        if winner is None:
            z = 0.0
        else:
            sign = 1.0 if player == winner else -1.0
            if is_tiebreaker:
                z = sign * base_reward
            else:
                steps_to_end = total_steps_in_game - current_step_idx
                z = sign * base_reward * (0.99 ** steps_to_end) 
        processed.append((obs, mask, p, z))

    return processed, terminal, step


class AlphaZeroTrainer:
    def __init__(self):
        self.cpu_threads = env_int("AZ_TORCH_THREADS", 20)
        torch.set_num_threads(self.cpu_threads)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = WallzNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.epochs = env_int("AZ_EPOCHS", 100)
        self.games_per_epoch = env_int("AZ_GAMES_PER_EPOCH", 10)
        
        self.mcts_simulations = 30
        self.temperature_moves = 15 
        
        self.batch_size = env_int("AZ_BATCH_SIZE", 128)
        self.save_every = env_int("AZ_SAVE_EVERY", 1)
        self.min_terminal_games = env_int("AZ_MIN_TERMINAL_GAMES", 1)
        self.max_steps_per_game = env_int("AZ_MAX_STEPS_PER_GAME", 120)
        self.repetition_limit = env_int("AZ_REPETITION_LIMIT", 3)
        self.show_progress = env_flag("AZ_PROGRESS", True)
        
        self.replay_buffer = deque(maxlen=20000)
        self.checkpoint_dir = PACKAGE_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = self._load_latest_checkpoint() + 1
        self.current_epoch = self.start_epoch - 1

    def _load_latest_checkpoint(self) -> int:
        checkpoints = []
        for pattern in ("alphazero_epoch_*.pt", "alphazero_interrupt_epoch_*.pt"):
            for path in self.checkpoint_dir.glob(pattern):
                match = re.search(r"epoch_(\d+)", path.name)
                if match:
                    checkpoints.append((int(match.group(1)), path.stat().st_mtime, path))

        if not checkpoints:
            print("No AlphaZero checkpoint found. Starting from scratch.")
            return 0

        epoch, _, path = max(checkpoints, key=lambda item: (item[0], item[1]))
        print(f"♻️ Loading AlphaZero checkpoint: {path}")
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Resuming after epoch {epoch}.")
            return epoch
        except RuntimeError:
            print("⚠️ Checkpoint architecture mismatch! Starting from scratch.")
            return 0

    def _apply_dynamic_scheduler(self, epoch):
        if epoch <= 50:
            current_lr = 1e-3
            self.mcts_simulations = 30
            self.temperature_moves = 15 
        else:
            current_lr = 1e-4 
            self.mcts_simulations = 150 
            self.temperature_moves = 4 
            
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] != current_lr:
                param_group['lr'] = current_lr
                tqdm.write(f"⚙️ Dynamic Scheduler Triggered! Epoch {epoch}: LR -> {current_lr}, MCTS -> {self.mcts_simulations}, TempMoves -> {self.temperature_moves}")

    def save_checkpoint(self, epoch: int, interrupted: bool = False):
        name = f"alphazero_interrupt_epoch_{epoch}.pt" if interrupted else f"alphazero_epoch_{epoch}.pt"
        path = self.checkpoint_dir / name
        torch.save(self.model.state_dict(), path)
        torch.save(self.model.state_dict(), self.checkpoint_dir / "alphazero_latest.pt")
        tqdm.write(f"💾 Saved ResNet checkpoint to {path}")

    def self_play(self):
        self.model.eval()
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        config = {
            'mcts_simulations': self.mcts_simulations,
            'max_steps': self.max_steps_per_game,
            'temp_moves': self.temperature_moves,
            'rep_limit': self.repetition_limit
        }
        args_list = [(state_dict, config, i) for i in range(self.games_per_epoch)]

        num_workers = min(os.cpu_count() or 4, 10) 
        completed, adjudicated, total_steps = 0, 0, 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_play_game, args) for args in args_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parallel Self-Play"):
                processed_history, terminal, steps = future.result()
                self.replay_buffer.extend(processed_history)
                total_steps += steps
                if terminal: completed += 1
                else: adjudicated += 1

        counted = completed + adjudicated
        return {
            "completed_games": completed,
            "adjudicated_games": adjudicated,
            "avg_steps": total_steps / counted if counted else 0,
            "replay_buffer": len(self.replay_buffer),
        }

    def train_network(self, terminal_games_this_epoch: int):
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.model.train()
        training_steps = max(20, len(self.replay_buffer) // self.batch_size)
        total_policy_loss, total_value_loss = 0, 0
        
        for _ in range(training_steps):
            batch = random.sample(self.replay_buffer, self.batch_size)
            state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
            mask_batch = torch.BoolTensor(np.array([x[1] for x in batch])).to(self.device)
            prob_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
            value_batch = torch.FloatTensor(np.array([x[3] for x in batch]).astype(np.float32)).unsqueeze(1).to(self.device)

            logits, values = self.model(state_batch, action_mask=mask_batch)
            
            policy_loss = -torch.sum(prob_batch * F.log_softmax(logits, dim=1), dim=1).mean()
            value_loss = F.mse_loss(values, value_batch)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            "policy_loss": total_policy_loss / training_steps,
            "value_loss": total_value_loss / training_steps,
            "total_loss": (total_policy_loss + total_value_loss) / training_steps,
        }

    def learn(self):
        final_epoch = self.start_epoch + self.epochs - 1
        epoch_iter = tqdm(range(self.start_epoch, 101), total=(101 - self.start_epoch), desc="AlphaZero ResNet", dynamic_ncols=True)

        for epoch in epoch_iter:
            self.current_epoch = epoch
            self._apply_dynamic_scheduler(epoch)
            
            stats = self.self_play()
            losses = self.train_network(stats["completed_games"])

            postfix = {
                "games": stats["completed_games"],
                "replay": stats["replay_buffer"],
            }
            if losses is not None:
                postfix["loss"] = f"{losses['total_loss']:.3f}"
            epoch_iter.set_postfix(postfix, refresh=True)

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)

if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    trainer = AlphaZeroTrainer()
    try:
        trainer.learn()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving ResNet checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, interrupted=True)