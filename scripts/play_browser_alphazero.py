import argparse
import sys
import types
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / "wallz_v2"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS
from play_browser_v2 import BrowserAgentV2

class AlphaZeroPredictor:
    def __init__(self, checkpoint: Path, device: str = "cpu", mcts_simulations: int = 80):
        self.device = torch.device(device)
        self.model = WallzNet().to(self.device)
        
        # Загружаем веса
        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        
        self.mcts_simulations = mcts_simulations
        print(f"🧠 [AlphaZero] Мозг загружен: {checkpoint.name}")
        print(f"🔍 [AlphaZero] Глубина просчета (MCTS): {self.mcts_simulations} симуляций")

    def predict(self, env, mask):
        mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
        try:
            # Используем temperature=0.0 для выбора 100% лучшего хода по мнению MCTS
            probs = mcts.get_action_prob(env, temperature=0.0)
            probs = np.asarray(probs, dtype=np.float64)
            probs[~mask] = 0.0
            
            if probs.sum() > 0:
                action = int(np.argmax(probs))
                return action
        except Exception as e:
            print(f"[AlphaZero Error] Ошибка генерации хода: {e}")
        
        # Если что-то пошло не так, берем первое доступное легальное действие
        valid_actions = np.where(mask)[0]
        return int(valid_actions[0]) if len(valid_actions) > 0 else -1

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Assistant Wrapper")
    parser.add_argument("--checkpoint", required=True, help="Путь к .pt файлу")
    parser.add_argument("--device", default="cpu", help="cpu или mps")
    parser.add_argument("--mcts-simulations", type=int, default=80)
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Файл не найден: {checkpoint_path}")
        return

    # 1. Инициализируем чистый MCTS-мозг
    predictor = AlphaZeroPredictor(
        checkpoint=checkpoint_path, 
        device=args.device, 
        mcts_simulations=args.mcts_simulations
    )
    
    # 2. Инициализируем нашего визуального ассистента из play_browser_v2
    agent = BrowserAgentV2(
        model_path=Path("dummy_path"), # Заглушка, так как мы подменим логику
        max_turn_seconds=15.0, # Даем чуть больше времени на рендер MCTS
        allow_backward=True
    )
    
    # 3. Подменяем метод предсказания PPO на наш крутой AlphaZero
    def custom_predict_action(self, masks):
        return predictor.predict(self.env, masks)
        
    agent.predict_action = types.MethodType(custom_predict_action, agent)
    
    # 4. Запускаем!
    agent.run("https://wallz.gg/")

if __name__ == "__main__":
    main()