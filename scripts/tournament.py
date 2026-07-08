import sys
import torch
import numpy as np
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS
from wallz_v2.env.wallz_env import WallzEnv

def run_tournament(model1_path, model2_path, num_games=10):
    device = torch.device('cpu')
    print(f"Loading Model 1: {model1_path}")
    print(f"Loading Model 2: {model2_path}")
    
    # Инициализация обеих моделей с теми же параметрами, что и при обучении
    model1 = WallzNet(num_channels=10, num_res_blocks=10, num_hidden=128).to(device)
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.eval()
    
    model2 = WallzNet(num_channels=10, num_res_blocks=10, num_hidden=128).to(device)
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.eval()
    
    mcts1 = MCTS(model1, num_simulations=50)
    mcts2 = MCTS(model2, num_simulations=50)
    
    results = {"m1_wins": 0, "m2_wins": 0, "draws": 0}
    
    for i in range(num_games):
        env = WallzEnv()
        terminal = False
        
        # Модели меняются местами для честности (первая половина игр: М1 первый, вторая: М2 первый)
        is_m1_first = (i % 2 == 0)
        mcts_p1 = mcts1 if is_m1_first else mcts2
        mcts_p2 = mcts2 if is_m1_first else mcts1
        
        while not terminal:
            current_mcts = mcts_p1 if env.current_player == 1 else mcts_p2
            action_probs = current_mcts.get_action_prob(env, temperature=0.0)
            action = int(np.argmax(action_probs))
            _, reward, terminal, _ = env.step(action)
            
        # reward 1.0 -> победил игрок 1. Если current_player стал 2, значит 1-й сходил и победил
        winner = 1 if (reward == 1.0 and env.current_player == 2) else 2
        
        # Подсчет очков
        if (winner == 1 and is_m1_first) or (winner == 2 and not is_m1_first):
            results["m1_wins"] += 1
            print(f"Game {i+1}: Model 1 wins")
        else:
            results["m2_wins"] += 1
            print(f"Game {i+1}: Model 2 wins")

    print(f"\nTournament Final: M1 wins: {results['m1_wins']}, M2 wins: {results['m2_wins']}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/tournament.py <path_to_model1> <path_to_model2>")
        sys.exit(1)
        
    m1 = Path(sys.argv[1])
    m2 = Path(sys.argv[2])
    
    run_tournament(m1, m2, num_games=10)