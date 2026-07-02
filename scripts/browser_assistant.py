import sys
import asyncio
import numpy as np
import torch
from pathlib import Path
from playwright.async_api import async_playwright

# Add project root to sys.path so we can import our modules
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS
from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.env.action_space import action_to_move, move_to_action

class WallzAssistant:
    def __init__(self, model_path):
        # Auto-detect Apple Silicon (MPS), CUDA, or CPU
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading AlphaZero model on: {self.device}")
        
        # Load your trained AlphaZero checkpoint
        self.model = WallzNet(num_channels=8) # Update num_channels if you changed your state representation
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Initialize MCTS. Increase num_simulations (e.g., 100-200) for stronger Elo play!
        self.mcts = MCTS(self.model, num_simulations=50) 

    async def extract_board_state(self, page):
        """
        Extracts the board state by reading the algebraic move history 
        and replaying it in a fresh internal environment.
        """
        env = WallzEnv()
        
        try:
            # 1. Grab all the moves played in the game so far chronologically
            # We target the spans containing the actual text like "e2", "e8", "e4h"
            move_elements = await page.locator('ol > li span[style*="color: var(--color-ink)"]').all_inner_texts()
            
            if not move_elements:
                print("Game hasn't started or no moves found yet. Displaying initial state.")
                return env

            # 2. Coordinate Mapping dictionaries based on Wallz.gg SVG axes
            # x-axis: 'i' is left (0), 'a' is right (8)
            col_map = {'i': 0, 'h': 1, 'g': 2, 'f': 3, 'e': 4, 'd': 5, 'c': 6, 'b': 7, 'a': 8}
            # y-axis: '9' is top (0), '1' is bottom (8)
            row_map = {'9': 0, '8': 1, '7': 2, '6': 3, '5': 4, '4': 5, '3': 6, '2': 7, '1': 8}

            # 3. Replay history to sync the internal state
            for move_str in move_elements:
                move_str = move_str.strip().lower()
                if not move_str: 
                    continue

                col_idx = col_map[move_str[0]]
                row_idx = row_map[move_str[1]]
                
                if len(move_str) == 2:
                    action = move_to_action('MOVE', row_idx, col_idx)
                elif len(move_str) == 3:
                    if move_str[2] == 'h':
                        action = move_to_action('WALL_H', row_idx, col_idx)
                    elif move_str[2] == 'v':
                        action = move_to_action('WALL_V', row_idx, col_idx)
                
                # Advance the internal engine one step
                env.step(action)
                
            print(f"✅ Synced board state successfully ({len(move_elements)} moves played).")
            
        except Exception as e:
            print(f"❌ Error parsing board history: {e}")
            
        return env

    async def run(self):
        print("🚀 Booting Stealth Assistant...")
        async with async_playwright() as p:
            # Launch standard Chromium (not headless, so you can play)
            browser = await p.chromium.launch(
                headless=False, 
                args=['--disable-blink-features=AutomationControlled']
            )
            
            # Create a stealthy context to avoid basic bot-detection
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            await page.goto("https://wallz.gg")
            print("\n✅ Browser open! Log into your account and start a ranked match.")
            print("When it is your turn, switch back to this terminal and press [ENTER] to get the AlphaZero move.")
            
            while True:
                user_input = await asyncio.to_thread(input, "\n[ENTER] to calculate move (or type 'q' to quit): ")
                if user_input.lower() == 'q':
                    break
                    
                print("🧠 Analyzing board state...")
                env = await self.extract_board_state(page)
                
                print("⏳ Running Monte Carlo Tree Search...")
                # Get the absolute best move from AlphaZero (temperature=0 means strict best move)
                action_probs = self.mcts.get_action_prob(env, temperature=0.0)
                best_action = np.argmax(action_probs)
                
                move_type, (r, c) = action_to_move(best_action)
                
                print("\n==================================")
                print(f"👑 TOP ELO MOVE SUGGESTION: ")
                if move_type == 'MOVE':
                    print(f"👉 MOVE PAWN to Row {r}, Col {c}")
                elif move_type == 'WALL_H':
                    print(f"🧱 PLACE HORIZONTAL WALL at Row {r}, Col {c}")
                elif move_type == 'WALL_V':
                    print(f"🧱 PLACE VERTICAL WALL at Row {r}, Col {c}")
                print("==================================\n")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    # Point this to your best AlphaZero checkpoint
    CHECKPOINT = ROOT_DIR / "wallz_v2" / "checkpoints" / "alphazero_latest.pt"
    
    if not CHECKPOINT.exists():
        print(f"Model not found at {CHECKPOINT}")
        print("Please check the path or run the training script first.")
        sys.exit(1)
        
    assistant = WallzAssistant(CHECKPOINT)
    asyncio.run(assistant.run())