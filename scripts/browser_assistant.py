import sys
import asyncio
import numpy as np
import torch
import re
from pathlib import Path
from playwright.async_api import async_playwright

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS
from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.env.action_space import action_to_move, move_to_action

class WallzAssistant:
    def __init__(self, model_path):
        self.device = torch.device('cpu') 
        print(f"Loading AlphaZero model on: {self.device}")
        
        self.model = WallzNet(num_channels=10, num_res_blocks=10, num_hidden=128).to(self.device) 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        try:
            dummy_obs = torch.zeros(1, 10, 9, 9).to(self.device)
            dummy_mask = torch.ones(1, 209, dtype=torch.bool).to(self.device)
            self.model = torch.jit.trace(self.model, (dummy_obs, dummy_mask))
            print("⚡ PyTorch JIT Compilation successful! Model is optimized.")
        except Exception as e:
            print(f"⚠️ JIT Compilation skipped due to error: {e}")

        self.mcts = MCTS(self.model, num_simulations=50)

    async def extract_board_state(self, page):
        env = WallzEnv()
        seen_states = {}
        player_positions = {1: [(8, 4)], 2: [(0, 4)]}
        
        def state_key(e): 
            return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())
        seen_states[state_key(env)] = 1
        
        try:
            elements = await page.locator('ol > li').all_inner_texts()
            history_text = " ".join(elements).lower()
            
            if not history_text:
                return env, seen_states, player_positions

            pattern = r'\b(?:([hv])\s*[-_]?\s*)?([a-i])\s*[-_]?\s*([1-9])(?:\s*[-_]?\s*([hv]))?\b'
            moves = re.findall(pattern, history_text)
            
            col_map = {'i': 0, 'h': 1, 'g': 2, 'f': 3, 'e': 4, 'd': 5, 'c': 6, 'b': 7, 'a': 8}
            row_map = {'9': 0, '8': 1, '7': 2, '6': 3, '5': 4, '4': 5, '3': 6, '2': 7, '1': 8}

            for m in moves:
                w_prefix, c_char, r_char, w_suffix = m
                w_char = w_prefix or w_suffix
                
                col_idx = col_map[c_char]
                row_idx = row_map[r_char]
                
                curr_p = env.current_player
                
                if not w_char:
                    action = move_to_action('MOVE', row_idx, col_idx)
                    player_positions[curr_p].append((row_idx, col_idx))
                else:
                    wall_r = row_idx - 1
                    wall_c = col_idx - 1
                    if wall_r < 0 or wall_r > 7 or wall_c < 0 or wall_c > 7:
                        continue
                    action = move_to_action(f'WALL_{w_char.upper()}', wall_r, wall_c)
                        
                env.step(action)
                key = state_key(env)
                seen_states[key] = seen_states.get(key, 0) + 1
                
        except Exception as e:
            print(f"❌ Error parsing board history: {e}")
            
        return env, seen_states, player_positions

    async def run(self):
        print("🚀 Booting Auto-Stealth Visual Assistant with Dynamic Perspective...")
        
        profile_dir = ROOT_DIR / "wallz_v2" / "browser_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=False, 
                args=['--disable-blink-features=AutomationControlled'],
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto("https://wallz.gg")
            print("\n✅ Browser open! UI injected.")
            
            status_js = """
            () => {
                if (!document.getElementById('az-widget')) {
                    const widget = document.createElement('div');
                    widget.id = 'az-widget';
                    widget.style.position = 'fixed';
                    widget.style.bottom = '15px';
                    widget.style.right = '15px';
                    widget.style.zIndex = '9999999';
                    widget.style.backgroundColor = '#0f172a';
                    widget.style.border = '2px solid #334155';
                    widget.style.borderRadius = '10px';
                    widget.style.padding = '12px 16px';
                    widget.style.boxShadow = '0 10px 15px -3px rgba(0, 0, 0, 0.7)';
                    widget.style.fontFamily = 'monospace';
                    widget.style.fontSize = '13px';
                    widget.style.pointerEvents = 'auto';
                    
                    const textDiv = document.createElement('div');
                    textDiv.id = 'az-text';
                    textDiv.style.color = '#94a3b8';
                    textDiv.style.fontWeight = 'bold';
                    textDiv.style.marginBottom = '8px';
                    textDiv.innerText = '🟢 AI Active';
                    widget.appendChild(textDiv);

                    const controls = document.createElement('div');
                    controls.style.display = 'flex';
                    controls.style.alignItems = 'center';
                    controls.style.gap = '8px';
                    controls.style.marginBottom = '10px';
                    
                    const label = document.createElement('span');
                    label.innerText = 'Power:';
                    label.style.color = '#64748b';
                    
                    const select = document.createElement('select');
                    select.id = 'az-sim-select';
                    select.style.appearance = 'select';
                    select.style.webkitAppearance = 'select';
                    select.style.backgroundColor = '#1e293b';
                    select.style.color = '#38bdf8';
                    select.style.border = '1px solid #475569';
                    select.style.borderRadius = '4px';
                    select.style.padding = '4px 8px';
                    select.style.cursor = 'pointer';
                    
                    const options = [
                        {val: 2, text: '2 (Blitz)'},
                        {val: 10, text: '10 (Fast)'},
                        {val: 50, text: '50 (Normal)'},
                        {val: 150, text: '150 (Strong)'}
                    ];
                    
                    options.forEach(opt => {
                        const option = document.createElement('option');
                        option.value = opt.val;
                        option.innerText = opt.text;
                        if (opt.val === 50) option.selected = true;
                        select.appendChild(option);
                    });
                    
                    controls.appendChild(label);
                    controls.appendChild(select);
                    widget.appendChild(controls);

                    const autoClickDiv = document.createElement('div');
                    autoClickDiv.style.display = 'flex';
                    autoClickDiv.style.alignItems = 'center';
                    autoClickDiv.style.gap = '6px';
                    
                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.id = 'az-autoclick';
                    cb.style.appearance = 'checkbox';
                    cb.style.webkitAppearance = 'checkbox';
                    cb.style.width = '14px';
                    cb.style.height = '14px';
                    cb.style.cursor = 'pointer';
                    
                    const cbLabel = document.createElement('label');
                    cbLabel.innerText = 'Auto-Click (Bot Plays)';
                    cbLabel.style.color = '#f87171';
                    cbLabel.style.cursor = 'pointer';
                    cbLabel.htmlFor = 'az-autoclick';
                    
                    autoClickDiv.appendChild(cb);
                    autoClickDiv.appendChild(cbLabel);
                    widget.appendChild(autoClickDiv);
                    
                    document.body.appendChild(widget);
                }
            }
            """
            
            last_processed_moves = -1
            col_map = {'i': 0, 'h': 1, 'g': 2, 'f': 3, 'e': 4, 'd': 5, 'c': 6, 'b': 7, 'a': 8}
            row_map = {'9': 0, '8': 1, '7': 2, '6': 3, '5': 4, '4': 5, '3': 6, '2': 7, '1': 8}
            col_map_inv = {v: k for k, v in col_map.items()}
            row_map_inv = {v: k for k, v in row_map.items()}
            move_pattern = r'\b(?:[hv]\s*[-_]?\s*)?[a-i]\s*[-_]?\s*[1-9](?:\s*[-_]?\s*[hv])?\b'

            while True:
                try:
                    await page.evaluate(status_js)
                    is_my_turn = await page.locator("text='Your turn'").is_visible()
                    
                    if is_my_turn:
                        history_text = " ".join(await page.locator('ol > li').all_inner_texts()).lower()
                        current_moves = len(re.findall(move_pattern, history_text))
                        
                        if current_moves != last_processed_moves:
                            await page.evaluate("document.getElementById('az-text').innerText = '⏳ Thinking...'")
                            
                            env, seen_states, player_positions = await self.extract_board_state(page)
                            
                            ui_sims = await page.evaluate("() => { const el = document.getElementById('az-sim-select'); return el ? parseInt(el.value) : null; }")
                            if ui_sims and ui_sims != self.mcts.num_simulations:
                                self.mcts.num_simulations = ui_sims
                                print(f"⚙️ AI Strength updated to {ui_sims} simulations")
                            
                            action_probs = self.mcts.get_action_prob(env, temperature=0.1)
                            
                            legal_mask = env.get_legal_action_mask()
                            legal_actions = np.flatnonzero(legal_mask)
                            probs = np.zeros(209)
                            probs[legal_actions] = action_probs[legal_actions] + 1e-6
                            
                            my_id = env.current_player
                            recent_history = set(player_positions[my_id][-4:])
                            def state_key(e): 
                                return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())
                            
                            # --- НОВЫЙ БЛОК: Замеряем текущую дистанцию до финиша ---
                            target_row = 0 if my_id == 1 else 8
                            current_pos = env.p1_pos if my_id == 1 else env.p2_pos
                            current_dist = env._get_bfs_distance(current_pos, target_row)
                            
                            for act in legal_actions:
                                saved_p1, saved_p2, saved_cp = env.p1_pos, env.p2_pos, env.current_player
                                saved_wl = env.walls_left.copy()
                                saved_hw, saved_vw = env.h_walls.copy(), env.v_walls.copy()
                                
                                move_type, (r, c) = action_to_move(act)
                                
                                env.step(int(act))
                                key = state_key(env)
                                
                                # --- ЭВРИСТИКА ПРОГРЕССИИ (Лечим "танцы" на месте) ---
                                if move_type == 'MOVE':
                                    new_pos = env.p1_pos if my_id == 1 else env.p2_pos
                                    new_dist = env._get_bfs_distance(new_pos, target_row)
                                    
                                    if new_dist < current_dist:
                                        probs[act] *= 2.0    # Мощно поощряем шаг вперед
                                    elif new_dist == current_dist:
                                        probs[act] *= 0.1    # Штрафуем шаги вбок (танцы)
                                    else:
                                        probs[act] *= 0.01   # Жестко штрафуем шаги назад
                                        
                                    # Жесткий запрет на возврат в недавние позиции
                                    if (r, c) in recent_history:
                                        probs[act] *= 0.001 
                                
                                # Обрезаем глобальное зацикливание
                                if seen_states.get(key, 0) >= 2:
                                    probs[act] = 0.0
                                        
                                env.p1_pos, env.p2_pos, env.current_player = saved_p1, saved_p2, saved_cp
                                env.walls_left = saved_wl
                                env.h_walls, env.v_walls = saved_hw, saved_vw

                            # Если все хорошие ходы заблокировались (чтобы не словить краш)
                            if probs.sum() > 0:
                                best_action = int(np.argmax(probs))
                            else:
                                fallback_probs = np.zeros(209)
                                fallback_probs[legal_actions] = action_probs[legal_actions] + 1e-6
                                best_action = int(np.argmax(fallback_probs))
                            
                            move_type, (r, c) = action_to_move(best_action)
                            
                            if move_type == 'MOVE':
                                hint_c_char = col_map_inv[c]
                                hint_r_char = row_map_inv[r]
                            elif move_type in ['WALL_V', 'WALL_H']:
                                hint_c_char = col_map_inv[c + 1]
                                hint_r_char = row_map_inv[r + 1]
                            
                            highlight_js = f"""
                            () => {{
                                const svg = document.querySelector('svg[aria-label="Wallz board"]');
                                if (!svg) return null;

                                let oldHighlight = document.getElementById('az-visual-hint');
                                if (oldHighlight) oldHighlight.remove();

                                const highlight = document.createElementNS("http://www.w3.org/2000/svg", "g");
                                highlight.id = 'az-visual-hint';
                                highlight.style.pointerEvents = 'none';

                                const moveType = '{move_type}';
                                const cChar = '{hint_c_char}';
                                const rChar = '{hint_r_char}';

                                const cols = Array.from(document.querySelectorAll('text[y="660"]')).map(t => t.textContent.trim().toLowerCase());
                                const rows = Array.from(document.querySelectorAll('text[x="-24"]')).map(t => t.textContent.trim().toLowerCase());

                                const visual_c_idx = cols.indexOf(cChar);
                                const visual_r_idx = rows.indexOf(rChar);

                                if (visual_c_idx === -1 || visual_r_idx === -1) return null;

                                let vis_c = visual_c_idx, vis_r = visual_r_idx;
                                let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                                
                                let rel_x = 0, rel_y = 0;
                                
                                if (moveType === 'MOVE') {{
                                    rel_x = vis_c * 72 + 30;
                                    rel_y = vis_r * 72 + 30;
                                    
                                    rect.setAttribute('x', vis_c * 72);
                                    rect.setAttribute('y', vis_r * 72);
                                    rect.setAttribute('width', 60);
                                    rect.setAttribute('height', 60);
                                    rect.setAttribute('rx', 9);
                                    rect.setAttribute('fill', 'rgba(56, 189, 248, 0.4)');
                                    rect.setAttribute('stroke', '#38bdf8');
                                    rect.setAttribute('stroke-width', 4);
                                }} else {{
                                    vis_c -= 1;
                                    vis_r -= 1;
                                    
                                    rel_x = vis_c * 72 + 66;
                                    rel_y = vis_r * 72 + 66;
                                    
                                    if (moveType === 'WALL_V') {{
                                        rect.setAttribute('x', (vis_c * 72) + 60);
                                        rect.setAttribute('y', vis_r * 72);
                                        rect.setAttribute('width', 12);
                                        rect.setAttribute('height', 132);
                                    }} else {{
                                        rect.setAttribute('x', vis_c * 72);
                                        rect.setAttribute('y', (vis_r * 72) + 60);
                                        rect.setAttribute('width', 132);
                                        rect.setAttribute('height', 12);
                                    }}
                                    
                                    rect.setAttribute('rx', 5);
                                    rect.setAttribute('fill', 'rgba(250, 204, 21, 0.8)');
                                    rect.setAttribute('stroke', '#facc15');
                                }}

                                const animate = document.createElementNS("http://www.w3.org/2000/svg", "animate");
                                animate.setAttribute('attributeName', 'opacity');
                                animate.setAttribute('values', '0.3; 1; 0.3');
                                animate.setAttribute('dur', '1s');
                                animate.setAttribute('repeatCount', 'indefinite');
                                rect.appendChild(animate);

                                highlight.appendChild(rect);
                                svg.appendChild(highlight);
                                
                                return {{ rel_x: rel_x, rel_y: rel_y }};
                            }}
                            """
                            coords = await page.evaluate(highlight_js)
                            await page.evaluate("document.getElementById('az-text').innerText = '👑 Move Ready!'")
                            print(f"Move {current_moves + 1} -> {move_type} at {hint_c_char}{hint_r_char}")
                            
                            auto_click = await page.evaluate("() => { const el = document.getElementById('az-autoclick'); return el ? el.checked : false; }")
                            if auto_click and coords:
                                svg = page.locator('svg[aria-label="Wallz board"]')
                                svg_box = await svg.bounding_box()
                                
                                if svg_box:
                                    target_abs_x = svg_box['x'] + coords['rel_x']
                                    target_abs_y = svg_box['y'] + coords['rel_y']
                                    
                                    if move_type == 'MOVE':
                                        await page.mouse.move(target_abs_x, target_abs_y, steps=10)
                                        await asyncio.sleep(0.1)
                                        await page.mouse.click(target_abs_x, target_abs_y)
                                    else:
                                        btn_prefix = "Drag a vertical wall" if move_type == "WALL_V" else "Drag a horizontal wall"
                                        tray_btn = page.locator(f'button[aria-label^="{btn_prefix}"]')
                                        
                                        if await tray_btn.count() > 0:
                                            btn_box = await tray_btn.first.bounding_box()
                                            if btn_box:
                                                start_x = btn_box['x'] + btn_box['width'] / 2
                                                start_y = btn_box['y'] + btn_box['height'] / 2
                                                
                                                await page.mouse.move(start_x, start_y, steps=10)
                                                await asyncio.sleep(0.1)
                                                await page.mouse.down()
                                                await asyncio.sleep(0.2)
                                                await page.mouse.move(target_abs_x, target_abs_y, steps=15)
                                                await asyncio.sleep(0.2)
                                                await page.mouse.up()
                                                await page.mouse.move(0, 0)
                                        
                            last_processed_moves = current_moves
                    else:
                        await page.evaluate("""() => {
                            let oldHighlight = document.getElementById('az-visual-hint');
                            if (oldHighlight) oldHighlight.remove();
                            let status = document.getElementById('az-text');
                            if (status && status.innerText !== '🟢 AI Active') status.innerText = '🟢 AI Active';
                        }""")
                            
                except Exception as e:
                    pass
                await asyncio.sleep(0.05)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    CHECKPOINT = ROOT_DIR / "wallz_v2" / "checkpoints" / "alphazero_latest.pt"
    
    if not CHECKPOINT.exists():
        print(f"Model not found at {CHECKPOINT}")
        sys.exit(1)
        
    assistant = WallzAssistant(CHECKPOINT)
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")