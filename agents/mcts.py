import math
import numpy as np
import torch

def invert_action_array(arr):
    inverted = np.zeros_like(arr)
    inverted[:81] = np.rot90(arr[:81].reshape(9, 9), k=2).flatten()
    inverted[81:145] = np.rot90(arr[81:145].reshape(8, 8), k=2).flatten()
    inverted[145:209] = np.rot90(arr[145:209].reshape(8, 8), k=2).flatten()
    return inverted

def flip_action_array_horizontal(arr):
    flipped = np.zeros_like(arr)
    flipped[:81] = np.fliplr(arr[:81].reshape(9, 9)).flatten()
    flipped[81:145] = np.fliplr(arr[81:145].reshape(8, 8)).flatten()
    flipped[145:209] = np.fliplr(arr[145:209].reshape(8, 8)).flatten()
    return flipped

def flip_obs_horizontal(obs):
    flipped = np.zeros_like(obs)
    for i in [0, 1, 4, 5, 8, 9]:
        if i < obs.shape[0]:
            flipped[i] = np.fliplr(obs[i])
    flipped[2, :8, :8] = np.fliplr(obs[2, :8, :8])
    flipped[3, :8, :8] = np.fliplr(obs[3, :8, :8])
    flipped[6] = obs[6]
    flipped[7] = obs[7]
    return flipped

def get_canonical_obs(obs):
    rotated = np.zeros_like(obs)
    for i in [0, 1, 4, 5, 8, 9]:
        if i < obs.shape[0]:
            rotated[i] = np.rot90(obs[i], k=2)
    rotated[2, :8, :8] = np.rot90(obs[2, :8, :8], k=2)
    rotated[3, :8, :8] = np.rot90(obs[3, :8, :8], k=2)
    rotated[6] = obs[6]
    rotated[7] = obs[7]
    return rotated

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, num_simulations=25, c_puct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = next(model.parameters()).device

    def get_action_prob(self, env, temperature=1.0):
        root = Node(prior=1.0)
        self._expand(root, env)

        legal_actions = list(root.children.keys())
        if len(legal_actions) > 0:
            dirichlet_alpha = 10.0 / max(1, len(legal_actions)) 
            epsilon = 0.25
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_actions))
            for i, action in enumerate(legal_actions):
                root.children[action].prior = root.children[action].prior * (1 - epsilon) + noise[i] * epsilon

        for _ in range(self.num_simulations):
            node = root
            # ИСПОЛЬЗУЕМ БЫСТРОЕ КОПИРОВАНИЕ
            sim_env = env.clone() 
            search_path = [node]

            while len(node.children) > 0:
                action, node = self._select_child(node)
                _, reward, terminal, _ = sim_env.step(action)
                search_path.append(node)
                if terminal:
                    break

            if not terminal:
                value = self._expand(node, sim_env)
            else:
                value = -reward

            self._backpropagate(search_path, value)

        action_visits = {a: child.visit_count for a, child in root.children.items()}
        actions = list(action_visits.keys())
        counts = list(action_visits.values())
        
        if temperature == 0:
            best_action = actions[np.argmax(counts)]
            probs = np.zeros(209)
            probs[best_action] = 1.0
            return probs

        counts = np.array(counts) ** (1.0 / temperature)
        probs = counts / np.sum(counts)
        
        full_probs = np.zeros(209)
        for a, p in zip(actions, probs):
            full_probs[a] = p
            
        return full_probs

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            if child.visit_count > 0:
                q_value = -child.value()
            else:
                q_value = -node.value() - 0.1

            u_value = self.c_puct * child.prior * math.sqrt(max(1, node.visit_count)) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node, env):
        obs = env.get_observation()
        mask = env.get_legal_action_mask()
        
        if env.current_player == 2:
            obs = get_canonical_obs(obs)
            mask = invert_action_array(mask)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs_tensor, mask_tensor)
            logits = logits.squeeze(0).cpu().numpy()
            value = value.item()
            
        if env.current_player == 2:
            logits = invert_action_array(logits)
            
        abs_mask = env.get_legal_action_mask()
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits * abs_mask
        
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs /= sum_probs
        else:
            probs = abs_mask / np.sum(abs_mask)

        legal_actions = np.where(abs_mask)[0]
        for action in legal_actions:
            node.children[action] = Node(prior=probs[action])
            
        return value

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value