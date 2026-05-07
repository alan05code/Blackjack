# BJRL.py
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from blackjack_env import BlackjackEnv


# --- 1. Codifica dello stato ---
def encode_state(obs):
    """
    Converte l'osservazione [player_sum, dealer_upcard, usable_ace]
    in una tupla hashabile per la Q-table.
    """
    return (int(obs[0]), int(obs[1]), int(obs[2]))


# --- 2. Classe Agente RL ---
class BlackjackAgent:
    def __init__(self, learning_rate=0.01, discount_factor=0.99):
        self.alpha = learning_rate
        self.gamma = discount_factor

        # Q-table: dict con default = array di 2 azioni
        self.Q = defaultdict(lambda: np.zeros(2))

    # ------------------------
    # Selezione azione
    # ------------------------
    def choose_action(self, obs, epsilon=0.0):
        """
        Ritorna un'azione: 0=STAND, 1=HIT
        epsilon = probabilità di esplorazione
        """
        state = encode_state(obs)

        if np.random.rand() < epsilon:
            return np.random.randint(2)
        return np.argmax(self.Q[state])

    # ------------------------
    # Aggiornamento Q-learning
    # ------------------------
    def update(self, obs, action, reward, next_obs, done):
        s = encode_state(obs)
        ns = encode_state(next_obs)

        best_next_action = np.argmax(self.Q[ns])
        td_target = reward + self.gamma * self.Q[ns][best_next_action] * (not done)
        td_error = td_target - self.Q[s][action]

        self.Q[s][action] += self.alpha * td_error

    # ------------------------
    # Salvataggio / Caricamento
    # ------------------------
    def save(self, path="q_table.npy"):
        np.save(path, dict(self.Q))

    def load(self, path="q_table.npy"):
        data = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(lambda: np.zeros(2), data)


# --- 3. Funzione di training ---
def train_agent(
    episodes=200000,
    learning_rate=0.01,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.99997,
    min_epsilon=0.05
):
    env = BlackjackEnv(render_mode=None)
    agent = BlackjackAgent(learning_rate, discount_factor)

    print(f"[BJRL] Training su {episodes} episodi...")

    for ep in tqdm(range(episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.choose_action(obs, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("[BJRL] Training completato.")
    return agent


# --- 4. Funzione di valutazione ---
def evaluate_agent(agent, episodes=50000):
    env = BlackjackEnv(render_mode=None)
    wins = 0
    draws = 0
    losses = 0

    print(f"[BJRL] Valutazione su {episodes} episodi...")

    for _ in tqdm(range(episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            # scelta greedy -> nessuna esplorazione
            action = agent.choose_action(obs, epsilon=0.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    total = wins + losses + draws

    print("\n--- RISULTATI ---")
    print(f"Vittorie:   {wins}  ({wins/total:.2%})")
    print(f"Pareggi:    {draws}  ({draws/total:.2%})")
    print(f"Sconfitte:  {losses} ({losses/total:.2%})")

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
    }



# --- 5. API opzionale per suggerimento (per run_game.py) ---
def agent_action(agent, observation):
    """
    Restituisce la migliore azione secondo la Q-table (0=STAND, 1=HIT).
    Non modifica lo stato: è solo un suggerimento.
    """
    return agent.choose_action(observation, epsilon=0.0)
