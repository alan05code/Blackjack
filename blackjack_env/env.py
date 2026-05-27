"""Ambiente Gymnasium con supporto grafico opzionale per Blackjack."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium import Env, spaces

from .game import BlackjackGame
from .rendering import BlackjackRenderer


class BlackjackEnv(Env):
    """Ambiente Blackjack compatibile con Gymnasium (azioni discrete Hit/Stand)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    HIT = 1
    STAND = 0

    def __init__(
        self,
        *,
        natural_payout: float = 1.5,
        num_decks: int = 1,
        dealer_hits_soft_17: bool = False,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.render_mode = render_mode
        self.game = BlackjackGame(
            num_decks=num_decks,
            natural_payout=natural_payout,
            dealer_hits_soft_17=dealer_hits_soft_17,
            seed=seed,
        )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0], dtype=np.int32),
            high=np.array([31, 11, 1], dtype=np.int32),
            dtype=np.int32,
        )
        self._renderer: BlackjackRenderer | None = None
        self._latest_info: Dict[str, object] = {}
        self._last_message = "Nuova mano"
        self._auto_resolve = False

    # ------------------------------------------------------------------ #
    # API Gymnasium
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: Dict[str, object] | None = None) -> Tuple[np.ndarray, Dict[str, object]]:
        super().reset(seed=seed)
        _ = options  # Placeholder per future parametri opzionali
        if seed is not None:
            self.game._rng.seed(seed)
        self.game.reset_round()
        self._auto_resolve = self.game.is_blackjack(self.game.player_hand) or self.game.is_blackjack(self.game.dealer_hand)
        self._last_message = "Nuova mano: scegli HIT (1) o STAND (0)."
        obs = self._build_observation()
        info = self._build_info(reveal=False)
        self._latest_info = info
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        """
        Esegue un passo nell'ambiente.

        Args:
            action: Azione da eseguire (0=STAND, 1=HIT). Deve essere sempre fornita.

        Returns:
            Tuple (observation, reward, terminated, truncated, info)

        Note:
            Il modello decisionale deve essere chiamato esternamente e passare
            l'azione qui, come farebbe un giocatore manuale.
        """

        if not self.action_space.contains(action):
            raise ValueError("Azione non valida. Usa 0=STAND, 1=HIT.")

        terminated = False
        truncated = False
        reward = 0.0

        if self._auto_resolve:
            terminated = True
            reward = self._evaluate_round()
            self._last_message = "Blackjack naturale." if reward > 0 else "Pareggio o blackjack dealer."
            self._auto_resolve = False
        elif action == self.HIT:
            card = self.game.player_hit()
            total = self.game.hand_value(self.game.player_hand)
            self._last_message = f"Hai pescato {card.label()} (totale {total})."
            if total > 21:
                terminated = True
                reward = -1.0
                self._last_message = "Sei sballato."
        elif action == self.STAND:
            self.game.dealer_play()
            terminated = True
            reward = self._evaluate_round()
            self._last_message = "Round completo."

        if terminated:
            self.game.round_over = True

        obs = self._build_observation()
        info = self._build_info(reveal=terminated)
        self._latest_info = info
        if self.render_mode is not None:
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        if self.render_mode is None:
            raise RuntimeError("render_mode non impostata. Passa 'human' o 'rgb_array' in __init__.")
        if self._renderer is None:
            self._renderer = BlackjackRenderer(auto_display=self.render_mode == "human")
        return self._renderer.render(self._latest_info, mode=self.render_mode, message=self._last_message)

    def close(self) -> None:
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    def _build_observation(self) -> np.ndarray:
        player_sum = self.game.hand_value(self.game.player_hand)
        dealer_upcard = self.game.upcard_value()
        usable_ace = int(self.game.has_usable_ace(self.game.player_hand))
        obs = np.array([player_sum, dealer_upcard, usable_ace], dtype=np.int32)
        return obs

    def _build_info(self, *, reveal: bool) -> Dict[str, object]:
        state = self.game.public_state(reveal_dealer=reveal)
        state.update(
            {
                "player_blackjack": self.game.is_blackjack(self.game.player_hand),
                "dealer_blackjack": self.game.is_blackjack(self.game.dealer_hand),
                "status": self._last_message,
            }
        )
        return state

    def _evaluate_round(self) -> float:
        player = self.game.hand_value(self.game.player_hand)
        dealer = self.game.hand_value(self.game.dealer_hand)

        player_blackjack = self.game.is_blackjack(self.game.player_hand)
        dealer_blackjack = self.game.is_blackjack(self.game.dealer_hand)

        if player_blackjack and dealer_blackjack:
            return 0.0
        if player_blackjack:
            return self.game.natural_payout
        if dealer_blackjack:
            return -1.0
        if player > 21:
            return -1.0
        if dealer > 21:
            return 1.0
        if player > dealer:
            return 1.0
        if player < dealer:
            return -1.0
        return 0.0

    def _should_end_round(self) -> bool:
        """Determina se, dato lo stato corrente, il round è concluso (blackjack o bust)."""
        player_total = self.game.hand_value(self.game.player_hand)
        dealer_total = self.game.hand_value(self.game.dealer_hand)
        if self.game.is_blackjack(self.game.player_hand) or self.game.is_blackjack(self.game.dealer_hand):
            return True
        if player_total > 21 or dealer_total > 21:
            return True
        # Vision mode: dealer past initial deal (3+ cards) or already at stand value (>=17)
        # implica che il giocatore ha già stato e il dealer ha completato la propria mano.
        if len(self.game.dealer_hand) >= 3:
            return True
        if len(self.game.dealer_hand) >= 2 and dealer_total >= 17:
            return True
        return False