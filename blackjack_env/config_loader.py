"""
Helper per caricare configurazione e inizializzare l'ambiente con i modelli appropriati.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .env import BlackjackEnv

import config


def load_config() -> dict:
    """
    Carica la configurazione da config.py.

    Returns:
        Dict con tutte le variabili di configurazione.

    Raises:
        ImportError: Se config.py non esiste.
    """
    return {
        "USE_VISION_RECOGNITION": config.USE_VISION_RECOGNITION,
        "NUM_DECKS": config.NUM_DECKS,
        "NATURAL_PAYOUT": config.NATURAL_PAYOUT,
        "DEALER_HITS_SOFT_17": config.DEALER_HITS_SOFT_17,
        "RANDOM_SEED": config.RANDOM_SEED,
        "RENDER_MODE": config.RENDER_MODE,
        "WINDOW_WIDTH": config.WINDOW_WIDTH,
        "WINDOW_HEIGHT": config.WINDOW_HEIGHT,
        "FPS": config.FPS,
        "POLICY_PATH": getattr(config, "POLICY_PATH", "blackjack_env/model/policy.npy"),
        "VISION_STATE_PATH": getattr(config, "VISION_STATE_PATH", "blackjack_env/tmp/vision_state.json"),
    }


def create_env_from_config() -> "BlackjackEnv":
    """Crea un'istanza di BlackjackEnv dalla configurazione in config.py."""
    from .env import BlackjackEnv

    cfg = load_config()
    return BlackjackEnv(
        natural_payout=cfg["NATURAL_PAYOUT"],
        num_decks=cfg["NUM_DECKS"],
        dealer_hits_soft_17=cfg["DEALER_HITS_SOFT_17"],
        render_mode=cfg["RENDER_MODE"],
        seed=cfg["RANDOM_SEED"],
    )