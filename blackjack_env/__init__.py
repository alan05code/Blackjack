"""Pacchetto blackjack_env: ambiente Gymnasium per Blackjack con supporto grafico e modelli AI."""

from .env import BlackjackEnv
from .game import BlackjackGame
from .rendering import BlackjackRenderer
from .vision import (
    CardRecognitionModel,
    GameState,
    RecognitionResult,
    RecognizedCard,
    VisionGameMiddleware,
)

# Config loader
try:
    from .config_loader import create_env_from_config, load_config
    __all__ = [
        "BlackjackEnv",
        "BlackjackGame",
        "BlackjackRenderer",
        "CardRecognitionModel",
        "GameState",
        "RecognitionResult",
        "RecognizedCard",
        "VisionGameMiddleware",
        "create_env_from_config",
        "load_config",
    ]
except ImportError:
    __all__ = [
        "BlackjackEnv",
        "BlackjackGame",
        "BlackjackRenderer",
        "CardRecognitionModel",
        "GameState",
        "RecognitionResult",
        "RecognizedCard",
        "VisionGameMiddleware",
    ]
