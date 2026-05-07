"""
Interfacce e hook per modelli di visione artificiale e middleware verso il gioco.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .env import BlackjackEnv


# ============================================================================
# Card Recognition Model
# ============================================================================


@dataclass
class RecognizedCard:
    """Rappresenta una carta riconosciuta dal modello di visione."""

    rank: str  # "A", "2"-"10", "J", "Q", "K"
    suit: str  # "♠️", "♥️", "♦️", "♣️"
    confidence: float  # 0.0-1.0, confidenza del riconoscimento
    position: str  # "player" o "dealer"

    def to_label(self) -> str:
        """Converte in formato label compatibile con il gioco."""
        return f"{self.rank}{self.suit}"


@dataclass
class RecognitionResult:
    """Risultato del riconoscimento carte da un frame."""

    player_cards: List[RecognizedCard]  # Carte riconosciute per il player
    dealer_cards: List[RecognizedCard]  # Carte riconosciute per il dealer
    frame_timestamp: float  # Timestamp del frame processato
    confidence_threshold: float = 0.5  # Soglia di confidenza minima

    def get_player_labels(self) -> List[str]:
        """Restituisce le label delle carte del player."""
        return [card.to_label() for card in self.player_cards if card.confidence >= self.confidence_threshold]

    def get_dealer_labels(self) -> List[str]:
        """Restituisce le label delle carte del dealer."""
        return [card.to_label() for card in self.dealer_cards if card.confidence >= self.confidence_threshold]


class CardRecognitionModel(Protocol):
    """
    Contratto per modelli di riconoscimento carte da immagini del tavolo reale.

    Il modello riceve un frame (da webcam o foto) e riconosce le carte
    presenti sul tavolo, distinguendo tra carte del player e del dealer.
    """

    def recognize_cards(self, frame: np.ndarray) -> RecognitionResult:
        """
        Riconosce le carte da un frame del tavolo.

        Args:
            frame: Immagine RGB del tavolo (H, W, 3) in formato numpy array.
                  Può provenire da webcam, screenshot, o qualsiasi fonte video.

        Returns:
            RecognitionResult con le carte riconosciute per player e dealer.

        Note:
            - Il frame deve essere in formato RGB (non BGR)
            - Le dimensioni possono variare, il modello deve gestire il resize
            - Le carte con confidence < threshold dovrebbero essere filtrate
        """
        ...  # pragma: no cover


class NoOpCardRecognitionModel:
    """Implementazione stub che non riconosce carte (per testing/development)."""

    def recognize_cards(self, frame: np.ndarray) -> RecognitionResult:
        """Restituisce un risultato vuoto."""
        return RecognitionResult(
            player_cards=[],
            dealer_cards=[],
            frame_timestamp=0.0,
            confidence_threshold=0.5,
        )


@dataclass
class GameState:
    """
    Stato del gioco proveniente da una sorgente esterna (es. modello di visione).
    Usato per sincronizzare il tavolo con ciò che è stato rilevato.
    """

    player_hand: List[str]
    dealer_hand: List[str]
    player_total: int = 0
    dealer_total: int = 0
    usable_ace: bool = False
    round_over: bool = False
    status: str | None = None

# ============================================================================
# Vision <-> Game middleware
# ============================================================================


class VisionGameMiddleware:
    """
    Collega il modello di visione con il gioco e il renderer, così che le
    carte rilevate possano aggiornare lo stato del gioco in tempo reale.
    """

    def __init__(
        self,
        env: "BlackjackEnv",
        *,
        render_mode: str | None = None,
        card_recognition_model: CardRecognitionModel | None = None,
    ) -> None:
        self.env = env
        if render_mode is not None:
            self.env.render_mode = render_mode
        if card_recognition_model is not None:
            self.env.card_recognition_model = card_recognition_model

    # ------------------------------------------------------------------ #
    # Aggiornamento dallo stream di visione
    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray, *, auto_reset: bool = False, render: bool = True) -> dict:
        """
        Processa un frame del tavolo, aggiorna lo stato del gioco e (opzionalmente) renderizza.

        Args:
            frame: frame RGB (H, W, 3) proveniente dal modello di visione.
            auto_reset: se True, resetta il round quando quello precedente è concluso.
            render: se True, chiama il renderer se `render_mode` è impostato.

        Returns:
            Dizionario info aggiornato dell'ambiente.
        """
        if auto_reset and getattr(self.env.game, "round_over", False):
            self.env.game.reset_round()

        info = self.env.update_state_from_recognition(frame)
        if render and self.env.render_mode is not None:
            self.env.render()
        return info

    # ------------------------------------------------------------------ #
    # Aggiornamento manuale (da label già stimate)
    # ------------------------------------------------------------------ #
    def update_from_labels(
        self,
        player_labels: List[str],
        dealer_labels: List[str],
        *,
        validate: bool = True,
        render: bool = True,
    ) -> dict:
        """
        Aggiorna lo stato del gioco partendo da label di carte già riconosciute.

        Utile quando il modello di visione produce direttamente le label o per test.
        """
        result = self.env.game.update_hands_from_recognition(player_labels, dealer_labels, validate=validate)
        self.env._last_message = (
            "Stato aggiornato da label fornite." if result.get("updated") else "Aggiornamento parziale da label."
        )
        # Determina se il round è concluso in base alle mani attuali
        self.env.game.round_over = self.env._should_end_round()
        info = self.env._build_info(reveal=self.env.game.round_over)
        info.update(
            {
                "recognition_updated": result.get("updated", False),
                "recognition_errors": result.get("errors", []),
                "recognized_player_cards": player_labels,
                "recognized_dealer_cards": dealer_labels,
            }
        )
        self.env._latest_info = info
        if render and self.env.render_mode is not None:
            self.env.render()
        return info

    # ------------------------------------------------------------------ #
    # Utilità
    # ------------------------------------------------------------------ #
    def last_info(self) -> dict:
        """Restituisce l'ultimo info renderizzato/aggiornato."""
        return getattr(self.env, "_latest_info", {})

    def close(self) -> None:
        """Chiude eventuali risorse grafiche dell'ambiente."""
        try:
            self.env.close()
        except Exception:
            pass