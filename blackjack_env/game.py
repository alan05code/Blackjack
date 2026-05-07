"""Logica di gioco del blackjack indipendente da Gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Sequence

RANKS: Sequence[str] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS: Sequence[str] = ("♠️", "♥️", "♦️", "♣️")  # Spade, Hearts, Diamonds, Clubs (ASCII only)
CARD_VALUES: Dict[str, int] = {
    "A": 11,
    "K": 10,
    "Q": 10,
    "J": 10,
    "10": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}


@dataclass(frozen=True)
class Card:
    """Rappresenta una singola carta."""

    rank: str
    suit: str

    @property
    def value(self) -> int:
        return CARD_VALUES[self.rank]

    def label(self) -> str:
        return f"{self.rank}{self.suit}"


class BlackjackGame:
    """Incarica la gestione di mazzo, punteggi e fasi di una mano."""

    def __init__(
        self,
        num_decks: int = 1,
        natural_payout: float = 1.5,
        dealer_hits_soft_17: bool = False,
        reshuffle_threshold: float = 0.25,
        seed: int | None = None,
    ) -> None:
        if num_decks < 1:
            raise ValueError("num_decks deve essere >= 1")
        self.num_decks = num_decks
        self.natural_payout = natural_payout
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.reshuffle_threshold = min(max(reshuffle_threshold, 0.05), 0.8)
        self._rng = random.Random(seed)
        self._shoe: List[Card] = []
        self.player_hand: List[Card] = []
        self.dealer_hand: List[Card] = []
        self.round_over: bool = False
        self._build_shoe()

    # --------------------------------------------------------------------- #
    # Mazzo e carte
    # --------------------------------------------------------------------- #
    def _build_shoe(self) -> None:
        self._shoe = [Card(rank, suit) for rank in RANKS for suit in SUITS for _ in range(self.num_decks)]
        self._rng.shuffle(self._shoe)
        self._reshuffle_cut = max(int(len(self._shoe) * self.reshuffle_threshold), 10)

    def _maybe_reshuffle(self) -> None:
        if len(self._shoe) <= self._reshuffle_cut:
            self._build_shoe()

    def _draw_card(self) -> Card:
        if not self._shoe:
            self._build_shoe()
        return self._shoe.pop()

    # --------------------------------------------------------------------- #
    # Round lifecycle
    # --------------------------------------------------------------------- #
    def reset_round(self) -> Dict[str, object]:
        self._maybe_reshuffle()
        self.round_over = False
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        return self.public_state(reveal_dealer=False)

    def player_hit(self) -> Card:
        card = self._draw_card()
        self.player_hand.append(card)
        return card

    def update_hands_from_recognition(
        self, player_cards: List[str], dealer_cards: List[str], validate: bool = True
    ) -> Dict[str, object]:
        """
        Aggiorna le mani del gioco basandosi sulle carte riconosciute dal modello di visione.

        Args:
            player_cards: Lista di label delle carte del player (es. ["A♠️", "10♥️"])
            dealer_cards: Lista di label delle carte del dealer
            validate: Se True, valida che le carte siano nel formato corretto

        Returns:
            Dict con informazioni sull'aggiornamento (es. {"updated": True, "errors": []})

        Note:
            Questo metodo permette di sincronizzare lo stato del gioco con le carte
            riconosciute dal modello di visione. Le carte vengono convertite da label
            a oggetti Card e assegnate alle mani corrispondenti.
        """
        errors: List[str] = []
        try:
            # Converti label in Card objects
            player_hand_new: List[Card] = []
            for label in player_cards:
                if validate and len(label) < 2:
                    errors.append(f"Label player non valida: {label}")
                    continue
                rank = label[:-2] if len(label) > 2 else label[0]  # Rimuovi emoji (2 caratteri)
                suit = label[-2:] if len(label) > 2 else label[1] if len(label) > 1 else ""
                if rank in RANKS and suit in SUITS:
                    player_hand_new.append(Card(rank, suit))
                else:
                    errors.append(f"Carta player non riconosciuta: {label}")

            dealer_hand_new: List[Card] = []
            for label in dealer_cards:
                if validate and len(label) < 2:
                    errors.append(f"Label dealer non valida: {label}")
                    continue
                rank = label[:-2] if len(label) > 2 else label[0]
                suit = label[-2:] if len(label) > 2 else label[1] if len(label) > 1 else ""
                if rank in RANKS and suit in SUITS:
                    dealer_hand_new.append(Card(rank, suit))
                else:
                    errors.append(f"Carta dealer non riconosciuta: {label}")

            # Aggiorna le mani solo se non ci sono errori o se validate=False
            if not errors or not validate:
                self.player_hand = player_hand_new
                self.dealer_hand = dealer_hand_new

            return {"updated": len(errors) == 0, "errors": errors}
        except Exception as e:
            return {"updated": False, "errors": [f"Errore durante aggiornamento: {str(e)}"]}

    def dealer_play(self) -> None:
        self.round_over = True
        while True:
            total = self.hand_value(self.dealer_hand)
            soft = self.has_usable_ace(self.dealer_hand) and total == 17
            if total > 21:
                break
            if total > 17 or (total == 17 and not (soft and self.dealer_hits_soft_17)):
                break
            self.dealer_hand.append(self._draw_card())

    # --------------------------------------------------------------------- #
    # Utility di punteggio
    # --------------------------------------------------------------------- #
    @staticmethod
    def hand_value(hand: Sequence[Card]) -> int:
        total = sum(card.value for card in hand)
        aces = sum(1 for card in hand if card.rank == "A")
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    @staticmethod
    def has_usable_ace(hand: Sequence[Card]) -> bool:
        total = sum(card.value for card in hand)
        aces = sum(1 for card in hand if card.rank == "A")
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return aces > 0

    @staticmethod
    def is_blackjack(hand: Sequence[Card]) -> bool:
        return len(hand) == 2 and BlackjackGame.hand_value(hand) == 21

    # --------------------------------------------------------------------- #
    # Informazioni per ambiente / interfaccia
    # --------------------------------------------------------------------- #
    def upcard_value(self) -> int:
        return self.dealer_hand[0].value if self.dealer_hand else 0

    def public_state(self, reveal_dealer: bool) -> Dict[str, object]:
        dealer_has_cards = len(self.dealer_hand) > 0
        return {
            "player_hand": self._hand_to_labels(self.player_hand),
            "dealer_hand": self._hand_to_labels(self.dealer_hand, hide_second=not reveal_dealer) if dealer_has_cards else [],
            "player_total": self.hand_value(self.player_hand),
            "dealer_total": self.hand_value(self.dealer_hand) if dealer_has_cards and reveal_dealer else (self.dealer_hand[0].value if dealer_has_cards else 0),
            "usable_ace": int(self.has_usable_ace(self.player_hand)),
            "shoe_size": len(self._shoe),
            "round_over": reveal_dealer,
        }

    def full_state(self) -> Dict[str, object]:
        return {
            "player_hand": self._hand_to_labels(self.player_hand),
            "dealer_hand": self._hand_to_labels(self.dealer_hand),
            "player_total": self.hand_value(self.player_hand),
            "dealer_total": self.hand_value(self.dealer_hand),
            "player_blackjack": self.is_blackjack(self.player_hand),
            "dealer_blackjack": self.is_blackjack(self.dealer_hand),
            "usable_ace": int(self.has_usable_ace(self.player_hand)),
        }

    @staticmethod
    def _hand_to_labels(hand: Iterable[Card], hide_second: bool = False) -> List[str]:
        labels: List[str] = []
        for idx, card in enumerate(hand):
            if hide_second and idx == 1:
                labels.append("??")
            else:
                labels.append(card.label())
        return labels