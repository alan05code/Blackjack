"""Logica di gioco del blackjack indipendente da Gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Optional, Sequence

RANKS: Sequence[str] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS: Sequence[str] = ("♠️", "♥️", "♦️", "♣️") # Spade, Hearts, Diamonds, Clubs 
_SUIT_BASE_TO_CANONICAL: Dict[str, str] = {s[0]: s for s in SUITS}
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


def parse_card_label(label: str) -> Optional[Card]:
    """
    Converte una stringa "RankSuit" in Card, tollerando variazioni unicode del seme.

    Accetta sia la forma con variation selector (es. "A♠️") che senza (es. "A♠").
    Restituisce None se la label non è valida.
    """
    if not label:
        return None
    # Prova prima il match esatto con la forma canonica (con VS-16)
    for suit in SUITS:
        if label.endswith(suit):
            rank = label[: -len(suit)]
            if rank in RANKS:
                return Card(rank, suit)
    # Fallback: match con il solo carattere base del seme (senza VS-16)
    last_char = label[-1]
    canonical = _SUIT_BASE_TO_CANONICAL.get(last_char)
    if canonical is not None:
        rank = label[:-1]
        if rank in RANKS:
            return Card(rank, canonical)
    return None


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
        Aggiorna le mani del gioco con le carte riconosciute dal modello di visione.

        Args:
            player_cards: Label delle carte del player (es. ["A♠️", "10♥️"])
            dealer_cards: Label delle carte del dealer
            validate: Se True, segnala errori per le label non riconosciute.
                Anche con validate=True le carte valide vengono comunque applicate
                (aggiornamento parziale): l'unico effetto è la presenza di
                `errors` nel risultato.

        Returns:
            Dict {"updated": bool, "errors": list[str], "partial": bool}.
            `updated` è True solo se tutte le label sono state riconosciute.
            `partial` è True se ci sono errori ma almeno una carta è stata applicata.
        """
        errors: List[str] = []

        def _parse_list(labels: List[str], owner: str) -> List[Card]:
            cards: List[Card] = []
            for raw in labels:
                card = parse_card_label(raw)
                if card is not None:
                    cards.append(card)
                elif validate:
                    errors.append(f"Carta {owner} non riconosciuta: {raw!r}")
            return cards

        try:
            self.player_hand = _parse_list(player_cards, "player")
            self.dealer_hand = _parse_list(dealer_cards, "dealer")
            return {
                "updated": not errors,
                "errors": errors,
                "partial": bool(errors) and bool(self.player_hand or self.dealer_hand),
            }
        except Exception as e:
            return {"updated": False, "errors": [f"Errore durante aggiornamento: {e}"], "partial": False}

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