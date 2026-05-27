"""Middleware per propagare lo stato dal modello di visione al gioco."""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .env import BlackjackEnv

# Token usato dal modello di visione per indicare una carta coperta (back card).
HIDDEN_CARD_TOKEN = "??"


def _split_hidden(labels: List[str]) -> Tuple[List[str], int]:
    """Separa le label valide dai token di carta coperta. Ritorna (visibili, conteggio_coperte)."""
    visible = [l for l in labels if l != HIDDEN_CARD_TOKEN]
    return visible, len(labels) - len(visible)


class VisionGameMiddleware:
    """
    Sottile adattatore fra il modello di visione (che produce label di carte)
    e l'ambiente di gioco. Espone un solo punto di ingresso, `update_from_labels`,
    che aggiorna le mani e ridisegna se richiesto.

    Le label uguali a `'??'` rappresentano carte coperte: non vengono passate al
    game (che lavora solo su carte note) ma vengono reiniettate come placeholder
    nel `info` finale così che il renderer possa disegnarle al posto giusto.
    """

    def __init__(self, env: "BlackjackEnv", *, render_mode: str | None = None) -> None:
        self.env = env
        if render_mode is not None:
            self.env.render_mode = render_mode

    def update_from_labels(
        self,
        player_labels: List[str],
        dealer_labels: List[str],
        *,
        validate: bool = True,
        render: bool = True,
    ) -> dict:
        """Aggiorna lo stato del gioco partendo dalle label riconosciute."""
        player_visible, player_hidden = _split_hidden(player_labels)
        dealer_visible, dealer_hidden = _split_hidden(dealer_labels)

        result = self.env.game.update_hands_from_recognition(
            player_visible, dealer_visible, validate=validate
        )
        if result.get("updated"):
            self.env._last_message = "Stato aggiornato da label fornite."
        elif result.get("partial"):
            self.env._last_message = "Aggiornamento parziale: alcune carte non riconosciute."
        else:
            self.env._last_message = "Nessuna carta riconosciuta dalle label fornite."

        self.env.game.round_over = self.env._should_end_round()
        info = self.env._build_info(reveal=self.env.game.round_over)

        # Reinietta le carte coperte come placeholder per il rendering.
        # I totali non cambiano: una carta coperta non viene contata nel punteggio.
        if player_hidden:
            info["player_hand"] = list(info.get("player_hand", [])) + [HIDDEN_CARD_TOKEN] * player_hidden
        if dealer_hidden:
            info["dealer_hand"] = list(info.get("dealer_hand", [])) + [HIDDEN_CARD_TOKEN] * dealer_hidden

        info.update(
            {
                "recognition_updated": result.get("updated", False),
                "recognition_errors": result.get("errors", []),
                "recognized_player_cards": player_labels,
                "recognized_dealer_cards": dealer_labels,
                "player_hidden_count": player_hidden,
                "dealer_hidden_count": dealer_hidden,
            }
        )
        self.env._latest_info = info
        if render and self.env.render_mode is not None:
            self.env.render()
        return info

    def last_info(self) -> dict:
        return getattr(self.env, "_latest_info", {})

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
