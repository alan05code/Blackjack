"""Supporto di rendering e componenti grafici riutilizzabili."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pygame

BACKGROUND_TOP = (6, 60, 52)
BACKGROUND_BOTTOM = (2, 25, 22)
TABLE_COLOR = (8, 122, 78)
TABLE_EDGE = (3, 80, 52)
TABLE_GLOW = (13, 158, 108, 50)
CARD_COLOR = (247, 247, 247)
CARD_BORDER = (35, 35, 35)
CARD_SHADOW = (0, 0, 0, 90)
TEXT_PRIMARY = (240, 240, 240)
TEXT_SECONDARY = (200, 200, 200)
ALERT_COLOR = (255, 210, 90)
RED_TEXT = (190, 50, 60)
PANEL_BG = (0, 0, 0, 140)
BADGE_WIN = (70, 190, 140)
BADGE_LOSE = (190, 70, 70)
BADGE_INFO = (230, 180, 80)
DIVIDER_COLOR = (255, 255, 255, 40)


class BlackjackRenderer:
    """Disegna lo stato del gioco su superfici pygame e produce frame RGB."""

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        title: str = "Gym Blackjack",
        auto_display: bool = False,
    ) -> None:
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.title = title
        self.auto_display = auto_display
        self.surface = pygame.Surface((width, height))
        self.window: pygame.Surface | None = None
        if auto_display:
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("arial", 26)
        self.small_font = pygame.font.SysFont("arial", 20)
        self.badge_font = pygame.font.SysFont("arial", 18, bold=True)
        self.tiny_font = pygame.font.SysFont("arial", 16)
        self._last_frame: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # API pubbliche
    # ------------------------------------------------------------------ #
    def render(self, info: Dict[str, object], mode: str = "human", message: str | None = None) -> np.ndarray | None:
        """Renderizza su finestra (human) o restituisce un frame RGB."""

        self._draw_scene(self.surface, info, message or "")
        frame = self._surface_to_frame(self.surface)
        if mode == "human":
            self._ensure_window()
            assert self.window is not None
            self.window.blit(self.surface, (0, 0))
            pygame.display.flip()
        if mode == "rgb_array":
            return frame.copy()
        return None

    def draw_on(
        self,
        target_surface: pygame.Surface,
        info: Dict[str, object],
        message: str = "",
        emit_frame: bool = False,
    ) -> np.ndarray | None:
        """Disegna su una superficie fornita (es. interfaccia interattiva)."""

        self._draw_scene(target_surface, info, message)
        if not emit_frame:
            return None
        frame = self._surface_to_frame(target_surface)
        return frame

    def get_last_frame(self) -> np.ndarray | None:
        return None if self._last_frame is None else self._last_frame.copy()

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
            self.window = None
        pygame.quit()

    # ------------------------------------------------------------------ #
    # Interni
    # ------------------------------------------------------------------ #
    def _draw_scene(self, surface: pygame.Surface, info: Dict[str, object], message: str) -> None:
        self._draw_background(surface)
        self._draw_table(surface)
        self._draw_header(surface, info)
        player_highlight = self._hand_highlight(info, owner="player")
        dealer_highlight = self._hand_highlight(info, owner="dealer")
        self._draw_hand(surface, info.get("dealer_hand", []), label="DEALER", top=140, highlight=dealer_highlight)
        self._draw_hand(surface, info.get("player_hand", []), label="PLAYER", top=400, highlight=player_highlight)
        self._draw_footer(surface, info)
        self._draw_status_panel(surface, message or info.get("status", ""))
        self._draw_shoe_indicator(surface, info)

    def _draw_background(self, surface: pygame.Surface) -> None:
        for y in range(self.height):
            ratio = y / max(self.height - 1, 1)
            color = tuple(
                int(BACKGROUND_TOP[idx] + (BACKGROUND_BOTTOM[idx] - BACKGROUND_TOP[idx]) * ratio) for idx in range(3)
            )
            pygame.draw.line(surface, color, (0, y), (self.width, y))

    def _draw_table(self, surface: pygame.Surface) -> None:
        table_rect = pygame.Rect(40, 90, self.width - 80, self.height - 280)
        halo_surface = pygame.Surface(table_rect.inflate(140, 80).size, pygame.SRCALPHA)
        pygame.draw.ellipse(halo_surface, TABLE_GLOW, halo_surface.get_rect())
        surface.blit(halo_surface, table_rect.inflate(140, 80).topleft)
        pygame.draw.rect(surface, TABLE_COLOR, table_rect, border_radius=120)
        pygame.draw.rect(surface, TABLE_EDGE, table_rect, width=6, border_radius=120)
        divider = pygame.Surface((table_rect.width - 80, 2), pygame.SRCALPHA)
        pygame.draw.rect(divider, DIVIDER_COLOR, divider.get_rect())
        surface.blit(divider, (table_rect.left + 40, table_rect.centery - 1))

    def _draw_header(self, surface: pygame.Surface, info: Dict[str, object]) -> None:
        panel_rect = pygame.Rect(40, 20, self.width - 80, 70)
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, PANEL_BG, panel.get_rect(), border_radius=20)
        title = self.font.render("Blackjack Gymnasium", True, TEXT_PRIMARY)
        panel.blit(title, (20, 15))
        subtitle = self.small_font.render("Modalità training & manuale", True, TEXT_SECONDARY)
        panel.blit(subtitle, (20, 40))
        cards_left = info.get("shoe_size", 0)
        gauge_rect = pygame.Rect(panel_rect.width - 220, 15, 200, 40)
        self._draw_shoe_gauge(panel, gauge_rect, cards_left)
        surface.blit(panel, panel_rect.topleft)

    def _draw_hand(
        self,
        surface: pygame.Surface,
        cards: Sequence[str],
        label: str,
        top: int,
        highlight: Tuple[str, Tuple[int, int, int]] | None = None,
    ) -> None:
        start_x = 80
        card_width, card_height = 100, 140
        gap = 20
        step = card_width + gap
        max_width = self.width - start_x - 60
        n = len(cards)
        if n > 1 and n * card_width + (n - 1) * gap > max_width:
            step = max(card_width // 3, (max_width - card_width) // (n - 1))
        label_render = self.small_font.render(label, True, TEXT_PRIMARY)
        surface.blit(label_render, (start_x, top - 40))
        if highlight:
            badge_text, color = highlight
            self._draw_badge(surface, (start_x + 130, top - 50), badge_text, color)
            glow_rect = pygame.Rect(start_x - 20, top - 20, card_width + step * max(n - 1, 0) + 40, card_height + 40)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*color, 50), glow_surface.get_rect(), border_radius=26)
            surface.blit(glow_surface, glow_rect.topleft)
        for idx, card_str in enumerate(cards):
            x = start_x + idx * step
            rect = pygame.Rect(x, top, card_width, card_height)
            shadow = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
            pygame.draw.rect(shadow, CARD_SHADOW, shadow.get_rect(), border_radius=14)
            surface.blit(shadow, (rect.x + 6, rect.y + 6))
            pygame.draw.rect(surface, CARD_COLOR, rect, border_radius=14)
            pygame.draw.rect(surface, CARD_BORDER, rect, width=2, border_radius=14)
            
            if card_str == "??":
                # Carta coperta
                text_color = (120, 120, 120)
                hidden_text = self.small_font.render("??", True, text_color)
                hidden_rect = hidden_text.get_rect(center=rect.center)
                surface.blit(hidden_text, hidden_rect)
            else:
                # Estrai rank e suit dalla stringa
                rank, suit = self._parse_card(card_str)
                text_color = self._card_text_color(card_str)
                
                # Valore in alto a sinistra
                rank_text = self.small_font.render(rank, True, text_color)
                surface.blit(rank_text, (rect.x + 8, rect.y + 8))
                
                # Seme al centro (più grande)
                suit_font = pygame.font.SysFont("arial", 48)
                suit_text = suit_font.render(suit, True, text_color)
                suit_rect = suit_text.get_rect(center=rect.center)
                surface.blit(suit_text, suit_rect)
                
                # Valore in basso a destra (ruotato)
                rank_rotated = pygame.transform.rotate(rank_text, 180)
                rank_rotated_rect = rank_rotated.get_rect()
                rank_rotated_rect.bottomright = (rect.right - 8, rect.bottom - 8)
                surface.blit(rank_rotated, rank_rotated_rect)

    def _draw_footer(self, surface: pygame.Surface, info: Dict[str, object]) -> None:
        panel_rect = pygame.Rect(40, self.height - 170, self.width - 80, 60)
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, PANEL_BG, panel.get_rect(), border_radius=18)
        player_total = info.get("player_total", 0)
        dealer_total = info.get("dealer_total", 0)
        round_over = info.get("round_over", False)
        usable = "SI" if info.get("usable_ace", 0) else "NO"
        left = self.small_font.render(f"Player: {player_total}  |  Usable Ace: {usable}", True, TEXT_PRIMARY)
        panel.blit(left, (20, 15))
        dealer_txt = dealer_total if round_over else "??"
        right = self.small_font.render(f"Dealer: {dealer_txt}", True, TEXT_PRIMARY)
        panel.blit(right, (panel_rect.width - right.get_width() - 20, 15))
        surface.blit(panel, panel_rect.topleft)

    def _draw_status_panel(self, surface: pygame.Surface, message: str) -> None:
        if not message:
            return
        panel_rect = pygame.Rect(40, self.height - 220, self.width - 80, 40)
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (0, 0, 0, 160), panel.get_rect(), border_radius=12)
        status = self.small_font.render(message, True, ALERT_COLOR)
        panel.blit(status, (20, 8))
        surface.blit(panel, panel_rect.topleft)

    def _draw_shoe_indicator(self, surface: pygame.Surface, info: Dict[str, object]) -> None:
        cards_left = info.get("shoe_size", 0)
        indicator_rect = pygame.Rect(self.width - 140, self.height - 330, 100, 100)
        circle_surface = pygame.Surface(indicator_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (0, 0, 0, 110), (indicator_rect.width // 2, indicator_rect.height // 2), indicator_rect.width // 2)
        pygame.draw.circle(
            circle_surface,
            (255, 255, 255, 60),
            (indicator_rect.width // 2, indicator_rect.height // 2),
            indicator_rect.width // 2 - 2,
            width=2,
        )
        text = self.small_font.render(str(cards_left), True, TEXT_PRIMARY)
        text_rect = text.get_rect(center=(indicator_rect.width // 2, indicator_rect.height // 2))
        label = self.tiny_font.render("CARTE", True, TEXT_SECONDARY)
        label_rect = label.get_rect(center=(indicator_rect.width // 2, indicator_rect.height // 2 + 28))
        circle_surface.blit(text, text_rect)
        circle_surface.blit(label, label_rect)
        surface.blit(circle_surface, indicator_rect.topleft)

    def _draw_shoe_gauge(self, panel: pygame.Surface, rect: pygame.Rect, cards_left: int) -> None:
        pygame.draw.rect(panel, (255, 255, 255, 30), rect, border_radius=14)
        inner = rect.inflate(-6, -14)
        capacity = 52
        ratio = max(0.0, min(cards_left / capacity, 1.0))
        fill_width = int(inner.width * ratio)
        fill_rect = pygame.Rect(inner.left, inner.top, fill_width, inner.height)
        color = (90, 200, 140) if ratio > 0.4 else (220, 180, 80) if ratio > 0.2 else (200, 90, 90)
        pygame.draw.rect(panel, color, fill_rect, border_radius=10)
        text = self.tiny_font.render(f"Mazzo: {cards_left} carte", True, TEXT_PRIMARY)
        panel.blit(text, (rect.left + 10, rect.top + 10))

    def _hand_highlight(self, info: Dict[str, object], owner: str) -> Tuple[str, Tuple[int, int, int]] | None:
        round_over = bool(info.get("round_over"))
        if owner == "player":
            if info.get("player_blackjack"):
                return ("BLACKJACK", BADGE_INFO)
            total = info.get("player_total", 0)
            dealer = info.get("dealer_total", 0)
            if total > 21:
                return ("BUST", BADGE_LOSE)
            if round_over:
                if dealer > 21 or total > dealer:
                    return ("WIN", BADGE_WIN)
                if total == dealer:
                    return ("PUSH", BADGE_INFO)
                return ("LOSE", BADGE_LOSE)
        else:
            if info.get("dealer_blackjack"):
                return ("BLACKJACK", BADGE_INFO)
            total = info.get("dealer_total", 0)
            player = info.get("player_total", 0)
            if round_over:
                if total > 21:
                    return ("BUST", BADGE_LOSE)
                if total > player:
                    return ("WIN", BADGE_WIN)
                if total == player:
                    return ("PUSH", BADGE_INFO)
                return ("LOSE", BADGE_LOSE)
        return None

    def _draw_badge(self, surface: pygame.Surface, position: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
        text_render = self.badge_font.render(text, True, (20, 20, 20))
        padding_x, padding_y = 14, 6
        badge_rect = pygame.Rect(position[0], position[1], text_render.get_width() + padding_x * 2, text_render.get_height() + padding_y * 2)
        badge_surface = pygame.Surface(badge_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(badge_surface, (*color, 180), badge_surface.get_rect(), border_radius=14)
        badge_surface.blit(text_render, text_render.get_rect(center=badge_surface.get_rect().center))
        surface.blit(badge_surface, badge_rect.topleft)

    def _parse_card(self, card_str: str) -> Tuple[str, str]:
        """Estrae rank e suit da una stringa carta (es. 'A♠️' -> ('A', '♠️'))."""
        # I semi Unicode sono: ♠️, ♥️, ♦️, ♣️
        suits = ["♠️", "♥️", "♦️", "♣️"]
        for suit in suits:
            if suit in card_str:
                rank = card_str.replace(suit, "")
                return (rank, suit)
        # Fallback: se non trova il seme, assume che sia tutto il rank
        return (card_str, "")
    
    def _card_text_color(self, card_label: str) -> Tuple[int, int, int]:
        if "♥" in card_label or "♦" in card_label:
            return RED_TEXT
        return CARD_BORDER

    def _surface_to_frame(self, surface: pygame.Surface) -> np.ndarray:
        array = pygame.surfarray.array3d(surface)
        frame = np.transpose(array, (1, 0, 2))
        self._last_frame = frame.copy()
        return frame

    def _ensure_window(self) -> None:
        if self.window is None:
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.title)