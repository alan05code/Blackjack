"""Avvia un client grafico minimale per giocare a Blackjack con Gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pygame
import numpy as np
import json
import os
import sys

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

from blackjack_env import BlackjackRenderer, VisionGameMiddleware
from blackjack_env.BJRL import BlackjackAgent, agent_action
from blackjack_env.config_loader import create_env_from_config, load_config


BUTTON_BASE = (45, 45, 45)
BUTTON_HOVER = (90, 90, 90)
BUTTON_TEXT = (240, 240, 240)
BUTTON_SHADOW = (0, 0, 0, 120)
HUD_PANEL = (0, 0, 0, 140)
HUD_TEXT = (235, 235, 235)
HUD_TEXT_MUTED = (185, 185, 185)

EMPTY_INFO = {
    "player_hand": [],
    "dealer_hand": [],
    "player_total": 0,
    "dealer_total": 0,
    "usable_ace": 0,
    "round_over": False,
    "player_blackjack": False,
    "dealer_blackjack": False,
    "status": "",
}


@dataclass
class Button:
    label: str
    action: Optional[int]  # None = reset nuovo round
    rect: pygame.Rect


class BlackjackApp:
    """Loop pygame che utilizza l'ambiente Gym per la logica di gioco."""

    def __init__(self) -> None:
        pygame.init()

        self.config = load_config()
        self.fps = self.config["FPS"]
        self.clock = pygame.time.Clock()
        self.width = self.config["WINDOW_WIDTH"]
        self.height = self.config["WINDOW_HEIGHT"]
        self.use_vision_recognition = self.config["USE_VISION_RECOGNITION"]

        pygame.display.set_caption(
            "Blackjack - Modalità Visione" if self.use_vision_recognition else "Blackjack - Modalità Manuale"
        )
        self.windowed_size = (self.width, self.height)
        self.fullscreen = False
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("arial", 20)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.tiny_font = pygame.font.SysFont("arial", 16)

        self.env = create_env_from_config()
        self.middleware = VisionGameMiddleware(self.env) if self.use_vision_recognition else None
        self.agent = self._load_agent()
        self.vision_state_path = self.config.get("VISION_STATE_PATH")
        self._vision_state_mtime: float = 0.0
        self._vision_state_signature: tuple | None = None

        self.renderer = BlackjackRenderer(
            width=self.width,
            height=self.height,
            auto_display=False,
        )
        self.buttons = self._create_buttons()
        self.status_message = (
            "Modalità visione attiva: in attesa delle carte dal modello."
            if self.use_vision_recognition
            else "Benvenuto al tavolo!"
        )
        self.round_over = False
        self.last_reward = 0.0

        if self.use_vision_recognition:
            self.observation = np.array([0, 0, 0], dtype=np.int32)
            self.info = {**EMPTY_INFO, "status": self.status_message}
            self._load_vision_json(force=True, log_prefix="[vision] Stato iniziale")
        else:
            self.observation, self.info = self.env.reset()

    def _create_buttons(self) -> Sequence[Button]:
        button_width, button_height = 160, 45
        spacing = 20
        start_x = 40
        y = self.height - 70
        labels = [("HIT", 1), ("STAND", 0), ("NUOVA MANO", None), ("SUGGERIMENTO AI", 2)]
        if self.use_vision_recognition:
            labels = [("SUGGERIMENTO AI", 2)]  # in visione mostriamo solo il suggerimento
        buttons: list[Button] = []
        for idx, (label, action) in enumerate(labels):
            rect = pygame.Rect(start_x + idx * (button_width + spacing), y, button_width, button_height)
            buttons.append(Button(label=label, action=action, rect=rect))
        return buttons

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._on_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    elif event.key == pygame.K_ESCAPE and self.fullscreen:
                        self._toggle_fullscreen()
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self._resize(event.w, event.h)

            if self.use_vision_recognition:
                self._load_vision_json()

            self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps)
        self.env.close()
        pygame.quit()

    def _draw(self) -> None:
        self.renderer.draw_on(self.screen, self.info, message=self.status_message, emit_frame=True)
        self._draw_hud()
        self._draw_button_bar()

    def _draw_button_bar(self) -> None:
        bar_rect = pygame.Rect(0, self.height - 100, self.width, 100)
        panel = pygame.Surface(bar_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (0, 0, 0, 180), panel.get_rect())
        pygame.draw.line(panel, (255, 255, 255, 50), (0, 0), (panel.get_width(), 0))
        self.screen.blit(panel, bar_rect.topleft)
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            hover = button.rect.collidepoint(mouse_pos)
            color = BUTTON_HOVER if hover else BUTTON_BASE
            shadow = pygame.Surface((button.rect.width, button.rect.height), pygame.SRCALPHA)
            pygame.draw.rect(shadow, BUTTON_SHADOW, shadow.get_rect(), border_radius=10)
            self.screen.blit(shadow, (button.rect.x + 4, button.rect.y + 5))
            pygame.draw.rect(self.screen, color, button.rect, border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 255), button.rect, width=1, border_radius=10)
            label = self.font.render(button.label, True, BUTTON_TEXT)
            label_rect = label.get_rect(center=button.rect.center)
            self.screen.blit(label, label_rect)

    def _draw_hud(self) -> None:
        panel_rect = pygame.Rect(self.width - 250, 110, 210, 200)
        hud = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(hud, HUD_PANEL, hud.get_rect(), border_radius=18)
        title = self.small_font.render("Pannello mano", True, HUD_TEXT)
        hud.blit(title, (16, 14))
        lines = [
            ("Somma player", f"{int(self.observation[0])}"),
            ("Upcard dealer", f"{int(self.observation[1])}"),
            ("Asso utilizzabile", "SI" if int(self.observation[2]) else "NO"),
            ("Ricompensa", f"{self.last_reward:+.1f}"),
            ("Stato round", "CHIUSO" if self.round_over else "IN CORSO"),
        ]
        for idx, (label, value) in enumerate(lines):
            text = self.tiny_font.render(label, True, HUD_TEXT_MUTED)
            val = self.tiny_font.render(value, True, HUD_TEXT)
            y = 50 + idx * 26
            hud.blit(text, (16, y))
            hud.blit(val, (16, y + 12))
        self.screen.blit(hud, panel_rect.topleft)

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.windowed_size = (self.width, self.height)
            sizes = pygame.display.get_desktop_sizes()
            self.width, self.height = sizes[0]
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME)
        else:
            self.width, self.height = self.windowed_size
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self._rebuild_layout()

    def _resize(self, w: int, h: int) -> None:
        self.width = max(640, w)
        self.height = max(480, h)
        self.windowed_size = (self.width, self.height)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self._rebuild_layout()

    def _rebuild_layout(self) -> None:
        self.renderer = BlackjackRenderer(
            width=self.width,
            height=self.height,
            auto_display=False,
        )
        self.buttons = self._create_buttons()

    def _on_click(self, mouse_pos) -> None:
        for button in self.buttons:
            if button.rect.collidepoint(mouse_pos):
                if button.action is None:
                    self._reset_round()
                else:
                    self._play_action(button.action)
                break

    def _reset_round(self) -> None:
        if self.use_vision_recognition:
            self.observation = np.array([0, 0, 0], dtype=np.int32)
            self.info = {**EMPTY_INFO, "status": "In attesa del modello di visione."}
            self.status_message = self.info["status"]
            self._vision_state_signature = None
        else:
            self.observation, self.info = self.env.reset()
            self.status_message = "Nuova mano pronta."
        self.round_over = False
        self.last_reward = 0.0

    def _play_action(self, action: int) -> None:
        """
        Esegue un'azione nel gioco (come un giocatore manuale).

        Args:
            action: Azione da eseguire (0=STAND, 1=HIT).
        """
        if self.round_over:
            self.status_message = "Round concluso: premi 'NUOVA MANO'."
            return

        if action == 2 and self.agent is not None:
            try:
                rl_action = agent_action(self.agent, self.observation)
            except Exception:
                self.status_message = "Agente non disponibile (policy non caricata?)."
                return
            # Solo suggerimento: non modifica lo stato del gioco
            self.status_message = "Suggerimento agente: HIT" if rl_action == 1 else "Suggerimento agente: STAND"
            return

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.observation, self.info = obs, info
        self.last_reward = reward if terminated else 0.0
        if terminated or truncated:
            self.round_over = True
            if reward > 0:
                self.status_message = f"Hai vinto! Ricompensa: {reward:.1f}"
            elif reward < 0:
                self.status_message = "Hai perso la mano."
            else:
                self.status_message = "Push: pareggio."
        else:
            self.status_message = info.get("status", "In corso...")

    def _obs_from_info(self, info: dict) -> "np.ndarray":
        player_sum = info.get("player_total", 0)
        dealer_upcard = info.get("dealer_total", 0)
        usable_ace = int(info.get("usable_ace", 0))
        return np.array([player_sum, dealer_upcard, usable_ace], dtype=np.int32)

    def _load_agent(self) -> Optional[BlackjackAgent]:
        policy_path = self.config.get("POLICY_PATH", "blackjack_env/model/policy.npy")
        try:
            agent = BlackjackAgent()
            agent.load(policy_path)
            return agent
        except Exception:
            return None

    def _load_vision_json(self, *, force: bool = False, log_prefix: str = "[vision]") -> bool:
        """
        Legge il file JSON condiviso e aggiorna lo stato se cambiato.

        Confronto duplice: mtime (veloce) e signature dei dati (robusto rispetto a
        scritture con stessa mtime o riscritture identiche).
        """
        if not self.middleware or not self.vision_state_path:
            return False
        if not os.path.exists(self.vision_state_path):
            return False
        try:
            mtime = os.path.getmtime(self.vision_state_path)
            if not force and mtime <= self._vision_state_mtime:
                return False
            with open(self.vision_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"{log_prefix} Errore lettura JSON {self.vision_state_path}: {e}")
            return False

        self._vision_state_mtime = mtime
        player_labels = list(data.get("player_hand") or [])
        dealer_labels = list(data.get("dealer_hand") or [])
        status = data.get("status")

        signature = (tuple(player_labels), tuple(dealer_labels), status)
        if not force and signature == self._vision_state_signature:
            return False
        self._vision_state_signature = signature

        if player_labels or dealer_labels:
            info = self.middleware.update_from_labels(player_labels, dealer_labels, render=True)
            self.info = info
            self.observation = self._obs_from_info(info)
            self.round_over = bool(info.get("round_over", False))
            if status:
                self.status_message = status
            elif self.round_over:
                self.status_message = "Round concluso (vision)."
            else:
                self.status_message = info.get("status", self.status_message)
            if not self.round_over:
                self.last_reward = 0.0
            print(f"{log_prefix} aggiornamento: player={player_labels} dealer={dealer_labels}")
        else:
            # Nessuna carta: tratta come "in attesa nuova mano"
            self.info = {**EMPTY_INFO, "status": status or "In attesa del modello di visione."}
            self.observation = np.array([0, 0, 0], dtype=np.int32)
            self.round_over = False
            self.last_reward = 0.0
            self.status_message = self.info["status"]
            print(f"{log_prefix} mani vuote, reset visivo.")
        return True


def main() -> None:
    """Avvia l'applicazione Blackjack usando la configurazione da config.py."""
    app = BlackjackApp()
    try:
        app.run()
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()


