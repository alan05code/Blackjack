"""Avvia un client grafico minimale per giocare a Blackjack con Gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pygame
import numpy as np
import json
import os

from blackjack_env import BlackjackRenderer, GameState, VisionGameMiddleware
from blackjack_env.BJRL import BlackjackAgent, agent_action
from blackjack_env.config_loader import create_env_from_config, load_config


BUTTON_BASE = (45, 45, 45)
BUTTON_HOVER = (90, 90, 90)
BUTTON_TEXT = (240, 240, 240)
BUTTON_SHADOW = (0, 0, 0, 120)
HUD_PANEL = (0, 0, 0, 140)
HUD_TEXT = (235, 235, 235)
HUD_TEXT_MUTED = (185, 185, 185)


@dataclass
class Button:
    label: str
    action: Optional[int]  # None = reset nuovo round
    rect: pygame.Rect


class BlackjackApp:
    """Loop pygame che utilizza l'ambiente Gym per la logica di gioco."""

    def __init__(self) -> None:
        """Inizializza l'applicazione Blackjack usando solo config.py."""
        pygame.init()
        
        # Carica configurazione da config.py
        self.config = load_config()
        self.fps = self.config["FPS"]
        self.clock = pygame.time.Clock()
        self.width = self.config["WINDOW_WIDTH"]
        self.height = self.config["WINDOW_HEIGHT"]
        self.use_vision_recognition = self.config["USE_VISION_RECOGNITION"]
        self.use_ai_decision = False  # modalità AI decisionale disabilitata

        pygame.display.set_caption("Blackjack - Modalità Manuale" if not self.use_ai_decision else "Blackjack - Modalità AI")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("arial", 20)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.tiny_font = pygame.font.SysFont("arial", 16)
        
        # Crea ambiente dalla configurazione
        self.env = create_env_from_config()
        self.middleware = VisionGameMiddleware(self.env) if self.use_vision_recognition else None
        self.decision_model = None
        self.agent = self._load_agent()
        self.vision_state_path = self.config.get("VISION_STATE_PATH")
        self._vision_state_mtime: float = 0.0
        
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
            # Nessun reset iniziale: lo stato viene solo dal modello di visione
            self.observation = np.array([0, 0, 0], dtype=np.int32)
            self.info = {
                "player_hand": [],
                "dealer_hand": [],
                "player_total": 0,
                "dealer_total": 0,
                "usable_ace": 0,
                "round_over": False,
                "player_blackjack": False,
                "dealer_blackjack": False,
                "status": self.status_message,
            }
        else:
            self.observation, self.info = self.env.reset()
            if self.use_vision_recognition and self.middleware:
                self.info = self.middleware.last_info() or self.info
        if self.use_vision_recognition:
           self._load_vision_json(force=True, log_prefix="[vision] Stato iniziale")

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

            if self.use_vision_recognition:
                self._pull_vision_state()

            self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps)
        self.env.close()
        if self.cap:
            self.cap.release()
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
            # In modalità visione non si resetta il mazzo: si aspetta un nuovo input dal modello
            self.observation = np.array([0, 0, 0], dtype=np.int32)
            self.info = {
                "player_hand": [],
                "dealer_hand": [],
                "player_total": 0,
                "dealer_total": 0,
                "usable_ace": 0,
                "round_over": False,
                "player_blackjack": False,
                "dealer_blackjack": False,
                "status": "In attesa del modello di visione.",
            }
            self.status_message = self.info["status"]
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

    def _pull_vision_state(self) -> None:
        """
        Aggiorna lo stato usando il middleware di visione.

        Nota: questa funzione assume che una sorgente esterna fornisca frame o label.
        Personalizza `provide_vision_state` per integrare la tua pipeline (es. notebook).
        """
        if not self.middleware:
            return

        # 1) Prova a leggere da file JSON esterno (vision_state_path)
        if self._load_vision_json():
            return

    def _obs_from_info(self, info: dict) -> "np.ndarray":
        import numpy as np

        player_sum = info.get("player_total", 0)
        dealer_upcard = info.get("dealer_total", 0) if info.get("round_over") else info.get("dealer_total", 0)
        usable_ace = int(info.get("usable_ace", 0))
        return np.array([player_sum, dealer_upcard, usable_ace], dtype=np.int32)

    def _load_agent(self) -> Optional[BlackjackAgent]:
        """
        Carica un agente RL da file policy.npy se disponibile.
        """
        policy_path = self.config.get("POLICY_PATH", "blackjack_env/model/policy.npy")
        try:
            agent = BlackjackAgent()
            agent.load(policy_path)
            return agent
        except Exception:
            return None

    def _load_vision_json(self, *, force: bool = False, log_prefix: str = "[vision]") -> bool:
        """
        Legge il file JSON condiviso e aggiorna lo stato se è più recente o se force=True.
        """
        if not self.vision_state_path or not os.path.exists(self.vision_state_path):
            return False
        try:
            mtime = os.path.getmtime(self.vision_state_path)
            if not force and mtime <= self._vision_state_mtime:
                return False
            with open(self.vision_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._vision_state_mtime = mtime
            player_labels = data.get("player_hand", [])
            dealer_labels = data.get("dealer_hand", [])
            status = data.get("status")
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
                print(f"{log_prefix} JSON aggiornato: {self.vision_state_path} (player={player_labels}, dealer={dealer_labels}, mtime={mtime})")
            elif status:
                self.status_message = status
                print(f"{log_prefix} JSON senza carte ma con status: {status} (mtime={mtime})")
            return True
        except Exception as e:
            print(f"{log_prefix} Errore lettura JSON {self.vision_state_path}: {e}")
            return False
    
def main() -> None:
    """Avvia l'applicazione Blackjack usando la configurazione da config.py."""
    app = BlackjackApp()
    try:
        app.run()
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()


