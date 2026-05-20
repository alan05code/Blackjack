"""
File di configurazione per Blackjack Environment.

Modifica queste variabili per cambiare il comportamento del gioco.
"""

# Flag modalità visione: True se lo stato arriva da modello di visione esterno
USE_VISION_RECOGNITION = True

# Path policy per l'agente RL (usato dal client per il suggerimento)
POLICY_PATH = "blackjack_env/model/policy.npy"

# File JSON condiviso per aggiornare lo stato da uno script di visione esterno
# Formato atteso:
# {"player_hand": ["A♠️", "10♥️"], "dealer_hand": ["9♣️", "8♦️"], "status": "Vision update"}
VISION_STATE_PATH = "blackjack_env/tmp/vision_state.json"

# ============================================================================
# Configurazione Webcam (solo se USE_VISION_RECOGNITION=True)
# ============================================================================

# Indice della webcam da usare (0 = prima webcam disponibile)
WEBCAM_INDEX = 0

# Risoluzione webcam (None = default)
WEBCAM_WIDTH = None
WEBCAM_HEIGHT = None

# Indice della telecamera del telefono collegato via cavo USB
# (il telefono appare come seconda webcam, tipicamente indice 1 o 2).
# Impostare a None per disabilitare; ha priorità su PHONE_CAMERA_URL.
PHONE_CAMERA_INDEX = None

# URL della telecamera del telefono via rete (es. http://192.168.1.10:8080/video)
# Usato solo se PHONE_CAMERA_INDEX è None.
# Lasciare stringa vuota per usare la webcam locale definita da WEBCAM_INDEX.
PHONE_CAMERA_URL = "http://10.126.70.122:8080/video?android.mjpeg"

# ============================================================================
# Configurazione Gioco
# ============================================================================

# Numero di mazzi nel shoe
NUM_DECKS = 1

# Payout per blackjack naturale
NATURAL_PAYOUT = 1.5

# Se True, dealer pesca su soft 17
DEALER_HITS_SOFT_17 = False

# Seed per riproducibilità (None = random)
RANDOM_SEED = None

# ============================================================================
# Configurazione Rendering
# ============================================================================

# Modalità rendering: "human", "rgb_array", o None
RENDER_MODE = "human"

# Dimensioni finestra (solo se RENDER_MODE="human")
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Frame per secondo del loop pygame
FPS = 30