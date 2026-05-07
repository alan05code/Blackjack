# Blackjack Gymnasium Environment

Ambiente Gymnasium completo per il gioco del Blackjack con supporto grafico e integrazione per modelli di visione artificiale e decisione.

## Caratteristiche

- ✅ Ambiente Gymnasium standard compatibile con RL
- ✅ Rendering grafico con pygame (opzionale)
- ✅ Logica di gioco completa e corretta
- ✅ Hook per modelli di riconoscimento carte da tavolo reale
- ✅ Hook per modelli di decisione automatica (HIT/STAND)
- ✅ Supporto per blackjack naturale, soft/hard hands, dealer rules

## Installazione

```bash
pip install -r requirements.txt
```

## Configurazione

Il file `config.py` permette di configurare il comportamento del gioco:

### Variabili Principali

- **`USE_VISION_RECOGNITION`**: Se `True`, usa le carte riconosciute dal modello di visione (da webcam/tavolo reale). Se `False`, usa la logica interna del gioco.
- **`USE_AI_DECISION`**: Se `True`, usa il modello AI per decidere HIT/STAND automaticamente. Se `False`, richiede input manuale.

### Esempi di Configurazione

**Modalità Normale (gioco tradizionale, input manuale):**
```python
USE_VISION_RECOGNITION = False
USE_AI_DECISION = False
```

**Modalità Visione (carte da webcam, input manuale):**
```python
USE_VISION_RECOGNITION = True
USE_AI_DECISION = False
CARD_RECOGNITION_MODEL_PATH = "models/card_model.pth"
```

**Modalità AI (gioco tradizionale, decisioni automatiche):**
```python
USE_VISION_RECOGNITION = False
USE_AI_DECISION = True
DECISION_MODEL_PATH = "models/decision_model.pth"
```

**Modalità Completa (visione + AI):**
```python
USE_VISION_RECOGNITION = True
USE_AI_DECISION = True
CARD_RECOGNITION_MODEL_PATH = "models/card_model.pth"
DECISION_MODEL_PATH = "models/decision_model.pth"
```

## Uso Base

### Gioco Manuale con Interfaccia Grafica

```bash
python run_game.py
```

Il gioco legge automaticamente `config.py`. Per ignorare la configurazione:
```bash
python run_game.py --no-config
```

### Uso Programmatico

```python
from blackjack_env import BlackjackEnv

env = BlackjackEnv(render_mode="human")
obs, info = env.reset()

# Gioca manualmente
obs, reward, terminated, truncated, info = env.step(1)  # HIT
obs, reward, terminated, truncated, info = env.step(0)  # STAND

env.close()
```

## Integrazione con modello di visione

### Architettura

- **CardRecognitionModel**: riconosce le carte dal tavolo reale (webcam/foto).
- **VisionGameMiddleware**: ponte tra modello di visione, logica di gioco e renderer; aggiorna lo stato in tempo reale e richiama il rendering.

### Card Recognition Model

Il modello di riconoscimento carte deve implementare l'interfaccia `CardRecognitionModel`:

```python
from blackjack_env import CardRecognitionModel, RecognitionResult
import numpy as np

class MyCardRecognitionModel:
    def recognize_cards(self, frame: np.ndarray) -> RecognitionResult:
        # frame: RGB image (H, W, 3) dal tavolo reale
        # Restituisce: RecognitionResult con carte riconosciute
        ...
```

### Flusso visione → gioco → rendering (realtime)

```python
import numpy as np
from blackjack_env import BlackjackEnv, VisionGameMiddleware

# Il modello di visione gira esternamente (es. vision_blackjack.py) e fornisce le label
env = BlackjackEnv(render_mode="human")  # userà NoOpCardRecognitionModel interno
middleware = VisionGameMiddleware(env)

# Se hai già le label riconosciute:
player_labels = ["A♠️", "10♥️"]
dealer_labels = ["9♣️", "8♦️"]
middleware.update_from_labels(player_labels, dealer_labels, render=True)
```

## Come eseguire

### Setup rapido (start.bat)
Lo script `start.bat` crea il virtualenv `.venv`, installa i requisiti e avvia in parallelo:
- `run_game.py` (UI con pulsanti: HIT, STAND, NUOVA MANO, SUGGERIMENTO; in visione resta visibile il suggerimento).
- `vision_blackjack.py` (mostra le immagini, disegna box/label e scrive lo stato in `blackjack_env/tmp/vision_state.json`).

### Passi completi
1) Configura `config.py`:
   - `USE_VISION_RECOGNITION = True` per usare lo stato dal modello di visione.
   - `POLICY_PATH` (default `blackjack_env/model/policy.npy`) per il suggeritore RL.
   - `VISION_STATE_PATH` (default `blackjack_env/tmp/vision_state.json`) per lo scambio stato visione→gioco.
   - `RENDER_MODE = "human"` per vedere la finestra Pygame.
2) Genera il dataset augmentato: `python dataset/augment_cards.py` (produce immagini in `dataset/carte_aug`).
3) Addestra il modello di visione in `card_dataset_training.ipynb` (usa le carte augmentate) e salva il classifier/label map.
4) Allena l’agente RL in `blackjack_env/BJRL.py` (crea la policy e salvala in `POLICY_PATH`).
5) Avvia `start.bat` (crea `.venv`, installa dipendenze e lancia `run_game.py` + `vision_blackjack.py`).

### Flusso visione → gioco
- `vision_blackjack.py` elabora le immagini (A/D per sfogliare, U per inviare) usando il preprocessing centralizzato: maschera, warp prospettico sui 4 angoli, binarizzazione e KNN (`card_knn.pkl`). Salva lo stato in JSON (`vision_state.json`).
- `run_game.py` monitora il JSON e aggiorna il rendering e l’HUD (punteggi, stato round). In visione i pulsanti restano visibili; il suggerimento RL non altera lo stato, indica solo HIT/STAND.

## Struttura Progetto

```
blackjack_env/
├── __init__.py          # Esportazioni pubbliche
├── game.py              # Logica di gioco (mazzo, punteggi, regole)
├── env.py               # Ambiente Gymnasium
├── rendering.py         # Rendering grafico pygame
└── vision.py            # Interfacce per modelli AI
```

## Note

- Il rendering grafico è opzionale e non richiesto per l'uso con modelli AI.
- I modelli lavorano con frame del tavolo reale, non con il rendering del gioco.
- Il sistema supporta sia modalità manuale che automatica.
- Il preprocessing immagini è centralizzato in `blackjack_env/card_preprocessing.py`:
  - maschera robusta, ricerca box, warp prospettico sui 4 angoli del convex hull, binarizzazione Otsu e resize 160x114;
  - corner detection deterministica (punti più vicini ai bordi);
  - normalizzazione binaria condivisa tra training, riconoscimento notebook, `vision_blackjack.py`;
  - classi retro supportate: `carte_rosse`, `carte_blu`;
  - possibilità di limitare le immagini per etichetta nel notebook di training (`MAX_PER_LABEL`).