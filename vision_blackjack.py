"""
Porting in Python script della logica del notebook `card_recognition_model.ipynb`:
- Carica modello Keras e mappa label dal notebook.
- Segmenta le carte con contorni/filtri e le mostra con bounding box e label predetta.
- Usa un'immagine fissa (BGR) da disco, es. `dataset/sim_gioco/prova1.jpg`.
- Con il tasto "u" invia lo stato (player/dealer) al gioco tramite middleware; il render Pygame mostra le carte.
- Con il tasto "q" esce.

Prerequisiti:
- `config.py` con `USE_VISION_RECOGNITION=True` e percorsi modello/label corretti.
- File modello/label presenti (default: `dataset/models/card_classifier.keras`, `dataset/models/card_labels.txt`).

MODIFICA: supporto diretto IP Webcam se `PHONE_CAMERA_URL` è configurata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import json
import os
import joblib

from blackjack_env import BlackjackEnv, VisionGameMiddleware
from blackjack_env import card_preprocessing as cp

# Percorsi di default (coerenti con il notebook aggiornato)
MODEL_PATHS = [
    Path("dataset/models/card_knn.pkl"),
    Path("models/card_knn.pkl"),
]
LABEL_PATHS = [
    Path("dataset/models/card_labels.txt"),
    Path("models/card_labels.txt"),
]
IMAGE_ROOT = Path("dataset/sim_gioco")
IMG_HEIGHT, IMG_WIDTH = cp.IMG_HEIGHT, cp.IMG_WIDTH  # 160x114

# Colori bounding box (BGR)
COLOR_FOUND      = (0, 165, 255)   # arancione — trovata ma non riconosciuta
COLOR_RED_SUIT   = (0, 0, 220)     # rosso  — riconosciuta, seme rosso (cuori/quadri)
COLOR_BLACK_SUIT = (220, 100, 0)   # blu    — riconosciuta, seme nero (picche/fiori)
CONFIDENCE_THRESHOLD = 0.5

RED_SUITS   = {"cuori", "quadri"}    # seme rosso nella label raw
BLACK_SUITS = {"picche", "fiori"}    # seme nero nella label raw

def suit_color(raw_label: str) -> tuple:
    """Restituisce il colore BGR in base al seme predetto nella label raw."""
    tokens = raw_label.lower().replace("-", "_").split("_")
    for tok in tokens:
        if tok in RED_SUITS:
            return COLOR_RED_SUIT
        if tok in BLACK_SUITS:
            return COLOR_BLACK_SUIT
    return COLOR_RED_SUIT  # fallback

# Se True, usa la sorgente video (webcam o stream) invece dello scorrimento immagini
USE_VIDEO = True
# Legge la sorgente video da config
try:
    from config import PHONE_CAMERA_URL, WEBCAM_INDEX, PHONE_CAMERA_INDEX
except Exception:
    PHONE_CAMERA_URL = ""
    WEBCAM_INDEX = 0
    PHONE_CAMERA_INDEX = None

# Numero massimo di indici webcam da scansionare
MAX_CAMERA_SCAN = 10

# Mapping da label del dataset a rank/suit di gioco
RANK_MAP = {
    "asso": "A",
    "due": "2",
    "tre": "3",
    "quattro": "4",
    "cinque": "5",
    "sei": "6",
    "sette": "7",
    "otto": "8",
    "nove": "9",
    "dieci": "10",
    "jack": "J",
    "regina": "Q",
    "re": "K",
}
SUIT_MAP = {
    "cuori": "♥️",
    "quadri": "♦️",
    "fiori": "♣️",
    "picche": "♠️",
}

def first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def load_label_map(labels_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                idx = int(parts[0])
                mapping[idx] = parts[1]
    return mapping

def to_game_label(raw_label: str) -> str:
    """
    Converte etichette tipo 'dieci_quadri' in '10♦️'.
    """
    token_str = raw_label.lower().replace("-", "_")
    tokens = token_str.split("_")
    rank = None
    suit = None
    for tok in tokens:
        if rank is None and tok in RANK_MAP:
            rank = RANK_MAP[tok]
        if suit is None and tok in SUIT_MAP:
            suit = SUIT_MAP[tok]
    if rank and suit:
        return f"{rank}{suit}"
    return raw_label

def map_labels(labels: List[str]) -> List[str]:
    """Applica la conversione rank/suit -> emoji per un elenco di label raw."""
    return [to_game_label(lbl) for lbl in labels]

def preprocess_card(card_bgr: np.ndarray) -> np.ndarray:
    """Usa preprocessing condiviso: binario + flatten."""
    return cp.card_to_vector(card_bgr, (IMG_HEIGHT, IMG_WIDTH))

def find_card_boxes(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Rileva contorni candidati carta usando le funzioni condivise.
    """
    boxes, _ = cp.find_card_contours(frame_bgr)
    boxes = cp.filter_valid_card_boxes(
        boxes,
        min_aspect=cp.CARD_MIN_ASPECT,
        max_aspect=cp.CARD_MAX_ASPECT,
        min_area_ratio=cp.CARD_MIN_AREA_RATIO,
        max_area_ratio=cp.CARD_MAX_AREA_RATIO,
        image_shape=frame_bgr.shape,
    )
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def predict_frame(frame_bgr: np.ndarray, model, idx_to_label: Dict[int, str]):
    """
    Segmenta, predice e restituisce (boxes, raw_labels, probs, positions).
    positions: "dealer" se nella metà superiore, "player" altrimenti.
    """
    boxes = find_card_boxes(frame_bgr)
    if not boxes:
        return [], [], [], []

    h_img = frame_bgr.shape[0]
    midline = h_img * 0.5
    crops = []
    positions = []
    for (x, y, w, h) in boxes:
        crop_raw = cp.extract_full_card(frame_bgr, x, y, w, h)
        crop_aligned, _ = cp.affine_align_card(crop_raw)
        crops.append(crop_aligned)
        positions.append("dealer" if (y + h / 2) < midline else "player")

    batch = np.stack([preprocess_card(c) for c in crops], axis=0)
    preds_idx = model.predict(batch)

    raw_labels = []
    probs = []
    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(batch)
        except Exception:
            probas = None

    for i, idx in enumerate(preds_idx):
        idx = int(idx)
        prob = float(np.max(probas[i])) if probas is not None else 0.0
        raw_label = idx_to_label.get(idx, str(idx))
        raw_labels.append(raw_label)
        probs.append(prob)
    return boxes, raw_labels, probs, positions

def rotate_if_vertical(frame: np.ndarray) -> np.ndarray:
    """Ruota l'immagine se è orientata verticalmente (più alta che larga)."""
    if frame.shape[0] > frame.shape[1]:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    imgs: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            imgs.append(p)
    return imgs

def load_and_resize(path: Path) -> np.ndarray | None:
    frame_orig = cv2.imread(str(path))
    if frame_orig is None:
        return None
    frame_orig = rotate_if_vertical(frame_orig)
    target_w, target_h = 1280, 900
    h0, w0 = frame_orig.shape[:2]
    scale = min(target_w / w0, target_h / h0, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        return cv2.resize(frame_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame_orig

def scan_video_sources() -> List[dict]:
    """
    Scansiona le sorgenti video disponibili (indici 0..MAX_CAMERA_SCAN-1).
    Restituisce una lista di dict con 'index', 'width', 'height', 'name'.
    """
    sources: List[dict] = []
    for i in range(MAX_CAMERA_SCAN):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = f"Camera {i}"
            sources.append({"index": i, "width": w, "height": h, "name": name})
            cap.release()
    return sources

def select_video_source() -> int | str | None:
    """
    Stampa le sorgenti video disponibili e chiede all'utente di sceglierne una.
    Restituisce l'indice (int) o l'URL (str) scelto, oppure None se nessuna è disponibile.
    """
    print("\n" + "=" * 60)
    print("  SORGENTI VIDEO DISPONIBILI")
    print("=" * 60)

    sources = scan_video_sources()

    if not sources:
        print("  Nessuna sorgente video trovata!")
        print("=" * 60)
        return None

    for i, src in enumerate(sources):
        tag = ""
        if PHONE_CAMERA_INDEX is not None and src["index"] == PHONE_CAMERA_INDEX:
            tag = " [TELEFONO USB - config]"
        elif src["index"] == WEBCAM_INDEX:
            tag = " [WEBCAM - config]"
        print(f"  [{i}] {src['name']}  ({src['width']}x{src['height']}){tag}")

    # Aggiungi opzione URL se configurato
    url_option = None
    if PHONE_CAMERA_URL:
        url_option = len(sources)
        print(f"  [{url_option}] Telefono via rete: {PHONE_CAMERA_URL}")

    print("=" * 60)

    # Determina sorgente di default
    default_idx = 0
    if PHONE_CAMERA_INDEX is not None:
        for i, src in enumerate(sources):
            if src["index"] == PHONE_CAMERA_INDEX:
                default_idx = i
                break
    else:
        for i, src in enumerate(sources):
            if src["index"] == WEBCAM_INDEX:
                default_idx = i
                break

    max_option = len(sources) - 1 + (1 if url_option is not None else 0)
    while True:
        try:
            raw = input(f"  Scegli sorgente [0-{max_option}] (default={default_idx}): ").strip()
            if raw == "":
                choice = default_idx
            else:
                choice = int(raw)
            if 0 <= choice <= max_option:
                break
            print(f"  Inserisci un numero tra 0 e {max_option}.")
        except ValueError:
            print("  Input non valido. Inserisci un numero.")

    if url_option is not None and choice == url_option:
        print(f"\n  -> Selezionata sorgente: Telefono via rete ({PHONE_CAMERA_URL})")
        return PHONE_CAMERA_URL

    selected = sources[choice]
    print(f"\n  -> Selezionata sorgente: {selected['name']} ({selected['width']}x{selected['height']})")
    print("=" * 60 + "\n")
    return selected["index"]

def main() -> None:
    model_path = first_existing(MODEL_PATHS)
    labels_path = first_existing(LABEL_PATHS)
    if model_path is None or labels_path is None:
        print("Modello o label non trovati. Controlla i percorsi in MODEL_PATHS/LABEL_PATHS.")
        return

    images = list_images(IMAGE_ROOT)
    if not USE_VIDEO and not images:
        print(f"Nessuna immagine trovata in {IMAGE_ROOT}")
        return

    print(f"Modello: {model_path}")
    print(f"Labels: {labels_path}")

    # -------------------------------------------------------
    # MODIFICA: supporto IP Webcam diretto senza selezione
    # -------------------------------------------------------
    video_source = None
    if USE_VIDEO:
        if PHONE_CAMERA_URL:
            print(f"\nUsando IP Webcam configurata: {PHONE_CAMERA_URL}")
            video_source = PHONE_CAMERA_URL
        else:
            video_source = select_video_source()
            if video_source is None:
                print("Nessuna sorgente video disponibile. Uscita.")
                return
    else:
        print(f"Immagini trovate: {len(images)} (A/D per navigare)")

    knn_bundle = joblib.load(model_path)
    model = knn_bundle["model"]
    idx_to_label = knn_bundle.get("idx_to_label", {})
    if not idx_to_label:
        idx_to_label = load_label_map(labels_path)

    env = BlackjackEnv(render_mode="human")
    mw = VisionGameMiddleware(env)

    idx = 0
    frame = None
    cap = None
    if USE_VIDEO:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            if isinstance(video_source, str):
                print(f"⚠️  Impossibile aprire lo stream IP Webcam: {video_source}")
                print("   Verifica che l'URL sia corretto e che il telefono sia connesso alla rete.")
                print("   Formati comuni: http://<ip>:8080/video?android.mjpeg")
            else:
                print(f"Sorgente video non disponibile: {video_source}")
            return
        print("Stream video avviato correttamente.")
    else:
        frame = load_and_resize(images[idx])
        if frame is None:
            print(f"Immagine non valida: {images[idx]}")
            return

    try:
        while True:
            if USE_VIDEO:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Frame video non valido o stream terminato.")
                    break
            display = frame.copy()
            boxes, raw_labels, probs, positions = predict_frame(display, model, idx_to_label)

            player_raw = [lbl for lbl, pos in zip(raw_labels, positions) if pos == "player"]
            dealer_raw = [lbl for lbl, pos in zip(raw_labels, positions) if pos == "dealer"]
            # Per display usiamo le label raw (no emoji). La conversione a emoji avviene solo all'invio.
            player_labels_send = map_labels(player_raw)
            dealer_labels_send = map_labels(dealer_raw)

            for (x, y, w, h), lbl, prob in zip(boxes, raw_labels, probs):
                # prob==0 significa che il modello non supporta predict_proba → tratta come riconosciuta
                recognized = (prob == 0.0) or (prob >= CONFIDENCE_THRESHOLD)
                if recognized:
                    color = suit_color(lbl)
                    label_text = f"R: {lbl} {prob:.2f}"
                else:
                    color = COLOR_FOUND
                    label_text = f"T: {lbl} {prob:.2f}"
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    display,
                    label_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Disegna linea di mezzeria (separa dealer/player in verticale)
            mid_y = display.shape[0] // 2
            cv2.line(display, (0, mid_y), (display.shape[1], mid_y), (255, 255, 0), 2)
            cv2.putText(
                display,
                "DEALER",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "PLAYER",
                (20, mid_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

            text_cmd = "U: send | Q: quit"
            (txt_w, txt_h), _ = cv2.getTextSize(text_cmd, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            margin = 20
            x_cmd = display.shape[1] - txt_w - margin
            y_cmd = 30
            cv2.putText(
                display,
                text_cmd,
                (x_cmd, y_cmd),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if not USE_VIDEO:
                text_name = f"{images[idx].name} [{idx+1}/{len(images)}]"
                (name_w, name_h), _ = cv2.getTextSize(text_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                x_name = display.shape[1] - name_w - margin
                y_name = display.shape[0] - 20
                cv2.putText(
                    display,
                    text_name,
                    (x_name, y_name),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Vision (IP Webcam)" if USE_VIDEO and isinstance(video_source, str) else "Vision (static image)", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("u"):
                if not player_labels_send and not dealer_labels_send:
                    print("Nessuna carta da inviare.")
                    continue
                # Scrivi stato su file JSON per il client run_game.py (asincrono)
                try:
                    os.makedirs("blackjack_env/tmp", exist_ok=True)
                    with open("blackjack_env/tmp/vision_state.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "player_hand": player_labels_send,
                                "dealer_hand": dealer_labels_send,
                                "status": "Vision update",
                            },
                            f,
                            ensure_ascii=False,
                        )
                    print("Stato salvato in blackjack_env/tmp/vision_state.json (player/dealer).")
                except Exception as e:
                    print(f"Errore nel salvataggio di vision_state.json: {e}")
            if not USE_VIDEO:
                if key == ord("a"):  # precedente
                    idx = (idx - 1) % len(images)
                    new_frame = load_and_resize(images[idx])
                    if new_frame is not None:
                        frame = new_frame
                    print(f"Caricata immagine: {images[idx]}")
                if key == ord("d"):  # successiva
                    idx = (idx + 1) % len(images)
                    new_frame = load_and_resize(images[idx])
                    if new_frame is not None:
                        frame = new_frame
                    print(f"Caricata immagine: {images[idx]}")
    finally:
        if cap is not None:
            cap.release()
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()