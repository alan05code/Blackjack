"""
Utilità condivise per il preprocessing delle carte.

Le funzioni sono organizzate per step così da poterle riutilizzare nei notebook
mostrando facilmente i risultati intermedi.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configurazione predefinita allineata a training/inference
IMG_HEIGHT: int = 160
IMG_WIDTH: int = 114
DEFAULT_SIZE: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)  # (h, w)

MORPH_KERNEL_SIZE: int = 7
MORPH_ITERATIONS: int = 3

# Soglie generiche per box (rapporti)
CARD_MIN_ASPECT: float = 0.55
CARD_MAX_ASPECT: float = 0.85
CARD_MIN_AREA_RATIO: float = 0.05
CARD_MAX_AREA_RATIO: float = 0.90

# Soglie assolute per area (usate nello split dealer/player)
CARD_MIN_AREA_ABS: int = 300_000
CARD_MAX_AREA_ABS: int = 700_000

# Filtro contorni
CARD_THRESHOLD: int = 180
CARD_MIN_CONTOUR_AREA: int = 1000


# ----------------------------
# Parsing etichette dataset
# ----------------------------
def parse_card_label(path: Path) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Estrae etichetta da nome file: valore_seme_colore.jpg -> usa SOLO valore+seme come classe.
    Restituisce (label_dict, error_message).
    """
    stem = path.stem.lower()
    parts = stem.split("_")
    if len(parts) < 3:
        return None, f"Formato nome non valido: {stem}"

    value, suit = parts[0], parts[1]
    label = f"{value}_{suit}"
    return {
        "label": label,
        "value": value,
        "suit": suit,
    }, None


# ----------------------------
# Step 1: maschera bianca
# ----------------------------
def create_white_mask(image: np.ndarray) -> np.ndarray:
    """Maschera bianca (gray + soglia fissa + close/open + dilatazione)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(gray_blur, CARD_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel_close = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    kernel_open = np.ones((max(3, MORPH_KERNEL_SIZE // 3), max(3, MORPH_KERNEL_SIZE // 3)), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=MORPH_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


# ----------------------------
# Step 1 bis: split immagine dealer/player
# ----------------------------
def split_image_dealer_player(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Divide l'immagine in due metà: dealer (alto) e player (basso); ruota se verticale."""
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = image.shape[:2]
    mid_y = h // 2
    dealer_image = image[0:mid_y, :]
    player_image = image[mid_y:h, :]
    return dealer_image, player_image


# ----------------------------
# Step 1 ter: contorni carte su immagine
# ----------------------------
def find_card_contours(image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Trova i contorni delle carte con soglia fissa su grigio + filtro area contorno.
    Ritorna (boxes, mask).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, CARD_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) > CARD_MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
    return boxes, mask


# ----------------------------
# Step 2: ricerca box plausibile
# ----------------------------
def find_best_card_region(
    mask: np.ndarray,
    image_shape: Tuple[int, int, int],
    card_min_aspect: float = CARD_MIN_ASPECT,
    card_max_aspect: float = CARD_MAX_ASPECT,
    card_min_area_ratio: float = CARD_MIN_AREA_RATIO,
    card_max_area_ratio: float = CARD_MAX_AREA_RATIO,
) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """Trova il bounding box più plausibile per la carta e il relativo contorno."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h_img, w_img = image_shape[:2]
    img_area = h_img * w_img
    candidates: List[Tuple[float, Tuple[int, int, int, int], np.ndarray]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / h if h > 0 else 0
        area_ratio = area / img_area
        if card_min_aspect <= aspect <= card_max_aspect and card_min_area_ratio <= area_ratio <= card_max_area_ratio:
            candidates.append((area, (x, y, w, h), contour))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, box, contour = candidates[0]
    return box, contour


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def corners_from_mask(mask_bin: np.ndarray) -> Optional[np.ndarray]:
    """
    Estrae 4 angoli dalla maschera binaria con un'unica logica deterministica:
    - trova il contorno principale
    - calcola il convex hull
    - seleziona i punti più vicini ai bordi immagine (distanza manhattan verso ciascun angolo)
    Ritorna (4,2) float32 ordinato tl,tr,br,bl oppure None.
    """
    m = mask_bin.astype(np.uint8)
    h, w = m.shape[:2]

    # Trova il contorno principale
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    hull = cv2.convexHull(contour)
    hull = hull.reshape(-1, 2)
    if len(hull) < 3:
        return None

    # Punti più vicini ai bordi (manhattan verso gli angoli)
    x = hull[:, 0]
    y = hull[:, 1]
    tl_idx = np.argmin(x + y)
    tr_idx = np.argmin((w - 1 - x) + y)
    br_idx = np.argmin((w - 1 - x) + (h - 1 - y))
    bl_idx = np.argmin(x + (h - 1 - y))

    tl = hull[tl_idx]
    tr = hull[tr_idx]
    br = hull[br_idx]
    bl = hull[bl_idx]

    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    return order_points(pts)


# ----------------------------
# Step 2 bis: filtro box per aspect/area
# ----------------------------
def filter_valid_card_boxes(
    boxes: List[Tuple[int, int, int, int]],
    min_aspect: float = CARD_MIN_ASPECT,
    max_aspect: float = CARD_MAX_ASPECT,
    min_area: Optional[int] = None,
    max_area: Optional[int] = None,
    min_area_ratio: Optional[float] = None,
    max_area_ratio: Optional[float] = None,
    image_shape: Optional[Tuple[int, int, int]] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Filtra i bounding box per proporzioni/area.
    Se min_area_ratio/max_area_ratio sono forniti, richiede image_shape.
    """
    valid: List[Tuple[int, int, int, int]] = []
    img_area = None
    if (min_area_ratio is not None or max_area_ratio is not None) and image_shape is not None:
        img_area = image_shape[0] * image_shape[1]

    for (x, y, w, h) in boxes:
        area = w * h
        aspect = w / h if h > 0 else 0
        if not (min_aspect <= aspect <= max_aspect):
            continue
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if img_area is not None:
            area_ratio = area / img_area
            if min_area_ratio is not None and area_ratio < min_area_ratio:
                continue
            if max_area_ratio is not None and area_ratio > max_area_ratio:
                continue
        valid.append((x, y, w, h))
    return valid


# ----------------------------
# Step 3: warp affine + crop stretto
# ----------------------------
def apply_affine_cleanup(
    image: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Riallina la carta usando warp prospettico:
    - preferisce i 4 vertici trovati con approxPolyDP sul contorno
    - fallback su minAreaRect (boxPoints)
    - binarizza e croppa sul contenuto
    """
    x, y, w, h = box
    mask_crop = mask[y : y + h, x : x + w]
    card_crop = image[y : y + h, x : x + w]

    card_masked = cv2.bitwise_and(card_crop, card_crop, mask=mask_crop)

    contour_shifted = contour.reshape(-1, 2).astype(np.float32) - np.array([x, y], dtype=np.float32)
    if contour_shifted.shape[0] < 3:
        return card_masked, mask_crop

    # Binarizza la maschera della carta ritagliata e trova i 4 angoli
    mask_bin = create_white_mask(card_masked)
    corners = corners_from_mask(mask_bin)
    if corners is None or len(corners) != 4:
        # fallback: usa minAreaRect
        rect = cv2.minAreaRect(contour_shifted)
        rect_w, rect_h = rect[1]
        if rect_w < 1 or rect_h < 1:
            return card_masked, mask_crop
        corners = order_points(cv2.boxPoints(rect))

    src_pts = corners

    dst_w, dst_h = IMG_WIDTH, IMG_HEIGHT
    dst = np.float32([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]])

    M = cv2.getPerspectiveTransform(src_pts, dst)
    warped = cv2.warpPerspective(card_masked, M, (dst_w, dst_h))
    warped_mask = cv2.warpPerspective(mask_crop, M, (dst_w, dst_h))

    # Pulizia bordo: binarizza e scansiona dai lati per trovare il contenuto
    _, mask_bin = cv2.threshold(warped_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rows_any = np.where(mask_bin.max(axis=1) > 0)[0]
    cols_any = np.where(mask_bin.max(axis=0) > 0)[0]
    if len(rows_any) > 0 and len(cols_any) > 0:
        top, bottom = rows_any[0], rows_any[-1]
        left, right = cols_any[0], cols_any[-1]
        warped = warped[top : bottom + 1, left : right + 1]
        mask_bin = mask_bin[top : bottom + 1, left : right + 1]

    warped = cv2.resize(warped, (dst_w, dst_h))
    warped_mask = cv2.resize(mask_bin.astype(np.uint8), (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    return warped, warped_mask


# ----------------------------
# Step 4: estrazione carta da immagine intera
# ----------------------------
def crop_card(image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Esegue maschera -> box -> affine -> crop stretto. Ritorna (card, mask, box)."""
    mask = create_white_mask(image)
    region = find_best_card_region(mask, image.shape)
    if region is None:
        return None, mask, None
    box, contour = region
    card_aligned, refined_mask = apply_affine_cleanup(image, mask, contour, box)
    return card_aligned, refined_mask, box


# ----------------------------
# Step 5: estrazione carta grezza da box noto
# ----------------------------
def extract_full_card(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Ritaglia l'intera carta dall'immagine."""
    return image[y : y + h, x : x + w]


# ----------------------------
# Step 6: allineamento di un ritaglio già isolato
# ----------------------------
def affine_align_card(card_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applica maschera+affine ad un ritaglio di carta già isolato.
    Utile quando il bounding box è stato trovato altrove (es. split dealer/player).
    Ritorna (card_allineata, maschera_allineata).
    """
    mask = create_white_mask(card_bgr)
    corners = corners_from_mask(mask)
    if corners is None:
        # fallback: usa contorno principale
        contour_info = find_best_card_region(
            mask,
            card_bgr.shape,
            card_min_aspect=CARD_MIN_ASPECT,
            card_max_aspect=CARD_MAX_ASPECT,
            card_min_area_ratio=CARD_MIN_AREA_RATIO,
            card_max_area_ratio=CARD_MAX_AREA_RATIO,
        )
        if contour_info is None:
            return card_bgr, mask
        box, contour = contour_info
        return apply_affine_cleanup(card_bgr, mask, contour, box)

    # Warp diretto con i corner trovati
    dst_w, dst_h = IMG_WIDTH, IMG_HEIGHT
    dst = np.float32([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]])
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(card_bgr, M, (dst_w, dst_h))
    warped_mask = cv2.warpPerspective(mask, M, (dst_w, dst_h))
    return warped, warped_mask


# ----------------------------
# Step 7: normalizzazione per il modello
# ----------------------------
def normalize_card(card_bgr: np.ndarray, target_size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Ridimensiona a target_size (h, w) e converte in bianco/nero binario."""
    h, w = target_size
    resized = cv2.resize(card_bgr, (w, h))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary[..., None]  # (h,w,1)


def card_to_vector(card_bgr: np.ndarray, target_size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Converte la carta in vettore normalizzato binario flatten per KNN o modelli classici."""
    binary = normalize_card(card_bgr, target_size)  # (h, w, 1)
    return (binary / 255.0).reshape(-1)


# ----------------------------
# Utility: bounds del contenuto in una maschera binaria
# ----------------------------
def compute_mask_bounds(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Ritorna (top, bottom, left, right) della regione non-zero di una maschera binaria.
    Restituisce None se vuota.
    """
    mask_uint8 = mask.astype(np.uint8)
    rows_any = np.where(mask_uint8.max(axis=1) > 0)[0]
    cols_any = np.where(mask_uint8.max(axis=0) > 0)[0]
    if len(rows_any) == 0 or len(cols_any) == 0:
        return None
    top, bottom = rows_any[0], rows_any[-1]
    left, right = cols_any[0], cols_any[-1]
    return top, bottom, left, right


__all__ = [
    "IMG_HEIGHT",
    "IMG_WIDTH",
    "DEFAULT_SIZE",
    "MORPH_KERNEL_SIZE",
    "MORPH_ITERATIONS",
    "CARD_MIN_ASPECT",
    "CARD_MAX_ASPECT",
    "CARD_MIN_AREA_RATIO",
    "CARD_MAX_AREA_RATIO",
    "CARD_MIN_AREA_ABS",
    "CARD_MAX_AREA_ABS",
    "CARD_THRESHOLD",
    "CARD_MIN_CONTOUR_AREA",
    "parse_card_label",
    "create_white_mask",
    "split_image_dealer_player",
    "find_card_contours",
    "find_best_card_region",
    "filter_valid_card_boxes",
    "corners_from_mask",
    "apply_affine_cleanup",
    "crop_card",
    "extract_full_card",
    "affine_align_card",
    "normalize_card",
    "card_to_vector",
    "compute_mask_bounds",
]