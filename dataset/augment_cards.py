"""
Data augmentation per le carte in `dataset/carte`.

Per ogni immagine .jpg trovata ricorsivamente in `SOURCE_DIR`, genera copie
augmentate in `OUTPUT_DIR`, mantenendo la stessa struttura di sottocartelle.

Dipendenze: solo OpenCV e NumPy (già usati nei notebook).
"""

from pathlib import Path
import random

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Configurazione
SOURCE_DIR = Path("dataset/carte")
OUTPUT_DIR = Path("dataset/carte_aug")

# 4 esempi per ciascuna delle 4 situazioni -> 16 varianti per immagine
COPIES_PER_MODE = 4
MODES = 4
COPIES_PER_IMAGE = COPIES_PER_MODE * MODES

# Range di trasformazioni
ROT_DEG = (-8, 8)  # rotazione leggera
BRIGHT_FACTOR = (0.85, 1.15)
CONTRAST_FACTOR = (0.85, 1.15)
NOISE_STD = (2, 8)  # rumore gaussiano std in 0..255
SHIFT_PX = (-8, 8)  # traslazione x,y in pixel
PERSPECTIVE_SHIFT = 0.04  # quota della dimensione per warp prospettico
BLUR_KSIZE = (3, 3)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def random_float(a, b):
    return a + (b - a) * random.random()


def augment_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    # Rotazione + traslazione leggera
    angle = random_float(*ROT_DEG)
    tx = random.randint(*SHIFT_PX)
    ty = random.randint(*SHIFT_PX)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += [tx, ty]
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

    # Brightness/contrast
    alpha = random_float(*CONTRAST_FACTOR)
    beta = int(255 * (random_float(*BRIGHT_FACTOR) - 1.0))
    bc = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)

    # Rumore gaussiano
    std = random_float(*NOISE_STD)
    noise = np.random.normal(0, std, bc.shape).astype(np.float32)
    noisy = bc.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def augment_mode(img: np.ndarray, mode: int) -> np.ndarray:
    """
    Quattro modalità distinte per varietà:
    0: rotazione/shift + brightness/contrast leggeri (baseline)
    1: rotazione più marcata + contrast stretch + rumore medio
    2: solo luminanza/contrast + blur leggero + no shift (simula esposizione diversa)
    3: warp prospettico + shift + rumore basso (simula inquadratura inclinata)
    """
    h, w = img.shape[:2]

    if mode == 0:
        return augment_image(img)

    if mode == 1:
        angle = random_float(-12, 12)
        tx = random.randint(-12, 12)
        ty = random.randint(-12, 12)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[:, 2] += [tx, ty]
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        alpha = random_float(0.8, 1.25)
        beta = int(255 * (random_float(0.80, 1.20) - 1.0))
        bc = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)
        std = random_float(4, 12)
        noise = np.random.normal(0, std, bc.shape).astype(np.float32)
        noisy = bc.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    if mode == 2:
        alpha = random_float(0.75, 1.30)
        beta = int(255 * (random_float(0.75, 1.25) - 1.0))
        bc = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        blurred = cv2.GaussianBlur(bc, BLUR_KSIZE, sigmaX=random_float(0.5, 1.5))
        std = random_float(1, 4)
        noise = np.random.normal(0, std, blurred.shape).astype(np.float32)
        noisy = blurred.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    if mode == 3:
        # Warp prospettico leggero
        shift_x = int(w * PERSPECTIVE_SHIFT)
        shift_y = int(h * PERSPECTIVE_SHIFT)
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([
            [random.randint(0, shift_x), random.randint(0, shift_y)],
            [w - random.randint(0, shift_x), random.randint(0, shift_y)],
            [random.randint(0, shift_x), h - random.randint(0, shift_y)],
            [w - random.randint(0, shift_x), h - random.randint(0, shift_y)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        # leggera regolazione colore
        alpha = random_float(0.9, 1.1)
        beta = int(255 * (random_float(0.9, 1.1) - 1.0))
        warped = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)
        std = random_float(1, 5)
        noise = np.random.normal(0, std, warped.shape).astype(np.float32)
        noisy = warped.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    return img.copy()


def main():
    ensure_dir(OUTPUT_DIR)

    images = sorted(SOURCE_DIR.rglob("*.jpg"))
    print(f"Trovate {len(images)} immagini in {SOURCE_DIR}")

    iterator = tqdm(images, desc="Augment") if tqdm else images
    for img_path in iterator:
        rel = img_path.relative_to(SOURCE_DIR)
        out_dir = OUTPUT_DIR / rel.parent
        ensure_dir(out_dir)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Immagine non leggibile: {img_path}")
            continue

        stem = img_path.stem
        ext = img_path.suffix

        for mode in range(MODES):
            for i in range(COPIES_PER_MODE):
                aug = augment_mode(img, mode)
                out_path = out_dir / f"{stem}_m{mode}_aug{i+1}{ext}"
                cv2.imwrite(str(out_path), aug)

    print(f"Augment completato. Output in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()