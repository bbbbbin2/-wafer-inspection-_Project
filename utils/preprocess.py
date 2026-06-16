import numpy as np
import cv2


# ==================================================
# P2
# Resize + Normalization
# ==================================================
def preprocess_p2(x):

    x = cv2.resize(
        x,
        (64, 64),
        interpolation=cv2.INTER_NEAREST
    )

    x = x.astype(np.float32) / 2.0

    return x


# ==================================================
# P3
# Resize + Median Filter + Normalization
# ==================================================
def preprocess_p3(x):

    x = cv2.resize(
        x,
        (64, 64),
        interpolation=cv2.INTER_NEAREST
    )

    x = cv2.medianBlur(
        x.astype(np.uint8),
        3
    )

    x = x.astype(np.float32) / 2.0

    return x


# ==================================================
# P4
# Resize + Data Augmentation + Normalization
# ==================================================
def preprocess_p4(x):

    x = cv2.resize(
        x,
        (64, 64),
        interpolation=cv2.INTER_NEAREST
    )

    if np.random.rand() < 0.3:

        noise = np.random.normal(
            0,
            0.1,
            x.shape
        )

        x = x + noise

    x = np.clip(x, 0, 2)

    x = x.astype(np.float32) / 2.0

    return x


# ==================================================
# P5
# Resize + Normalization
# Class Weight는 학습단에서 적용
# ==================================================
def preprocess_p5(x):

    x = cv2.resize(
        x,
        (64, 64),
        interpolation=cv2.INTER_NEAREST
    )

    x = x.astype(np.float32) / 2.0

    return x
