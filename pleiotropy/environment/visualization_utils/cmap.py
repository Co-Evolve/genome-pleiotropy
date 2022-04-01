from pathlib import Path

import cv2
import numpy as np

pth = str(Path(__file__).parent.resolve() / "cmap.jpg")
IMG = cv2.imread(pth)[::-1, 0, ::-1].astype(np.uint8)


def cmap(x):
    x = max(0., min(1., x))
    assert 0 <= x <= 1

    ii = min(255, int(x * 256))
    return IMG[ii]
