# utils/color_model.py
import numpy as np
from sklearn.linear_model import Ridge
from skimage.color import rgb2lab

class PaintColorModel:
    """
    Lernt: Ziel-Farbe (sRGB) -> Maschinen-Parameter (Pigment/Tool/etc.).
    V1: Ridge-Regression auf LAB. Später: echte Kalibrier-Daten einspeisen.
    """
    def __init__(self, alpha: float = 1.0):
        self.reg = Ridge(alpha=alpha)
        self._fitted = False

    def fit_dummy(self):
        # Platzhalter: Identität für Demo (keine echte Kalibrierung)
        X = np.array([[50,0,0],[80,0,0],[20,0,0]], dtype=float)
        y = np.array([[0.5,0.5,0.0],[0.8,0.2,0.0],[0.1,0.9,0.0]], dtype=float)
        self.reg.fit(X, y)
        self._fitted = True

    def predict_params_from_srgb01(self, srgb01: np.ndarray) -> np.ndarray:
        """ srgb01: (...,3) in [0,1] → Maschinen-Param-Matrix (...,M) """
        if not self._fitted:
            self.fit_dummy()
        lab = rgb2lab(srgb01.reshape(-1,1,3)).reshape(-1,3)   # (N,3)
        out = self.reg.predict(lab)                            # (N,M)
        return out.reshape(srgb01.shape[:-1] + (out.shape[-1],))
