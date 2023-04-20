import cv2
import numpy as np
from FilterBasics import FilterBasics


class ColorMagicFilter(FilterBasics):
    def __init__(self):
        FilterBasics.__init__(self)
        self._icon = cv2.imread("../data/colors/colors.png", cv2.IMREAD_UNCHANGED)
        self._icon = cv2.cvtColor(self._icon, cv2.COLOR_BGRA2RGBA)
        self._variants = ['swap_rb', 'swap_rg', 'swap_bg', 'shift', 'only_red', 'remove_red']
        self._actual_variant = 0

    def apply(self, img: np.ndarray) -> np.ndarray:
        if self._variants[self._actual_variant] == 'swap_rb':
            img = self._swap_rb(img)
        if self._variants[self._actual_variant] == 'swap_rg':
            img = self._swap_rg(img)
        if self._variants[self._actual_variant] == 'swap_bg':
            img = self._swap_bg(img)
        if self._variants[self._actual_variant] == 'shift':
            img = self._shift(img)
        if self._variants[self._actual_variant] == 'only_red':
            img = self._only_red(img)
        if self._variants[self._actual_variant] == 'remove_red':
            img = self._remove_red(img)
        return img

    def _swap_rb(self, img: np.ndarray) -> np.ndarray:
        tmp = img[:, :, 2].copy()
        img[:, :, 2] = img[:, :, 0].copy()
        img[:, :, 0] = tmp
        return img

    def _swap_rg(self, img: np.ndarray) -> np.ndarray:
        tmp = img[:, :, 0].copy()
        img[:, :, 0] = img[:, :, 1].copy()
        img[:, :, 1] = tmp
        return img

    def _swap_bg(self, img: np.ndarray) -> np.ndarray:
        tmp = img[:, :, 2].copy()
        img[:, :, 2] = img[:, :, 1].copy()
        img[:, :, 1] = tmp
        return img

    def _shift(self, img: np.ndarray) -> np.ndarray:
        tmp = img[:, :, 2].copy()
        img[:, :, 2] = img[:, :, 1].copy()
        img[:, :, 1] = img[:, :, 0].copy()
        img[:, :, 0] = tmp
        return img

    def _only_red(self, img: np.ndarray) -> np.ndarray:
        img[:, :, 1] = np.zeros([img.shape[0], img.shape[1]])
        img[:, :, 2] = np.zeros([img.shape[0], img.shape[1]])
        return img

    def _remove_red(self, img: np.ndarray) -> np.ndarray:
        img[:, :, 0] = np.zeros([img.shape[0], img.shape[1]])
        return img
