import numpy as np
from typing import List
from ImageProcesser import ImageProcesser


class FilterBasics(ImageProcesser):
    def __init__(self):
        ImageProcesser.__init__(self)
        self._icon = np.zeros((1, 1, 4), np.uint8)
        self._variants = ['None']
        self._actual_variant = 0

    def get_variants(self) -> List[str]:
        return self._variants

    def change_actual_variant(self, set_to_zero: bool = False):
        if (self._actual_variant >= len(self._variants) - 1) or (set_to_zero):
            self._actual_variant = 0
        else:
            self._actual_variant += 1

    def get_icon(self) -> np.ndarray:
        return self._icon
