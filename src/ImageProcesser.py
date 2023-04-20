import math
import numpy as np
import imutils
from typing import List


class ImageProcesser:
    def __init__(self):
        pass

    def _blend_images(self, base_img: np.ndarray, img_to_add: np.ndarray, top_left_point: np.array) -> np.ndarray:
        for i in range(img_to_add.shape[0]):
            for j in range(img_to_add.shape[1]):
                if (img_to_add[i, j, 3] != 0) \
                        and (0 < top_left_point[1] + i < base_img.shape[0]) \
                        and (0 < top_left_point[0] + j < base_img.shape[1]):
                    base_img[top_left_point[1] + i, top_left_point[0] + j, :] = img_to_add[i, j, :]
        return base_img

    def _rotate_img_based_on_2points(self, img: np.ndarray, points: List[np.array]) -> (np.ndarray, int, int, float):
        dx = points[0][0] - points[1][0]
        dy = -(points[0][1] - points[1][1])
        alpha = math.degrees(math.atan2(dy, dx))
        rotation = 180 - alpha
        img = imutils.rotate_bound(img, rotation)
        width = img.shape[1]
        height = img.shape[0]
        return img, width, height, rotation
