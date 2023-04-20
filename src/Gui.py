import cv2
import numpy as np
from typing import List

from ImageProcesser import ImageProcesser


class Gui(ImageProcesser):
    def __init__(self, resolution: [int, int]):
        ImageProcesser.__init__(self)
        self._video_shape = resolution
        self._icon_width = 100
        self._element_size = 110
        self._elements_coordinates = \
            [[[int((i + 1) * 0.25 * self._video_shape[0]) - int(self._element_size / 2),
               int(0.9 * self._video_shape[1] - self._element_size)],
              [int((i + 1) * 0.25 * self._video_shape[0]) - int(self._element_size / 2) + self._element_size,
               int(0.9 * self._video_shape[1])]] for i in range(0, 3)]
        self.save_img = False
        self.next_filter = False
        self.prev_filter = False
        self.next_variant = False

    def get_elements(self) -> List[List[List[int]]]:
        return self._elements_coordinates

    def draw_gui(self, img: np.ndarray, icons: List[np.ndarray]) -> np.ndarray:
        for i, icon in enumerate(icons):
            icon_height = int(icon.shape[0] * self._icon_width / icon.shape[1])
            icon = cv2.resize(icon, (self._icon_width, icon_height))
            pt_icon_top_left = int((i + 1) * 0.25 * self._video_shape[0]) - int(self._element_size / 2) + \
                int((self._element_size - self._icon_width) / 2), \
                int(0.9 * self._video_shape[1] - int(self._element_size / 2) - int(icon_height / 2))
            pt_rect_1, pt_rect_2 = self._elements_coordinates[i]

            img = self._blend_images(img, icon, pt_icon_top_left)
            color = [255, 255, 255] if i == 1 else [75, 75, 75]  # white for active filter, gray for next and prev
            img = cv2.rectangle(img, pt_rect_1, pt_rect_2, color, thickness=3)
        return img

    def mouse_callback(self, action: int, x: int, y: int, flags: int, *userdata: tuple):
        prev_rectangle, actual_rectangle, next_rectangle = self.get_elements()
        if action == cv2.EVENT_LBUTTONDOWN:
            if (prev_rectangle[0][0] < x < prev_rectangle[1][0]) and (prev_rectangle[0][1] < y < prev_rectangle[1][1]):
                self.prev_filter = True
            elif (next_rectangle[0][0] < x < next_rectangle[1][0]) and (
                    next_rectangle[0][1] < y < next_rectangle[1][1]):
                self.next_filter = True
            elif (actual_rectangle[0][0] < x < actual_rectangle[1][0]) and (
                    actual_rectangle[0][1] < y < actual_rectangle[1][1]):
                self.save_img = True
            else:
                self.next_variant = True
