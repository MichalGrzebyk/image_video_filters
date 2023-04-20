import cv2
import numpy as np
import dlib
from typing import List

from FilterBasics import FilterBasics


class GlassesFilter(FilterBasics):
    def __init__(self):
        FilterBasics.__init__(self)
        self._thug_img = cv2.imread("../data/glasses/thuglife.png", cv2.IMREAD_UNCHANGED)
        self._aviators_img = cv2.imread("../data/glasses/blue_aviators.png", cv2.IMREAD_UNCHANGED)
        self._thug_img = cv2.cvtColor(self._thug_img, cv2.COLOR_BGRA2RGBA)
        self._aviators_img = cv2.cvtColor(self._aviators_img, cv2.COLOR_BGRA2RGBA)
        self._icon = self._thug_img.copy()
        self._variants = ['blue_aviators', 'thuglife']
        self._actual_variant = 0

    def apply(self, face_elements: dlib.full_object_detection, img: np.ndarray) -> np.ndarray:
        image = self._aviators_img if self._variants[self._actual_variant] == 'blue_aviators' else \
            self._thug_img if self._variants[self._actual_variant] == 'thuglife' else None
        if image is not None:
            left_ear_landmarks, right_ear_landmarks = self.get_temples_points(face_elements)
            img = self.draw_glasses(img, left_ear_landmarks, right_ear_landmarks, image)
        return img

    def get_temples_points(self, face_points: dlib.full_object_detection) -> (List[np.array], np.array):
        right_temple_landmarks = np.array([face_points.part(16).x, face_points.part(16).y])
        left_temple_landmarks = np.array([face_points.part(0).x, face_points.part(0).y])
        between_eyes_point = np.array([face_points.part(28).x, face_points.part(28).y])
        return [left_temple_landmarks, right_temple_landmarks], between_eyes_point

    def draw_glasses(self, img: np.ndarray, glasses_points, glasses_center, glasses_img: np.ndarray) -> np.ndarray:
        # Resize and rotate element
        glasses_width = int(np.linalg.norm(glasses_points[0] - glasses_points[1]))
        glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
        glasses_img_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
        glasses_img_resized, glasses_width, glasses_height, phi = self._rotate_img_based_on_2points(glasses_img_resized,
                                                                                                    glasses_points)
        # Translate element to correct position
        glasses_top_left = (glasses_center[0] - int(glasses_width / 2), glasses_center[1] - int(glasses_height / 2))
        img = self._blend_images(img, glasses_img_resized, glasses_top_left)
        return img
