import cv2
import numpy as np
import math
import dlib
from FilterBasics import FilterBasics


class HatFilter(FilterBasics):
    def __init__(self):
        FilterBasics.__init__(self)
        self._baseball_cap_img = cv2.imread("../data/hats/baseball_cap.png", cv2.IMREAD_UNCHANGED)
        self._baseball_cap_img = cv2.cvtColor(self._baseball_cap_img, cv2.COLOR_BGRA2RGBA)
        self._icon = self._baseball_cap_img.copy()
        self._variants = ['baseball_cap']
        self._actual_variant = 0

    def apply(self, face_elements: dlib.full_object_detection, img: np.ndarray) -> np.ndarray:
        if self._variants[self._actual_variant] == 'baseball_cap':
            left_hat_landmarks, right_hat_landmarks = self.get_hat_points(face_elements)
            img = self.draw_hat(img, left_hat_landmarks, right_hat_landmarks, self._baseball_cap_img)
        return img

    def get_hat_points(self, face_points) -> (np.array, np.array):
        right_ear_landmarks = np.array([face_points.part(16).x, face_points.part(25).y])
        left_ear_landmarks = np.array([face_points.part(0).x, face_points.part(18).y])
        return left_ear_landmarks, right_ear_landmarks

    def draw_hat(self, img: np.ndarray, left_ear_points: np.array, right_ear_points: np.array,
                 element_img: np.ndarray) -> np.ndarray:
        # Resize and rotate element
        width = int(np.linalg.norm([left_ear_points[0] - right_ear_points[0],
                                    right_ear_points[1] - left_ear_points[1]]))
        height = int(width * element_img.shape[0] / element_img.shape[1])
        img_resized = cv2.resize(element_img, (width, height))
        img_resized, width, height, phi = self._rotate_img_based_on_2points(img_resized,
                                                                            [left_ear_points, right_ear_points])
        # Translate element to correct position
        dx = int(width * math.tan(math.radians(phi)))
        ears_top_left = (left_ear_points[0] + dx, left_ear_points[1] - height)
        img = self._blend_images(img, img_resized, ears_top_left)
        return img
