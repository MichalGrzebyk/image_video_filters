import cv2
import numpy as np
import math
import dlib
from FilterBasics import FilterBasics


class DogFilter(FilterBasics):
    def __init__(self):
        FilterBasics.__init__(self)
        self._nose_img = cv2.imread("../data/dog/dog_nose.png", cv2.IMREAD_UNCHANGED)
        self._ears_img = cv2.imread("../data/dog/dog_ears.png", cv2.IMREAD_UNCHANGED)
        self._nose_img = cv2.cvtColor(self._nose_img, cv2.COLOR_BGRA2RGBA)
        self._ears_img = cv2.cvtColor(self._ears_img, cv2.COLOR_BGRA2RGBA)
        self._icon = self._nose_img.copy()
        self._variants = ['nose', 'ears', 'all']
        self._actual_variant = 0

    def apply(self, face_elements: dlib.full_object_detection, img: np.ndarray) -> np.ndarray:
        if self._variants[self._actual_variant] in ['nose', 'all']:
            nose_landmarks, nose_center = self.get_nose_points(face_elements)
            img = self.draw_nose(img, nose_landmarks, nose_center, self._nose_img)
        if self._variants[self._actual_variant] in ['ears', 'all']:
            left_ear_landmarks, right_ear_landmarks = self.get_ears_points(face_elements)
            img = self.draw_ears(img, left_ear_landmarks, right_ear_landmarks, self._ears_img)
        return img

    def get_nose_points(self, face_points: dlib.full_object_detection) -> (np.array, np.array):
        landmarks = np.array([(face_points.part(31).x, face_points.part(31).y),
                              (face_points.part(35).x, face_points.part(35).y)])
        center = np.array([face_points.part(30).x, face_points.part(30).y])
        return landmarks, center

    def get_ears_points(self, face_points: dlib.full_object_detection) -> (np.array, np.array):
        right_ear_landmarks = np.array([face_points.part(16).x, face_points.part(25).y])
        left_ear_landmarks = np.array([face_points.part(0).x, face_points.part(18).y])
        return left_ear_landmarks, right_ear_landmarks

    def draw_nose(self, img: np.ndarray, points: np.array, center: np.array, element_img: np.ndarray) -> np.ndarray:
        # Resize and rotate element
        width = int(np.linalg.norm(points[0] - points[1])) * 2
        height = int(width * element_img.shape[0] / element_img.shape[1])
        img_resized = cv2.resize(element_img, (width, height))
        img_resized, nose_width, nose_height, phi = self._rotate_img_based_on_2points(img_resized, points)
        # Translate element to correct position
        top_left = (center[0] - int(nose_width / 2), center[1] - int(nose_height / 2))
        img = self._blend_images(img, img_resized, top_left)
        return img

    def draw_ears(self, img: np.ndarray, left_ear_points: np.array, right_ear_points: np.array,
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
