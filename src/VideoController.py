import cv2
import time
import datetime
import dlib
from DogFilter import DogFilter
from HatFilter import HatFilter
from GlassesFilter import GlassesFilter
from ColorMagicFilter import ColorMagicFilter
from FilterBasics import FilterBasics
from Gui import Gui
from VideoCapture import VideoCapture


# main class
class VideoController:
    def __init__(self):
        self._filters = [FilterBasics(), DogFilter(), HatFilter(), GlassesFilter(), ColorMagicFilter()]
        self._actual_state = 0
        self._vid = VideoCapture(0)
        self._video_shape = self._vid.get_resolution()
        self._gui = Gui(self._video_shape)
        self._icons = [self._filters[i].get_icon() for i in range(0, len(self._filters))]

    def main_process(self):
        # Load face detection model
        detector = dlib.get_frontal_face_detector()

        # Load landmark detection model
        predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

        framerate_time = time.time()
        framerate_frames = 0
        framerate = 0
        while self._vid.isOpened():
            img = self._vid.read()
            framerate_frames += 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            # Detect faces
            face_based = 0 < self._actual_state < 4
            if face_based:
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                rects = detector(gray, 0)
                for rect in rects:
                    # Detect landmarks
                    landmarks = predictor(gray, rect)
                    self._filters[self._actual_state].apply(landmarks, img)
            elif self._actual_state != 0:
                self._filters[self._actual_state].apply(img)
            icon0 = self._icons[self._actual_state - 1] if self._actual_state != 0 else self._icons[-1]
            icon1 = self._icons[self._actual_state]
            icon2 = self._icons[self._actual_state + 1] if self._actual_state < len(self._filters) - 1 else \
                self._icons[0]
            img_to_save = img.copy()
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGBA2BGR)
            img = self._gui.draw_gui(img, [icon0, icon1, icon2])
            # Convert back to BGR format
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            if framerate_time > 1:
                framerate_prev_time = framerate_time
                framerate_time = time.time()
                framerate = framerate_frames / (framerate_time - framerate_prev_time)
                framerate_frames = 0
            cv2.putText(img, str(framerate)[:4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Snapchat Filter", img)
            if self._gui.save_img:
                cv2.imwrite(f'saved_images/image_{datetime.datetime.now().strftime("%H_%M_%S")}.jpg', img_to_save)
                self._gui.save_img = False
                print("IMAGE SAVED!")
            if self._gui.next_filter:
                self._increment_actual_state()
                self._gui.next_filter = False
            elif self._gui.prev_filter:
                self._decrement_actual_state()
                self._gui.prev_filter = False
            elif self._gui.next_variant:
                self._filters[self._actual_state].change_actual_variant(set_to_zero=False)
                self._gui.next_variant = False
            cv2.setMouseCallback("Snapchat Filter", self._gui.mouse_callback)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self._vid.release()
        cv2.destroyAllWindows()

    def _increment_actual_state(self):
        self._filters[self._actual_state].change_actual_variant(set_to_zero=True)
        if self._actual_state >= len(self._filters) - 1:
            self._actual_state = 0
        else:
            self._actual_state += 1
        self._filters[self._actual_state].change_actual_variant(set_to_zero=True)

    def _decrement_actual_state(self):
        self._filters[self._actual_state].change_actual_variant(set_to_zero=True)
        if self._actual_state <= 0:
            self._actual_state = len(self._filters) - 1
        else:
            self._actual_state -= 1
        self._filters[self._actual_state].change_actual_variant(set_to_zero=True)
