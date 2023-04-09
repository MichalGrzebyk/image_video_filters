import cv2
import imutils
import dlib
import numpy as np
import math
import queue
import threading
import time
import datetime


# debug == True - show face points as numbers on image
debug = False


# global variables used in mouse click callback
global states, variants, actual_state, actual_variant, video_shape
global prev_rectangle, actual_rectangle, next_rectangle


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.resolution = [self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def get_resolution(self):
        return self.resolution

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        return self.cap.release()


def bgra_rgba_conversion(image):
    image[:, :, :3] = image[:, :, 2::-1]
    return image


def change_params(action, x, y, flags, *userdata):
    global states, actual_state, actual_variant, video_shape
    global prev_rectangle, actual_rectangle, next_rectangle, save_img
    if action == cv2.EVENT_LBUTTONDOWN:
        if (prev_rectangle[0][0] < x < prev_rectangle[1][0]) and (prev_rectangle[0][1] < y < prev_rectangle[1][1]):
            actual_state = actual_state - 1 if actual_state > 0 else len(states) - 1
            actual_variant = 0
        elif (next_rectangle[0][0] < x < next_rectangle[1][0]) and (next_rectangle[0][1] < y < next_rectangle[1][1]):
            actual_state = actual_state + 1 if actual_state < len(states) - 1 else 0
            actual_variant = 0
        elif(actual_rectangle[0][0] < x < actual_rectangle[1][0]) and (actual_rectangle[0][1] < y < actual_rectangle[1][1]):
            save_img = True
        else:
            actual_variant = actual_variant + 1 if actual_variant < len(variants[states[actual_state]]) - 1 else 0


def camera_filter():
    # define a video capture object
    vid = VideoCapture(0)
    global states, variants, actual_state, actual_variant, video_shape, save_img
    video_shape = vid.get_resolution()
    # Load images
    nose_img = cv2.imread("data/dog/dog_nose.png", cv2.IMREAD_UNCHANGED)
    ears_img = cv2.imread("data/dog/dog_ears.png", cv2.IMREAD_UNCHANGED)
    thug_img = cv2.imread("data/glasses/thuglife.png", cv2.IMREAD_UNCHANGED)
    aviators_img = cv2.imread("data/glasses/blue_aviators.png", cv2.IMREAD_UNCHANGED)
    baseball_cap = cv2.imread("data/hats/baseball_cap.png", cv2.IMREAD_UNCHANGED)
    color_swap = cv2.imread("data/colors/swap.png", cv2.IMREAD_UNCHANGED)
    colors_shift = cv2.imread("data/colors/shift.png", cv2.IMREAD_UNCHANGED)
    color_remove = cv2.imread("data/colors/remove.png", cv2.IMREAD_UNCHANGED)
    nose_img = bgra_rgba_conversion(nose_img)
    ears_img = bgra_rgba_conversion(ears_img)
    thug_img = bgra_rgba_conversion(thug_img)
    aviators_img = bgra_rgba_conversion(aviators_img)
    baseball_cap = bgra_rgba_conversion(baseball_cap)
    color_swap = bgra_rgba_conversion(color_swap)
    colors_shift = bgra_rgba_conversion(colors_shift)
    color_remove = bgra_rgba_conversion(color_remove)
    save_img = False
    states = ['raw', 'dog', 'glasses', 'hat', 'color_swap', 'colors_shift', 'color_remove']
    variants = {
        'raw': ['raw'],
        'dog': ['nose', 'ears', 'full'],
        'glasses': ['aviators', 'thug'],
        'hat': ['baseball'],
        'color_swap': ['rb', 'rg', 'gb'],
        'colors_shift': ['+1', '+2'],
        'color_remove': ['b', 'r', 'g']
    }
    icons = {
        'raw': np.zeros((1, 1, 4), np.uint8),
        'dog': nose_img,
        'glasses': aviators_img,
        'hat': baseball_cap,
        'color_swap': color_swap,
        'colors_shift': colors_shift,
        'color_remove': color_remove
    }
    actual_state = 0
    actual_variant = 0
    # Load face detection model
    detector = dlib.get_frontal_face_detector()

    # Load landmark detection model
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    framerate_time = time.time()
    framerate_frames = 0
    framerate = 0
    while vid.isOpened():
        img = vid.read()
        framerate_frames += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        # Detect faces
        face_based = 0 < actual_state < 4
        if face_based:
            face_detection_width = 480
            downscale = int(img.shape[0] / face_detection_width)
            face_img = cv2.resize(img, (int(img.shape[1] / downscale), face_detection_width), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGBA2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                # Detect landmarks
                if face_based:
                    landmarks = predictor(gray, rect)
                if (states[actual_state] == 'dog') and\
                        ((variants[states[actual_state]][actual_variant] == 'nose') or (variants[states[actual_state]][actual_variant] == 'full')):
                    nose_landmarks, nose_center = preprocess_nose_points(landmarks, downscale)
                    img = add_dog_nose(img, nose_landmarks, nose_center, nose_img)
                if (states[actual_state] == 'dog') and\
                        ((variants[states[actual_state]][actual_variant] == 'ears') or (variants[states[actual_state]][actual_variant] == 'full')):
                    left_ear_landmarks, right_ear_landmarks, left_bottom_point = preprocess_ears_points(landmarks, downscale)
                    img = add_hat_or_ears(img, left_ear_landmarks, right_ear_landmarks, left_bottom_point, ears_img)
                if states[actual_state] == 'glasses':
                    temples_landmarks, center = preprocess_temples_points(landmarks, downscale)
                    glasses = aviators_img if variants[states[actual_state]][actual_variant] == 'aviators' else thug_img
                    img = add_glasses(img, temples_landmarks, center, glasses)
                if states[actual_state] == 'hat':
                    left_ear_landmarks, right_ear_landmarks, center = preprocess_ears_points(landmarks, downscale)
                    img = add_hat_or_ears(img, left_ear_landmarks, right_ear_landmarks, center, baseball_cap)
                if debug:
                    for i in range(0, 68):
                        point = (landmarks.part(i).x, landmarks.part(i).y)
                        img = cv2.putText(img, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
        if (states[actual_state] == 'color_swap') and (variants[states[actual_state]][actual_variant] == 'rb'):
            tmp = img[:, :, 2].copy()
            img[:, :, 2] = img[:, :, 0].copy()
            img[:, :, 0] = tmp
        if (states[actual_state] == 'color_swap') and (variants[states[actual_state]][actual_variant] == 'rg'):
            tmp = img[:, :, 0].copy()
            img[:, :, 0] = img[:, :, 1].copy()
            img[:, :, 1] = tmp
        if (states[actual_state] == 'color_swap') and (variants[states[actual_state]][actual_variant] == 'gb'):
            tmp = img[:, :, 2].copy()
            img[:, :, 2] = img[:, :, 1].copy()
            img[:, :, 1] = tmp
        if (states[actual_state] == 'colors_shift') and (variants[states[actual_state]][actual_variant] == '+1'):
            tmp2 = img[:, :, 2].copy()
            img[:, :, 2] = img[:, :, 1].copy()
            img[:, :, 1] = img[:, :, 0].copy()
            img[:, :, 0] = tmp2
        if (states[actual_state] == 'colors_shift') and (variants[states[actual_state]][actual_variant] == '+2'):
            tmp0 = img[:, :, 0].copy()
            img[:, :, 0] = img[:, :, 1].copy()
            img[:, :, 1] = img[:, :, 2].copy()
            img[:, :, 2] = tmp0
        if (states[actual_state] == 'color_remove') and (variants[states[actual_state]][actual_variant] == 'b'):
            img[:, :, 2] = np.zeros([img.shape[0], img.shape[1]])
        if (states[actual_state] == 'color_remove') and (variants[states[actual_state]][actual_variant] == 'r'):
            img[:, :, 0] = np.zeros([img.shape[0], img.shape[1]])
        if (states[actual_state] == 'color_remove') and (variants[states[actual_state]][actual_variant] == 'g'):
            img[:, :, 1] = np.zeros([img.shape[0], img.shape[1]])
        icon0 = icons[states[actual_state - 1]] if actual_state != 0 else icons[states[-1]]
        icon1 = icons[states[actual_state]]
        icon2 = icons[states[actual_state + 1]] if actual_state < len(states) - 1 else icons[states[0]]
        img = add_gui(img, icon0, icon1, icon2)
        # Convert back to BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if framerate_time > 1:
            framerate_prev_time = framerate_time
            framerate_time = time.time()
            framerate = framerate_frames / (framerate_time - framerate_prev_time)
            framerate_frames = 0
        cv2.putText(img, str(framerate)[:4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Snapchat Filter", img)
        if save_img:
            cv2.imwrite(f'saved_images/image_{datetime.datetime.now().strftime("%H_%M_%S")}.jpg', img)
            save_img = False
            print("IMAGE SAVED!")
        cv2.setMouseCallback("Snapchat Filter", change_params)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def preprocess_nose_points(face_points, scale):
    nose_landmarks = np.array([(face_points.part(35).x, face_points.part(35).y), (face_points.part(31).x, face_points.part(31).y)]) * scale
    nose_center = np.array([face_points.part(30).x, face_points.part(30).y]) * scale
    return nose_landmarks, nose_center


def preprocess_ears_points(face_points, scale):
    right_ear_landmarks = np.array([face_points.part(16).x, face_points.part(25).y]) * scale
    left_ear_landmarks = np.array([face_points.part(0).x, face_points.part(18).y]) * scale
    left_bottom_point = [left_ear_landmarks[0], min(right_ear_landmarks[1], left_ear_landmarks[1])]
    return right_ear_landmarks, left_ear_landmarks, left_bottom_point


def preprocess_temples_points(face_points, scale):
    right_temple_landmarks = np.array([face_points.part(16).x, face_points.part(16).y]) * scale
    left_temple_landmarks = np.array([face_points.part(0).x, face_points.part(0).y]) * scale
    between_eyes_point = np.array([face_points.part(28).x, face_points.part(28).y]) * scale
    return [right_temple_landmarks, left_temple_landmarks], between_eyes_point


def add_dog_nose(img, nose_points, nose_center, nose_img):
    # Resize dog nose and ears images to fit face
    nose_width = int(np.linalg.norm(nose_points[0] - nose_points[1])) * 2
    nose_height = int(nose_width * nose_img.shape[0] / nose_img.shape[1])
    nose_img_resized = cv2.resize(nose_img, (nose_width, nose_height))
    nose_img_resized, nose_width, nose_height = rotate_img_based_on_2points(nose_img_resized, nose_points)
    # Translate dog nose and ears images to face
    nose_top_left = (nose_center[0] - int(nose_width / 2), nose_center[1] - int(nose_height / 2))

    img = blend_images(img, nose_img_resized, nose_top_left)
    return img


def add_hat_or_ears(img, left_ear_points, right_ear_points, left_bottom_point, thing_img):
    width = int(np.linalg.norm([left_ear_points[0] - right_ear_points[0], left_ear_points[1] - right_ear_points[1]]))
    height = int(width * thing_img.shape[0] / thing_img.shape[1])
    img_resized = cv2.resize(thing_img, (width, height))
    img_resized, width, height = rotate_img_based_on_2points(img_resized, [left_ear_points, right_ear_points], point='left_bottom')
    # Translate dog nose and ears images to face
    ears_top_left = (left_bottom_point[0], left_bottom_point[1] - height)
    # Blend dog nose and ears images with input image
    img = blend_images(img, img_resized, ears_top_left)
    return img


def add_glasses(img, glasses_points, glasses_center, glasses_img):
    glasses_width = int(np.linalg.norm(glasses_points[0] - glasses_points[1]))
    glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
    glasses_img_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
    glasses_img_resized, glasses_width, glasses_height = rotate_img_based_on_2points(glasses_img_resized, glasses_points)
    glasses_top_left = (glasses_center[0] - int(glasses_width / 2), glasses_center[1] - int(glasses_height / 2))
    img = blend_images(img, glasses_img_resized, glasses_top_left)
    return img


def blend_images(base_img, img_to_add, top_left_point):
    for i in range(img_to_add.shape[0]):
        for j in range(img_to_add.shape[1]):
            if (img_to_add[i, j, 3] != 0) \
                    and (0 < top_left_point[1] + i < base_img.shape[0]) \
                    and (0 < top_left_point[0] + j < base_img.shape[1]):
                base_img[top_left_point[1] + i, top_left_point[0] + j, :] = img_to_add[i, j, :]
    return base_img


def rotate_img_based_on_2points(img, points, point='center'):
    dx = points[0][0] - points[1][0]
    dy = -(points[0][1] - points[1][1])
    alpha = math.degrees(math.atan2(dy, dx))
    rotation = - alpha
    if point == 'center':
        img = imutils.rotate_bound(img, rotation)
    elif point == 'left_bottom':
        M = cv2.getRotationMatrix2D(center=(0, img.shape[1] - 1), angle=alpha, scale=1)
        img = cv2.warpAffine(img, M, [img.shape[1], img.shape[0]])
    width = img.shape[1]
    height = img.shape[0]
    return img, width, height


def add_gui(img, prev_icon, actual_icon, next_icon):
    global video_shape
    global prev_rectangle, actual_rectangle, next_rectangle
    icon_width = 100
    rect_size = 110
    for i, icon in enumerate([prev_icon, actual_icon, next_icon]):
        icon_height = int(icon.shape[0] * icon_width / icon.shape[1])
        icon = cv2.resize(icon, (icon_width, icon_height))
        pt_icon_top_left = int((i + 1) * 0.25 * video_shape[0]) - int(rect_size / 2) + int((rect_size - icon_width) / 2), \
            int(0.9 * video_shape[1] - int(rect_size / 2) - int(icon_height / 2))
        pt_rect_1 = int((i + 1) * 0.25 * video_shape[0]) - int(rect_size / 2), int(0.9 * video_shape[1] - rect_size)
        pt_rect_2 = int((i + 1) * 0.25 * video_shape[0]) - int(rect_size / 2) + rect_size, int(0.9 * video_shape[1])
        img = blend_images(img, icon, pt_icon_top_left)
        color = [255, 255, 255] if i == 1 else [75, 75, 75]
        img = cv2.rectangle(img, pt_rect_1, pt_rect_2, color, thickness=3)
    prev_rectangle, actual_rectangle, next_rectangle = [[[int((i + 1) * 0.25 * video_shape[0]) - int(rect_size / 2),
                                                        int(0.9 * video_shape[1] - rect_size)],
                                                        [int((i + 1) * 0.25 * video_shape[0]) - int(rect_size / 2) + rect_size,
                                                        int(0.9 * video_shape[1])]] for i in range(0, 3)]
    return img


if __name__ == '__main__':
    camera_filter()
