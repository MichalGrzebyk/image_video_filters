import cv2
import dlib
import numpy as np
import queue
import threading
import time


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
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

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        return self.cap.release()


def bgra_rgba_conversion(image):
    image[:, :, :3] = image[:, :, 2::-1]
    return image


def camera_filter():
    # define a video capture object
    vid = VideoCapture(0)
    states = [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    # Indexes of states:
    # 0 - none, 1 - nose, 2 - ears, 3 - full dog,
    # 4 - aviators, 5 - thug_glasses
    # 6 - rb_change, 7 - rg_change, 8 - gb_change,
    # 9 - shift_colors+1, 10 - shift_colors+2,
    # 11 - remove_blue, 12 - remove_red, 13 - remove_green
    # 14 - baseball_cap,
    actual_state = 0
    # Load face detection model
    detector = dlib.get_frontal_face_detector()

    # Load landmark detection model
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Load dog nose and ears images
    nose_img = cv2.imread("data/dog/dog_nose.png", cv2.IMREAD_UNCHANGED)
    ears_img = cv2.imread("data/dog/dog_ears.png", cv2.IMREAD_UNCHANGED)
    thug_img = cv2.imread("data/glasses/thuglife.png", cv2.IMREAD_UNCHANGED)
    aviators_img = cv2.imread("data/glasses/blue_aviators.png", cv2.IMREAD_UNCHANGED)
    baseball_cap = cv2.imread("data/hats/baseball_cap.png", cv2.IMREAD_UNCHANGED)
    # RGBA to BGRA conversion
    nose_img = bgra_rgba_conversion(nose_img)
    ears_img = bgra_rgba_conversion(ears_img)
    thug_img = bgra_rgba_conversion(thug_img)
    aviators_img = bgra_rgba_conversion(aviators_img)
    baseball_cap = bgra_rgba_conversion(baseball_cap)
    framerate_time = time.time()
    framerate_frames = 0
    framerate = 0
    while vid.isOpened():
        img = vid.read()
        framerate_frames += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        # Detect faces
        face_based = any(states[1:6]) or states[14]
        if face_based:
            face_detection_width = 480
            downscale = int(img.shape[0] / face_detection_width)
            face_img = cv2.resize(img, (int(img.shape[1] / downscale), face_detection_width), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGBA2GRAY)
            rects = detector(gray, 0)
        else:
            rects = []
        # Loop through each face
        for rect in rects:
            # Detect landmarks
            if face_based:
                landmarks = predictor(gray, rect)
            if states[1] or states[3]:
                nose_landmarks, nose_center = preprocess_nose_points(landmarks, downscale)
                img = add_dog_nose(img, nose_landmarks, nose_center, nose_img)
            if states[2] or states[3]:
                left_ear_landmarks, right_ear_landmarks = preprocess_ears_points(landmarks, downscale)
                img = add_hat_or_ears(img, left_ear_landmarks, right_ear_landmarks, ears_img)
            if states[4]:
                temples_landmarks, center = preprocess_temples_points(landmarks, downscale)
                img = add_glasses(img, temples_landmarks, center, aviators_img)
            if states[5]:
                temples_landmarks, center = preprocess_temples_points(landmarks, downscale)
                img = add_glasses(img, temples_landmarks, center, thug_img)
            if states[14]:
                temples_landmarks, center = preprocess_ears_points(landmarks, downscale)
                img = add_hat_or_ears(img, temples_landmarks, center, baseball_cap)
            # debug - show face detection points numbers
            # for i in range(0, 68):
            #     point = (landmarks.part(i).x, landmarks.part(i).y)
            #     img = cv2.putText(img, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
        if states[6]:
            tmp = img[:, :, 2].copy()
            img[:, :, 2] = img[:, :, 0].copy()
            img[:, :, 0] = tmp
        if states[7]:
            tmp = img[:, :, 2].copy()
            img[:, :, 2] = img[:, :, 1].copy()
            img[:, :, 1] = tmp
        if states[8]:
            tmp = img[:, :, 1].copy()
            img[:, :, 1] = img[:, :, 0].copy()
            img[:, :, 0] = tmp
        if states[9]:
            tmp2 = img[:, :, 2].copy()
            img[:, :, 2] = img[:, :, 1].copy()
            img[:, :, 1] = img[:, :, 0].copy()
            img[:, :, 0] = tmp2
        if states[10]:
            tmp0 = img[:, :, 0].copy()
            img[:, :, 0] = img[:, :, 1].copy()
            img[:, :, 1] = img[:, :, 2].copy()
            img[:, :, 2] = tmp0
        if states[11]:
            img[:, :, 2] = np.zeros([img.shape[0], img.shape[1]])
        if states[12]:
            img[:, :, 0] = np.zeros([img.shape[0], img.shape[1]])
        if states[13]:
            img[:, :, 1] = np.zeros([img.shape[0], img.shape[1]])
        # Convert back to BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if framerate_time > 1:
            framerate_prev_time = framerate_time
            framerate_time = time.time()
            framerate = framerate_frames / (framerate_time - framerate_prev_time)
            framerate_frames = 0
        # Display the output
        cv2.putText(img, str(framerate)[:4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Snapchat Filter", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('d'):
            states[actual_state] = False
            actual_state = actual_state + 1 if actual_state < len(states) - 1 else 0
            states[actual_state] = True
        if key == ord('a'):
            states[actual_state] = False
            actual_state = actual_state - 1 if actual_state > 0 else len(states) - 1
            states[actual_state] = True

    vid.release()
    cv2.destroyAllWindows()


def preprocess_nose_points(face_points, scale):
    nose_landmarks = np.array([(face_points.part(36).x, face_points.part(36).y), (face_points.part(32).x, face_points.part(32).y)]) * scale
    nose_center = np.array([face_points.part(30).x, face_points.part(30).y]) * scale
    return nose_landmarks, nose_center


def preprocess_ears_points(face_points, scale):
    right_ear_landmarks = np.array([face_points.part(16).x, face_points.part(25).y]) * scale
    left_ear_landmarks = np.array([face_points.part(0).x, face_points.part(18).y]) * scale
    return right_ear_landmarks, left_ear_landmarks


def preprocess_temples_points(face_points, scale):
    right_temple_landmarks = np.array([face_points.part(16).x, face_points.part(16).y]) * scale
    left_temple_landmarks = np.array([face_points.part(0).x, face_points.part(0).y]) * scale
    between_eyes_point = np.array([face_points.part(28).x, face_points.part(28).y]) * scale
    return [right_temple_landmarks, left_temple_landmarks], between_eyes_point


def add_dog_nose(img, nose_points, nose_center, nose_img):
    # Resize dog nose and ears images to fit face
    nose_width = int(np.linalg.norm(nose_points[0] - nose_points[1]))
    nose_height = int(nose_width * nose_img.shape[0] / nose_img.shape[1])
    nose_img_resized = cv2.resize(nose_img, (nose_width, nose_height))

    # Translate dog nose and ears images to face
    nose_top_left = (nose_center[0] - int(nose_width / 2), nose_center[1] - int(nose_height / 2))

    img = blend_images(img, nose_img_resized, nose_top_left)
    return img


def add_hat_or_ears(img, left_ear_points, right_ear_points, thing_img):
    width = int(np.linalg.norm([left_ear_points[0] - right_ear_points[0], left_ear_points[1] - right_ear_points[1]]))
    height = int(width * thing_img.shape[0] / thing_img.shape[1])
    img_resized = cv2.resize(thing_img, (width, height))

    # Translate dog nose and ears images to face
    ears_top_left = (right_ear_points[0], left_ear_points[1] - height)

    # Blend dog nose and ears images with input image
    img = blend_images(img, img_resized, ears_top_left)
    return img


def add_glasses(img, glasses_points, glasses_center, glasses_img):
    glasses_width = int(np.linalg.norm(glasses_points[0] - glasses_points[1]))
    glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
    glasses_img_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
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


if __name__ == '__main__':
    camera_filter()
