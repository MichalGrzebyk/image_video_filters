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


def camera_filter():
    # define a video capture object
    vid = VideoCapture(0)
    nose = True
    ears = True
    # Load face detection model
    detector = dlib.get_frontal_face_detector()

    # Load landmark detection model
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Load dog nose and ears images
    nose_img = cv2.imread("data/dog/dog_nose.png", cv2.IMREAD_UNCHANGED)
    ears_img = cv2.imread("data/dog/dog_ears.png", cv2.IMREAD_UNCHANGED)
    # RRBA to BGRA conversion
    nose_img[:, :, :3] = nose_img[:, :, 2::-1]
    ears_img[:, :, :3] = ears_img[:, :, 2::-1]
    framerate_time = time.time()
    framerate_frames = 0
    framerate = 0
    while vid.isOpened():
        img = vid.read()
        framerate_frames += 1

        # Convert to RGBA format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        # Detect faces
        face_detection_width = 480
        downscale = int(img.shape[0] / face_detection_width)
        face_img = cv2.resize(img, (int(img.shape[1] / downscale), face_detection_width), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGBA2GRAY)
        rects = detector(gray, 0)
        # Loop through each face
        for rect in rects:
            # Detect landmarks
            landmarks = predictor(gray, rect)
            if nose:
                nose_landmarks, nose_center = preprocess_face_points(landmarks, 'dog_nose', downscale)
                img = add_dog_nose(img, nose_landmarks, nose_center, nose_img)
            if ears:
                left_ear_landmarks, right_ear_landmarks = preprocess_face_points(landmarks, 'dog_ears', downscale)
                img = add_dog_ears(img, left_ear_landmarks, right_ear_landmarks, ears_img)

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def preprocess_face_points(face_points, filter_name, scale):
    if filter_name == 'dog_nose':
        nose_landmarks = np.array([(face_points.part(36).x, face_points.part(36).y), (face_points.part(32).x, face_points.part(32).y)]) * scale
        nose_center = np.array([face_points.part(30).x, face_points.part(30).y]) * scale
        return nose_landmarks, nose_center

    if filter_name == 'dog_ears':
        right_ear_landmarks = np.array([face_points.part(16).x, face_points.part(25).y]) * scale
        left_ear_landmarks = np.array([face_points.part(0).x, face_points.part(18).y]) * scale
        return right_ear_landmarks, left_ear_landmarks

    #for future usage:
    if filter_name == 'sunglases':
        return face_points
    if filter_name == 'tongue':
        return face_points
    raise Exception("Wrong parameter: filter_name")


def add_dog_nose(img, nose_points, nose_center, nose_img):
    # Resize dog nose and ears images to fit face
    nose_width = int(np.linalg.norm(nose_points[0] - nose_points[1]) * 2)
    nose_height = int(nose_width * nose_img.shape[0] / nose_img.shape[1])
    nose_img_resized = cv2.resize(nose_img, (nose_width, nose_height))

    # Translate dog nose and ears images to face
    nose_top_left = (nose_center[0] - int(nose_width / 2), nose_center[1] - int(nose_height / 2))

    # Blend dog nose and ears images with input image
    for i in range(nose_img_resized.shape[0]):
        for j in range(nose_img_resized.shape[1]):
            if (nose_img_resized[i, j, 3] != 0) and \
                    (0 <= nose_top_left[1] + i < img.shape[0]) and \
                    (0 <= nose_top_left[0] + j < img.shape[1]):
                img[nose_top_left[1] + i, nose_top_left[0] + j, :] = nose_img_resized[i, j, :]
    return img


def add_dog_ears(img, left_ear_points, right_ear_points, ears_img):
    # Resize dog nose and ears images to fit face
    ears_width = int(np.linalg.norm([left_ear_points[0] - right_ear_points[0], left_ear_points[1] - right_ear_points[1]]))
    ears_height = int(ears_width * ears_img.shape[0] / ears_img.shape[1])
    ears_img_resized = cv2.resize(ears_img, (ears_width, ears_height))

    # Translate dog nose and ears images to face
    ears_top_left = (right_ear_points[0], left_ear_points[1] - ears_height)

    # Blend dog nose and ears images with input image
    for i in range(ears_img_resized.shape[0]):
        for j in range(ears_img_resized.shape[1]):
            if (ears_img_resized[i, j, 3] != 0) \
                    and (0 < ears_top_left[1] + i < img.shape[0]) \
                    and (0 < ears_top_left[0] + j < img.shape[1]):
                img[ears_top_left[1] + i, ears_top_left[0] + j, :] = ears_img_resized[i, j, :]
    return img


if __name__ == '__main__':
    camera_filter()
