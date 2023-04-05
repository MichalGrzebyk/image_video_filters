import cv2
import dlib
import numpy as np
import queue
import threading


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
          self.q.get_nowait()   # discard previous (unprocessed) frame
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
    # Load face detection model
    detector = dlib.get_frontal_face_detector()

    # Load landmark detection model
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Load dog nose and ears images
    nose_img = cv2.imread("data/dog/dog_nose.png", cv2.IMREAD_UNCHANGED)
    ears_img = cv2.imread("data/dog/dog_ears.png", cv2.IMREAD_UNCHANGED)
    nose_img[:, :, :3] = nose_img[:, :, 2::-1]
    ears_img[:, :, :3] = ears_img[:, :, 2::-1]
    while vid.isOpened():
        img = vid.read()

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
            landmarks = landmarks
            # Extract nose and ears landmarks
            nose_landmarks = np.array([(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(32).x, landmarks.part(32).y)]) * downscale
            nose_center = np.array([landmarks.part(30).x, landmarks.part(30).y]) * downscale
            right_ear_landmarks = np.array([landmarks.part(16).x, landmarks.part(25).y]) * downscale
            left_ear_landmarks = np.array([landmarks.part(0).x, landmarks.part(18).y]) * downscale

            # TODO! check if coordinates are in range of image.shape

            # Resize dog nose and ears images to fit face
            nose_width = int(np.linalg.norm(nose_landmarks[0] - nose_landmarks[1]) * 2)
            nose_height = int(nose_width * nose_img.shape[0] / nose_img.shape[1])
            nose_img_resized = cv2.resize(nose_img, (nose_width, nose_height))

            ears_width = int(np.linalg.norm(
                [left_ear_landmarks[0] - right_ear_landmarks[0],
                 left_ear_landmarks[1] - right_ear_landmarks[1]]))
            ears_height = int(ears_width * ears_img.shape[0] / ears_img.shape[1])
            ears_img_resized = cv2.resize(ears_img, (ears_width, ears_height))

            # Translate dog nose and ears images to face
            nose_top_left = (nose_center[0] - int(nose_width / 2), nose_center[1] - int(nose_height / 2))
            ears_top_left = (left_ear_landmarks[0], left_ear_landmarks[1] - ears_height)

            # Blend dog nose and ears images with input image
            # TODO! check if coordinates are in range of image.shape
            for i in range(nose_img_resized.shape[0]):
                for j in range(nose_img_resized.shape[1]):
                    if nose_img_resized[i, j, 3] != 0:
                        img[nose_top_left[1] + i, nose_top_left[0] + j, :] = nose_img_resized[i, j, :]

            for i in range(ears_img_resized.shape[0]):
                for j in range(ears_img_resized.shape[1]):
                    if ears_img_resized[i, j, 3] != 0:
                        img[ears_top_left[1] + i, ears_top_left[0] + j, :] = ears_img_resized[i, j, :]

        # Convert back to BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # TODO! add framerate info

        # Display the output
        cv2.imshow("Snapchat Filter", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera_filter()
