import cv2
import dlib
import numpy as np


def image_filter():
    # Load face detection model
    detector = dlib.get_frontal_face_detector()

    # Load landmark detection model
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Load dog nose and ears images
    nose_img = cv2.imread("data/dog/dog_nose.png", cv2.IMREAD_UNCHANGED)
    ears_img = cv2.imread("data/dog/dog_ears.png", cv2.IMREAD_UNCHANGED)
    nose_img[:, :, :3] = nose_img[:, :, 2::-1]
    ears_img[:, :, :3] = ears_img[:, :, 2::-1]

    # Read input image
    img = cv2.imread("input/imput_multiple.jpg")
    # img = cv2.resize(img, [500, 500], interpolation=cv2.INTER_AREA)
    # Convert to RGBA format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    rects = detector(gray, 0)

    # Loop through each face
    for rect in rects:
        # Detect landmarks
        landmarks = predictor(gray, rect)

        # Extract nose and ears landmarks
        nose_landmarks = np.array([(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(32).x, landmarks.part(32).y)])
        right_ear_landmarks = np.array([(landmarks.part(26).x, landmarks.part(26).y), (landmarks.part(16).x, landmarks.part(16).y)])
        left_ear_landmarks = np.array([(landmarks.part(0).x, landmarks.part(0).y), (landmarks.part(17).x, landmarks.part(17).y)])

        # Resize dog nose and ears images to fit face
        nose_width = int(np.linalg.norm(nose_landmarks[0] - nose_landmarks[1]) * 1.5)
        nose_height = int(nose_width * nose_img.shape[0] / nose_img.shape[1])
        nose_img_resized = cv2.resize(nose_img, (nose_width, nose_height))

        ears_width = int(np.linalg.norm(left_ear_landmarks[0] - right_ear_landmarks[1]))
        print(ears_width)
        print(left_ear_landmarks)
        print(right_ear_landmarks)
        ears_height = int(ears_width * ears_img.shape[0] / ears_img.shape[1] * 1.2)
        ears_img_resized = cv2.resize(ears_img, (ears_width, ears_height))

        # Translate dog nose and ears images to face
        nose_center = (landmarks.part(30).x, landmarks.part(30).y)
        nose_top_left = (nose_center[0] - int(nose_width / 2), nose_center[1] - int(nose_height / 2))
        ears_top_left = (landmarks.part(0).x, landmarks.part(0).y - ears_height)

        # Blend dog nose and ears images with input image
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
    # for i in range(0, 68):
    #     point = (landmarks.part(i).x, landmarks.part(i).y)
    #     img = cv2.putText(img, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
    # Display the output
    cv2.imshow("Snapchat Filter", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_filter()
