# from keras.preprocessing.image import load_img
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.inference import detect_faces
from utils.inference import draw_bounding_box
from utils.inference import draw_text
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input

import cv2

WEIGHTS_EMOTION="./res/saved_weights/emotion_model.hdf5"
HAAR_CASCADE="./res/opencv/casc/haarcascade_frontalface_default.xml"
IMAGE_IN="./res/img/abba.png"
IMAGE_OUT="./out/predicted.png"

if __name__ == "__main__":

    print("---------EMOTION CLASSIFIER---------")

    # OPEN CV2 INIT
    font = cv2.FONT_HERSHEY_SIMPLEX # font to use in output text
    emotion_offsets = (0,0) # set for drawing bounding boxes
    face_detection = load_detection_model(HAAR_CASCADE)

    #Load the trained model + labels
    emotion_classifier = load_model(WEIGHTS_EMOTION, compile=False) # load weights
    emotion_target_size = emotion_classifier.input_shape[1:3]
    emotion_labels = get_labels('fer2013') #{0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

    video_capture = cv2.VideoCapture(0)

    while True:

        if not video_capture.isOpened():
            print("Unable to open camera.")
            break

        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load images and conver to greyscale
        # rgb_image = load_image(frame, color_mode='rgb')
        # gray_image = load_image(frame, color_mode='grayscale')
        gray_image = np.squeeze(gray)
        gray_image = gray_image.astype('uint8')

        #Get the faces from the image
        faces = detect_faces(face_detection, gray_image)

        for face_coords in faces:
            # Obtain the coordinates from each face to extract them from original image
            x1, x2, y1, y2 = apply_offsets(face_coords, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            # Attempt to resize the face to what the model expects
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                print("Unable to resize", gray_face)

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            # classify current face
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))

            # get label text corresponding to the number returned by the model
            emotion_text = emotion_labels[emotion_label_arg]

            color = (255,0,0)
            
            # draw box around face
            draw_bounding_box(face_coords, frame, color)
            # draw text around face
            draw_text(face_coords, frame, emotion_text, 0, -20, 1, 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()