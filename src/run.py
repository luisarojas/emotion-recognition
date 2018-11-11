import numpy as np
import cv2

from keras.models import load_model
from utils.helper import apply_offsets,preprocess_input

WEIGHTS_EMOTION="./res/saved_weights/emotion_model.hdf5"
HAAR_CASCADE="./res/opencv/casc/haarcascade_frontalface_default.xml"
IMAGE_IN="./res/img/abba.png"
IMAGE_OUT="./out/predicted.png"
EMOTION_LABELS = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

if __name__ == "__main__":

    print("---------EMOTION CLASSIFIER---------")

    # OPEN CV2 INIT
    font = cv2.FONT_HERSHEY_SIMPLEX # font to use in output text
    emotion_offsets = (0,0) # set for drawing bounding boxes
    face_detection = cv2.CascadeClassifier(HAAR_CASCADE)

    # load the trained model from keras, and use the given weights
    emotion_classifier = load_model(WEIGHTS_EMOTION, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]

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
        faces = face_detection.detectMultiScale(
            gray, # greyscale image
            scaleFactor=1.3, # compensate for faces being closer
                             # or further away from the camera
            minNeighbors=5, # how many objects are detected near the
                            # current one before it declares the face found
            minSize=(20, 20) # gives the size of each window
            # flags=cv2.cv.CV_HAAR_SCALE_IMAGE # not needed in cv v3
        )

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
            emotion_prediciton = emotion_classifier.predict(gray_face)

            # get percentages for each label
            emotion_label_arg = np.argmax(emotion_prediciton)
            emotion_percentage = np.max(emotion_prediciton)

            # draw box around face
            box_color = (255, 0, 0)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

            # draw text around face
            # get label text corresponding to the number returned by the model
            emotion_percentage_text = "{:.0f}%".format(emotion_percentage * 100)
            emotion_text = EMOTION_LABELS[emotion_label_arg] + " " + emotion_percentage_text
            text_color = (0, 255, 0)
            x, y = face_coords[:2]
            font_scale = 1
            thickness = 2
            # draw_text(face_coords, frame, emotion_text, 0, -20, 1, 2)
            cv2.putText(frame, emotion_text, (x + emotion_offsets[0], y + emotion_offsets[1]), font, font_scale, text_color, thickness, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()