import numpy as np
import cv2

from keras.models import load_model
from utils.helper import apply_offsets,preprocess_input

EMOTION_WEIGHTS = "./res/saved_weights/emotion_model_weights.hdf5"
GENDER_WEIGHTS = "./res/saved_weights/gender_model_weights.hdf5"

HAAR_CASCADE="./res/opencv/casc/haarcascade_frontalface_default.xml"

EMOTION_LABELS = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
GENDER_LABELS = {0:'woman', 1:'man'}

if __name__ == "__main__":

    # OPEN CV2 INIT
    font = cv2.FONT_HERSHEY_SIMPLEX # font to use in output text
    emotion_offsets = (0,0) # set for drawing bounding boxes
    # gender_offsets = (10,10) # set for drawing bounding boxes
    face_detection = cv2.CascadeClassifier(HAAR_CASCADE)

    # load the trained model from keras, and use the given weights
    emotion_classifier = load_model(EMOTION_WEIGHTS, compile=False)
    gender_classifier = load_model(GENDER_WEIGHTS, compile=False)

    emotion_target_size = emotion_classifier.input_shape[1:3]
    gender_target_size = gender_classifier.input_shape[1:3]

    video_capture = cv2.VideoCapture(0)
    print("Opening camera...")

    while True:

        if not video_capture.isOpened():
            print("Unable to open camera.")
            break

        ret, frame = video_capture.read()

        # Load images and conver to greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = np.squeeze(gray)
        gray_image = gray_image.astype('uint8')

        # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb_image = np.squeeze(rgb_image)
        # rgb_image = rgb_image.astype('uint8')

        #Get the faces from the image
        faces = face_detection.detectMultiScale(
            gray, # greyscale image
            scaleFactor=1.3, # compensate for faces being closer
                             # or further away from the camera
            minNeighbors=5, # how many objects are detected near the
                            # current one before it declares the face found
            minSize=(30, 30) # gives the size of each face window
        )

        for face_coords in faces:
            # Obtain the coordinates from each face to extract them from original image
            x1, x2, y1, y2 = apply_offsets(face_coords, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            # x1, x2, y1, y2 = apply_offsets(face_coords, gender_offsets)
            # rgb_face = rgb_image[y1:y2, x1:x2]
            # Attempt to resize the face to what the model expects
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                print("Unable to resize gray image.")

            # try:
            #     rgb_face = cv2.resize(rgb_face, (gender_target_size))
            # except Exception as e:
            #     print("Unable to resize rgb image: " + e)

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            # rgb_face = preprocess_input(rgb_face, False)
            # rgb_face = np.expand_dims(rgb_face, 0)

            # classify current face
            emotion_prediciton = emotion_classifier.predict(gray_face)
            gender_prediciton = gender_classifier.predict(gray_face)

            # get percentages for each label
            emotion_code = np.argmax(emotion_prediciton)
            emotion_percentage = np.max(emotion_prediciton)

            gender_code = np.argmax(gender_prediciton)
            gender_percentage = np.max(gender_prediciton)

            # --- draw rectangles and text around faces found ---

            # get label text corresponding to the number returned by the model
            emotion_percentage_text = "{:.0f}%".format(emotion_percentage * 100)
            gender_percentage_text = "{:.0f}%".format(gender_percentage * 100)
            emotion_text = EMOTION_LABELS[emotion_code] + ": " + emotion_percentage_text
            gender_text =  GENDER_LABELS[gender_code] + ": " + gender_percentage_text
            text_color = (0, 255, 0)
            x, y = face_coords[:2]
            font_scale = 1
            font_thickness = 2

            # draw box around face
            box_color = (0, 255, 0)
            box_thickness = 2
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, box_thickness)
                cv2.putText(frame, emotion_text, (x, y - 10), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                cv2.putText(frame, gender_text, (x, y + h + 25), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    print("Closing camera...")
    cv2.destroyAllWindows()