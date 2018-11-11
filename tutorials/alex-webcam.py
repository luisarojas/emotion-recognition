import cv2
import sys


def simple_facedetect():
    #https://realpython.com/face-recognition-with-python/
    cascade_face = cv2.CascadeClassifier(casc_path)
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade_face.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

def webcam_facedetect():
    cascade_face = cv2.CascadeClassifier(casc_path)

    video_capture = cv2.VideoCapture(0)

    while True:
        #Capture frame by frame
        ret,frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade_face.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        #Draw the rectangle on the grame
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #Release resources on completion
    video_capture.release()
    # cv2.DestroyAllWindows()     
    
        

if __name__ == "__main__":
    print("---------FACE DETECTION WITH OPENCV---------")
    
    img_path = "./res/img/abba.png"
    casc_path = "./res/opencv/casc/haarcascade_frontalface_default.xml"

    # simple_facedetect()
    webcam_facedetect()
    
