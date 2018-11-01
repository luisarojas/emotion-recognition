import cv2
import sys

# get user-supplied image
# imagePath = sys.argv[1]
# cascPath = sys.argv[2]
face_cascade_path = "haarcascade_frontalface_default.xml"

# create haar cascade from xml file
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# set video source to default webcam
# note that you could also provide a filename here, but would need
# ffmpeg library installed
video_capture = cv2.VideoCapture(0)

# capture each frame in the video
while True:

    if not video_capture.isOpened():
        print("Unable to load camera.")
        break

    # ret: return code. tells us if we ran out of frames. not as relevant when using a webcam (_could_ record forever...)
    # frame: actual video frame
    ret, frame = video_capture.read()

    # convert current frame to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    # detectMultiScale: detects objects.
    #    detects faces by calling it on the face cascade
    # faces: list of rectangles in which it belieces it found a face
    #   x and y - location of the rectangle
    #   w and h - the rectangle's width and height
    faces = face_cascade.detectMultiScale(
        gray, # greyscale image
        scaleFactor=1.3, # compensate for faces being closer
                         # or further away from the camera
        minNeighbors=5, # how many objects are detected near the
                        # current one before it declares the face found
        minSize=(20, 20) # gives the size of each window
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE # not needed in cv v3
    )

    # draw a rectangle around all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # display resulting frame
    cv2.imshow('Video', frame)

    # if 'q' key is pressed, exit the script
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when all is done, release the capture and clean up
video_capture.release()
cv2.destroyAllWindows()