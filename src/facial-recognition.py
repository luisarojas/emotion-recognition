import cv2
import sys

# get user-supplied image
imagePath = sys.argv[1]
# cascPath = sys.argv[2]
cascPath = "haarcascade_frontalface_default.xml"

#create haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# create cascade and initialize it
image = cv2.imread(imagePath)

# convert image to greyscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the image
# detectMultiScale: detects objects.
#    detects faces by calling it on the face cascade
# faces: list of rectangles in which it belieces it found a face
#   x and y - location of the rectangle
#   w and h - the rectangle's width and height
faces = faceCascade.detectMultiScale(
    gray, # greyscale image
    scaleFactor=1.2, # compensate for faces being closer
                     # or further away from the camera
    minNeighbors=5, # how many objects are detected near the
                    # current one before it declares the face found
    minSize=(30, 30) # gives the size of each window
    # flags=cv2.cv.CV_HAAR_SCALE_IMAGE # not needed in cv v3
)

print("Found " + str(len(faces)) + " faces!")

# draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)