# Identifying Facial Expressions using Machine Learning

# Instructions
1. Ensure that you have the right docker image `make docker-build` (Only need to do this once)
2. `make`

### OpenCV tutorials used:
* [Face Recognition with Python, in Under 25 Lines of Code]( https://realpython.com/face-recognition-with-python/)
* [Face Detection in Python Using a Webcam](https://realpython.com/face-detection-in-python-using-a-webcam/)
* [Face Detection using Haar Cascades
](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)
* [Emotion GitHub repository](https://github.com/petercunha/Emotion) by [petercunha](https://github.com/petercunha)
* [Face classification GitHub repository](https://github.com/oarriaga/face_classification) by [oarriaga](https://github.com/oarriaga)

### Dataset

[Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

| Emotion  | Code |
|----------|------|
| Angry    | 0    |
| Disgust  | 1    |
| Fear     | 2    |
| Happy    | 3    |
| Sad      | 4    |
| Surprise | 5    |
| Neutral  | 6    |

This dataset consists 48x48 pixel grayscale images of faces.

`train.csv` contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image.

Total samples: 32,298
* Training: 28,709 samples.
* Testing: 3,589

---
<font size="1">
Other data resources:<br>
- Emotions: [This](https://github.com/muxspace/facial_expressions) public GitHub repository.
<br>
- Gender:
[Here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), "Download faces only (7 GB)" under IMDB.
</font>