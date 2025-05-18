import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def get_features(path):
    """thumbnail -> numbers the model can learn from"""
    img = cv2.imread(path)
    img = cv2.resize(img, (320, 180))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = gray.mean()
    contrast = gray.std()
    saturation = hsv[:, :, 1].mean()

    edges = cv2.Canny(gray, 100, 200)
    edge_pct = (edges > 0).mean()

    # colourful-ness formula i found on stackoverflow
    b, g, r = cv2.split(img.astype("float"))
    rg = r - g
    yb = 0.5 * (r + g) - b
    colour = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return [brightness, contrast, saturation, edge_pct, colour, len(faces)]
