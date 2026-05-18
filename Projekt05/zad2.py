import cv2
import os
import numpy as np

folder = "bird_miniatures"
files = os.listdir(folder)

def count_birds(image_path):
    img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        13,
        2
    )

    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bird_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1 < area < 100:
            bird_count += 1

    return bird_count

results = []

for f in files:
    path = os.path.join(folder, f)
    results.append((f, count_birds(path)))

with open("list_zad2.txt", "w") as file:
    for name, cnt in results:
        file.write(f"{name}: {cnt}\n")