import cv2
import os

folder = "bird_miniatures"
files = os.listdir(folder)

def count_birds(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8)

    count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 2:
            count += 1

    return count


results = []

for f in files:
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(folder, f)
        results.append((f, count_birds(path)))

with open("bird_count_zad2.txt", "w") as file:
    for name, cnt in results:
        file.write(f"{name}: {cnt}\n")