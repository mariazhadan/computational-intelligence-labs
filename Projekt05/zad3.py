from ultralytics import YOLO
import cv2
import os

folder = "bird_miniatures"
files = os.listdir(folder)

model = YOLO("yolo26n.pt")

def count_birds(path):
    img = cv2.imread(path)

    img = cv2.resize(img, None, fx=8, fy=8)

    results = model.predict(
        img,
        conf=0.001,
        verbose=False
    )

    count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        if name in ["bird", "airplane", "kite", "sports ball", "frisbee"]:
            count += 1

    return count

results = []

for f in files:
    path = os.path.join(folder, f)
    results.append((f, count_birds(path)))

with open("list_zad3.txt", "w") as file:
    for name, cnt in results:
        file.write(f"{name}: {cnt}\n")

"""
YOLO nie działa dobrze, ponieważ obiekty na obrazach są bardzo małe i wyglądają jak czarne kropki. 
Nawet po powiększeniu obrazów model najczęściej nie wykrywa żadnych ptaków.
"""