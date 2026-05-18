from ultralytics import YOLO
import cv2
import json
import os
from collections import defaultdict

model = YOLO("yolo26n.pt")

image_path = "source/office_yolo.png"
video_path = "source/office_yolo.mp4"

conf_levels = [0.1, 0.3, 0.5, 0.7]

os.makedirs("results", exist_ok=True)


def run_image(conf):
    results = model.predict(source=image_path, conf=conf)

    detections = []

    for r in results:
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls[0])
            conf_score = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": conf_score,
                "bbox": [x1, y1, x2, y2]
            })

        img = r.plot()
        cv2.imwrite(f"results/image_conf_{conf}.jpg", img)

    with open(f"results/image_conf_{conf}.json", "w") as f:
        json.dump({"image": image_path, "detections": detections}, f, indent=2)


def run_video(conf):
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    data = {
        "video": video_path,
        "frames": []
    }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        f"results/video_conf_{conf}.mp4",
        fourcc,
        fps,
        (width, height)
    )

    class_stats = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf)

        frame_dets = []

        for r in results:
            names = r.names

            for b in r.boxes:
                cls_id = int(b.cls[0])
                cls_name = names[cls_id]
                conf_score = float(b.conf[0])
                x1, y1, x2, y2 = b.xyxy[0].tolist()

                frame_dets.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf_score,
                    "bbox": [x1, y1, x2, y2]
                })

                class_stats[cls_name] += 1

            annotated = r.plot()
            out.write(annotated)

        data["frames"].append({
            "frame_id": frame_id,
            "time": frame_id / fps,
            "detections": frame_dets
        })

        frame_id += 1

    cap.release()
    out.release()

    with open(f"results_zad1/video_conf_{conf}.json", "w") as f:
        json.dump(data, f, indent=2)

    with open(f"results_zad1/statistic_conf_{conf}.txt", "w") as f:
        for k, v in sorted(class_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{k}: {v}\n")


for c in conf_levels:
    run_image(c)
    run_video(c)