import cv2
from pathlib import Path
from urllib.request import urlretrieve

from ultralytics import YOLO


def ensure_yunet_model():
    model_path = Path("face_detection_yunet_2023mar.onnx")
    if model_path.exists():
        return str(model_path)

    url = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    print("Lade YuNet Gesichtsmodell herunter...")
    urlretrieve(url, model_path)
    return str(model_path)


def main():
    model = YOLO("yolo26s.pt")  # genauer als nano

    face_detector = None
    try:
        yunet_path = ensure_yunet_model()
        face_detector = cv2.FaceDetectorYN.create(
            yunet_path,
            "",
            (320, 320),
            0.7,
            0.3,
            5000,
        )
        print("YuNet Gesichtserkennung aktiv.")
    except Exception as e:
        print(f"YuNet nicht verfuegbar ({e}), nutze Haar-Cascade Fallback.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Kamera geht nicht auf")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kein Frame")
            break

        results = model(frame, imgsz=960, conf=0.12, iou=0.5, verbose=False)[0]
        names = results.names

        for box in results.boxes:
            cls = int(box.cls[0])
            class_name = names.get(cls, str(cls))
            if class_name != "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if conf < 0.20:
                continue

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            label = f"Person {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if face_detector is not None:
            h, w = frame.shape[:2]
            face_detector.setInputSize((w, h))
            _, faces = face_detector.detect(frame)
            if faces is not None:
                for face in faces:
                    x, y, fw, fh = face[:4].astype(int)
                    score = float(face[-1])
                    if score < 0.65:
                        continue
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Face {score:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=6,
                minSize=(35, 35),
            )
            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Face",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        cv2.imshow("Person detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
