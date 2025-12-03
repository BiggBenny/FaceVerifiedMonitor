import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")  # genauer als n

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Kamera geht nicht auf")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kein Frame")
            break

        results = model(frame, imgsz=640)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue  # nur person

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

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

        cv2.imshow("Person detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
