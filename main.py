import cv2
from ultralytics import YOLO

# Modell laden
model = YOLO("yolo11n.pt")
# Kamera öffnen
cap = cv2.VideoCapture(0)  # 0 = Standard Webcam

if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferenz
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        # COCO Klasse 0 = person
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        # Rechteck und Label zeichnen
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )
        label = f"Person {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("Personen Erkennung", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
