import cv2

def main():
    # 0 ist meistens die Standardkamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Frame erhalten")
            break

        cv2.imshow("Kamera", frame)

        # Mit q beenden
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
