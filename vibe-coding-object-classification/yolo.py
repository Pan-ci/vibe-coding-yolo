from ultralytics import YOLO
import cv2
# import time

model = YOLO("yolo-classify-cifar-100-best[1].pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("", annotated)

    if cv2.waitKey(500) & 0xFF == 27: # 1-30 ms cukup
        break

    # time.sleep(0.5)  # jeda 0.5 detik antar inferensi

cap.release()
cv2.destroyAllWindows()
