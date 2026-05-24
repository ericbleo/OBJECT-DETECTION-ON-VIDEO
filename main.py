# TIKTOK: https://www.tiktok.com/@ericbleo

import cv2
import cvzone
from ultralytics import YOLO
import math
import os

class SecuritySystem:
    # Initialize the object detector
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")
        self.video = cv2.VideoCapture("videos/boat.mp4")
        self.output_path = "output.mp4"
        self.class_names = []
        with open("names/coco.names", "r") as file:
            self.class_names = file.read().splitlines()

    def resize(self, frame, scale):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    # Run the object detector
    def run(self):
        fps = self.video.get(cv2.CAP_PROP_FPS) or 30
        writer = None

        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            # Resize the frame
            frame = self.resize(frame, 1)
            results = self.model(frame, stream=True)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    name = self.class_names[int(box.cls[0])]
                    
                    cvzone.cornerRect(frame, (x1, y1, w, h), 7, 3)
                    cvzone.putTextRect(frame, f'{name}', (x1, y1 - 10), 1, 2, (0, 0, 0))

            if writer is None:
                h, w, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

            writer.write(frame)
            cv2.imshow("TIKTOK: @ericbleo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    security_system = SecuritySystem()
    security_system.run()