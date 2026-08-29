# TIKTOK: https://www.tiktok.com/@ericbleo

import cv2
from ultralytics import YOLO


class SecuritySystem:
    def __init__(self):
        self.model = YOLO("models/yolo26l.pt")
        self.video_path = "videos/9.mp4"
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")
        self.output_path = "output/video.mp4"

        with open("names/coco.names", "r") as file:
            self.class_names = file.read().splitlines()

        self.default_color = (255, 0, 255)
        # BGR colors — one unique color per COCO class
        self.label_colors = {
            "person": (0, 255, 0),
            "bicycle": (255, 128, 0),
            "car": (0, 128, 255),
            "motorbike": (255, 0, 128),
            "aeroplane": (255, 255, 0),
            "bus": (0, 200, 255),
            "train": (128, 0, 255),
            "truck": (0, 255, 255),
            "boat": (255, 0, 0),
            "traffic light": (0, 255, 128),
            "fire hydrant": (0, 0, 255),
            "stop sign": (0, 64, 255),
            "parking meter": (128, 255, 0),
            "bench": (180, 105, 255),
            "bird": (203, 192, 255),
            "cat": (0, 140, 255),
            "dog": (19, 69, 139),
            "horse": (45, 82, 160),
            "sheep": (220, 220, 220),
            "cow": (50, 205, 50),
            "elephant": (128, 128, 128),
            "bear": (42, 42, 165),
            "zebra": (200, 200, 200),
            "giraffe": (0, 215, 255),
            "backpack": (147, 20, 255),
            "umbrella": (255, 0, 255),
            "handbag": (255, 192, 203),
            "tie": (128, 0, 128),
            "suitcase": (70, 130, 180),
            "frisbee": (0, 255, 127),
            "skis": (240, 248, 255),
            "snowboard": (255, 248, 240),
            "sports ball": (0, 215, 0),
            "kite": (255, 20, 147),
            "baseball bat": (139, 69, 19),
            "baseball glove": (0, 180, 0),
            "skateboard": (255, 69, 0),
            "surfboard": (0, 191, 255),
            "tennis racket": (50, 205, 154),
            "bottle": (64, 224, 208),
            "wine glass": (221, 160, 221),
            "cup": (255, 128, 64),
            "fork": (192, 192, 192),
            "knife": (144, 128, 112),
            "spoon": (222, 196, 176),
            "bowl": (140, 180, 210),
            "banana": (0, 255, 200),
            "apple": (0, 0, 220),
            "sandwich": (185, 218, 255),
            "orange": (0, 165, 255),
            "broccoli": (0, 100, 0),
            "carrot": (0, 100, 255),
            "hot dog": (71, 99, 255),
            "pizza": (0, 69, 255),
            "donut": (147, 112, 219),
            "cake": (193, 182, 255),
            "chair": (43, 90, 139),
            "sofa": (0, 0, 128),
            "pottedplant": (34, 139, 34),
            "bed": (139, 61, 72),
            "diningtable": (90, 60, 30),
            "toilet": (245, 245, 245),
            "tvmonitor": (112, 25, 25),
            "laptop": (153, 136, 119),
            "mouse": (169, 169, 169),
            "remote": (79, 79, 47),
            "keyboard": (100, 100, 100),
            "cell phone": (255, 105, 180),
            "microwave": (80, 80, 160),
            "oven": (60, 60, 60),
            "toaster": (30, 105, 210),
            "sink": (250, 230, 230),
            "refrigerator": (200, 180, 160),
            "book": (226, 43, 138),
            "clock": (0, 215, 180),
            "vase": (214, 112, 218),
            "scissors": (255, 191, 0),
            "teddy bear": (63, 133, 205),
            "hair drier": (122, 160, 255),
            "toothbrush": (152, 251, 152),
        }

    def resize(self, frame, scale):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def get_color(self, label):
        return self.label_colors.get(label, self.default_color)

    def draw_box(self, frame, x1, y1, x2, y2, label, color=(255, 0, 255), thickness=2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        padding = 5

        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        label_y1 = max(y1 - text_h - baseline - (padding * 2), 0)
        label_y2 = label_y1 + text_h + baseline + (padding * 2)
        label_x2 = x1 + text_w + (padding * 2)

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + padding, label_y2 - baseline - padding // 2),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    def run(self):
        fps = self.video.get(cv2.CAP_PROP_FPS) or 30
        infer_scale = 0.5
        inv_scale = 1.0 / infer_scale
        writer = None

        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            # Detect on a smaller frame for speed; draw/save on full resolution
            infer_frame = self.resize(frame, infer_scale)
            results = self.model(infer_frame, stream=True, conf=0.5)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v * inv_scale) for v in box.xyxy[0]]
                    name = self.class_names[int(box.cls[0])]
                    color = self.get_color(name)
                    self.draw_box(frame, x1, y1, x2, y2, name, color=color)

            if writer is None:
                h, w, _ = frame.shape
                # Prefer H.264 when available; fall back to mp4v
                writer = cv2.VideoWriter(
                    self.output_path,
                    cv2.VideoWriter_fourcc(*"avc1"),
                    fps,
                    (w, h),
                )
                if not writer.isOpened():
                    writer = cv2.VideoWriter(
                        self.output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )

            writer.write(frame)
            cv2.imshow("TIKTOK: @ericbleo", self.resize(frame, infer_scale))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    security_system = SecuritySystem()
    security_system.run()
