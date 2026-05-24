# Object Detection on Video

This project runs **object detection** on a video file. It finds objects (people, cars, boats, animals, and more) in each frame, draws boxes around them, labels them, and saves the result as a new video.

You do **not** need to train a model. The project uses a pre-trained **YOLOv8** model that already knows 80 common object types (from the COCO dataset).

---

## What you will see when it works

1. A window opens and plays your input video with boxes and labels.
2. A file named `output.mp4` is created in this folder with the same annotated video.
3. Press `q` on your keyboard to stop early (or wait until the video ends).

---

## 1) What you need

- **Windows, Mac, or Linux**
- **Python 3.10 or 3.11** (recommended for beginners)
- An internet connection **only for the first install** (to download Python packages)
- A video file in the `videos/` folder (sample videos are already included)

You do **not** need a GPU. The project can run on a normal laptop (it may be slower on long videos).

---

## 2) Project structure (important)

Your folder should look like this:

```text
OBJECT DETECTION ON VIDEO/
  main.py                 # Main script — run this
  output.mp4              # Created after you run the app (your result)
  models/
    yolov8n.pt            # Small, fast YOLO model (used by default)
    yolov8l.pt            # Larger model (optional, not used unless you change code)
  names/
    coco.names            # List of 80 object names the model can detect
  videos/
    boat.mp4              # Sample videos you can try
    fishes.mp4
    train.mp4
```

**Do not rename** the `models`, `names`, or `videos` folders unless you also update the paths inside `main.py`.

---

## 3) Install Python (if you do not have it yet)

1. Download Python from [python.org/downloads](https://www.python.org/downloads/).
2. During install on Windows, check **“Add python.exe to PATH”**.
3. Open a new terminal and check:

```powershell
python --version
```

You should see something like `Python 3.11.x`.

---

## 4) Install required packages

Open a terminal **inside this project folder**, then run:

```powershell
python -m pip install --upgrade pip
pip install opencv-python cvzone ultralytics
```

What these packages do (simple explanation):

| Package        | Role                                      |
|----------------|-------------------------------------------|
| `opencv-python`| Read video, show window, save `output.mp4` |
| `cvzone`       | Draw nice boxes and text on the video     |
| `ultralytics`  | Run the YOLO object detection model       |

The first run may take a few minutes because pip downloads libraries.

---

## 5) Choose which video to use

Open `main.py` in any text editor and find this line inside `__init__`:

```python
self.video = cv2.VideoCapture("videos/boat.mp4")
```

Change `boat.mp4` to another file in `videos/`, for example:

```python
self.video = cv2.VideoCapture("videos/fishes.mp4")
```

or:

```python
self.video = cv2.VideoCapture("videos/train.mp4")
```

You can also add your own `.mp4` file to the `videos/` folder and use its name the same way.

---

## 6) Run the app

Make sure your terminal is still in this project folder, then run:

```powershell
python main.py
```

**While it runs:**

- Keep the video window focused if you want to press `q` to quit.
- Wait for the video to finish, or press `q` to stop.

**When it finishes:**

- Open `output.mp4` in any video player (VLC, Movies & TV, etc.).
- That file is your saved result with all detections drawn on it.

---

## 7) What objects can it detect?

The model reads labels from `names/coco.names`. Examples include:

- `person`, `car`, `bus`, `train`, `boat`
- `bird`, `cat`, `dog`, `horse`
- `bottle`, `cup`, `chair`, `laptop`, `cell phone`

There are **80 classes** in total. The model only labels objects it is confident about. If something is not in that list, it will not appear with a correct name.

---

## 8) Customize (optional)

### Use a different model size

In `main.py`, the default model is the small and fast one:

```python
self.model = YOLO("models/yolov8n.pt")
```

You can switch to the larger model (more accurate, slower):

```python
self.model = YOLO("models/yolov8l.pt")
```

### Change output file name

Find this line in `__init__`:

```python
self.output_path = "output.mp4"
```

Change `"output.mp4"` to any name you like, for example `"my_result.mp4"`.

### Resize the video (make it smaller = faster)

In `run()`, this line controls size:

```python
frame = self.resize(frame, 1)
```

- `1` = original size  
- `0.5` = half width and height (faster, less detail)

Example for half size:

```python
frame = self.resize(frame, 0.5)
```

---

## 9) Beginner troubleshooting

### `python` is not recognized

- Reinstall Python and enable **Add to PATH**, or use `py main.py` on Windows instead of `python main.py`.

### Error: cannot open `videos/boat.mp4`

- Check that the file exists inside the `videos` folder.
- Check spelling and `.mp4` extension in `main.py`.
- Run the terminal from the **project root** (the folder that contains `main.py`).

### Error: cannot find `models/yolov8n.pt`

- Confirm `yolov8n.pt` is inside the `models` folder.
- Do not move the model file without updating the path in `main.py`.

### Error: cannot find `names/coco.names`

- Confirm the `names` folder exists and contains `coco.names`.

### Window opens but nothing is detected

- Some videos have small or blurry objects — try another sample video.
- Try the larger model: `yolov8l.pt` (slower but more accurate).
- Make sure objects in the video are among the 80 COCO classes.

### `output.mp4` is empty or will not play

- Let the script run for at least a few seconds before pressing `q`.
- If you stopped immediately, delete `output.mp4` and run again.
- Try playing the file with [VLC](https://www.vlc-media-player.org/).

### Script is very slow

- Use `yolov8n.pt` (not `yolov8l.pt`).
- Resize frames to `0.5` as shown in section 8.
- Use a shorter video while learning.

### pip install fails

- Upgrade pip first: `python -m pip install --upgrade pip`
- Run the install command again.
- Use Python 3.10 or 3.11 if you are on an older or very new Python version.

---

## 10) Quick summary

1. Install Python and packages (`opencv-python`, `cvzone`, `ultralytics`).
2. Put your video in `videos/` (or use a sample).
3. Set the video path in `main.py`.
4. Run `python main.py`.
5. Press `q` to stop, or wait until the end.
6. Watch `output.mp4`.

---

## Credits

- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Class names: COCO dataset labels in `names/coco.names`
