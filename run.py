import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load YOLO model
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
labels_path = "coco.names"

LABELS = open(labels_path).read().strip().split('\n')
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# GUI App
class YOLOApp:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLOv3 Object Detection")
        self.img_path = None

        self.btn_browse = tk.Button(master, text="Browse Image", command=self.load_image)
        self.btn_browse.pack()

        self.btn_detect = tk.Button(master, text="Run Detection", command=self.run_detection)
        self.btn_detect.pack()

        self.panel = tk.Label(master)
        self.panel.pack()

    def load_image(self):
        self.img_path = filedialog.askopenfilename()
        self.show_image(self.img_path)

    def show_image(self, path):
        img = Image.open(path)
        img = img.resize((600, 400))
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def run_detection(self):
        if not self.img_path:
            return

        image = cv2.imread(self.img_path)
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                text = "{}: {:.2f}".format(LABELS[class_ids[i]], confidences[i])
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # Show image in GUI
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 400))
        image = ImageTk.PhotoImage(image)
        self.panel.configure(image=image)
        self.panel.image = image

# Run the GUI
root = tk.Tk()
app = YOLOApp(root)
root.mainloop()
