from flask import Flask, Response
from flask_cors import CORS
import cv2
import numpy as np
import requests
import os

app = Flask(__name__)
CORS(app)

# Define paths to the assets
classFile = os.path.join(os.path.dirname(__file__), 'assets', 'coco.names')
configPath = os.path.join(os.path.dirname(__file__), 'assets', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(os.path.dirname(__file__), 'assets', 'frozen_inference_graph.pb')

# Load class labels
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Initialize the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# URL for fetching the image
image_url = 'http://192.168.60.185/cam-hi.jpg'

def process_image():
    try:
        # Fetch the image from the URL
        img_response = requests.get(image_url, timeout=1.4, verify=False)
        img_data = img_response.content  # Get raw bytes
        img_np = np.frombuffer(img_data, dtype=np.uint8)  # Convert to NumPy array
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # Decode image

        # Rotate the image (if necessary)
        img = cv2.rotate(img, cv2.ROTATE_180)

        # Perform object detection
        class_ids, confs, bbox = net.detect(img, confThreshold=0.6)

        # Draw bounding boxes and labels based on confidence
        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                if confidence >= 0.7:
                    color = (0, 255, 0)  # Green
                    label_color = (0, 255, 0)
                elif 0.5 <= confidence < 0.7:
                    color = (0, 255, 255)  # Yellow
                    label_color = (0, 255, 255)
                else:
                    continue
                cv2.rectangle(img, box, color=color, thickness=3)
                cv2.putText(img, classNames[class_id - 1], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)

        # Encode the processed image
        _, img_encoded = cv2.imencode('.jpg', img)
        return img_encoded.tobytes()

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def home():
    return 'Welcome home'

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            processed_frame = process_image()
            if processed_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
