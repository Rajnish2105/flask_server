import os
import cv2
import numpy as np
from flask import Flask, Response, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define paths to the assets
classFile = os.path.join(os.path.dirname(__file__), 'assets', 'coco.names')
configPath = os.path.join(os.path.dirname(__file__), 'assets', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(os.path.dirname(__file__), 'assets', 'frozen_inference_graph.pb')

# Load class labels
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Initialize the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Global variable to store the latest processed image
latest_frame = None

@app.route('/fetch_image', methods=['POST'])
def fetch_image():
    global latest_frame
    try:
        # Read raw binary data directly from the request body
        img_data = request.data
        if not img_data:
            return "No image data received", 400

        # Decode the binary data to an OpenCV image
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if img is None:
            return "Invalid image data", 400

        # Process the image and store the result in the global variable
        latest_frame = process_image(img)
        return "Image processed", 200
    except Exception as e:
        print(f"Error receiving image: {e}")
        return "Error", 500

def process_image(img):
    try:
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

@app.route('/video_feed')
def video_feed():
    """Live video feed route."""
    def generate():
        while True:
            # Check if the latest frame is available
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return 'Welcome to Flask on Vercel!'

if __name__ == '__main__':
    app.run(debug=True)
