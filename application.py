from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model('optimize3_model.h5')

# Define a function to detect emotions from an image
def detect_emotion(image_data):
    emotions = ["surprise", "fear", "angry", "neutral", "sad", "disgust", "happy"]
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, (1, 48, 48, 1))
    result = model.predict(img)
    print(result)
    emotion = emotions[np.argmax(result)]
    return emotion

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_endpoint():
    try:
        image_data = request.files['photo'].read()
        detected_emotion = detect_emotion(image_data)
        return jsonify({'emotion': detected_emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
