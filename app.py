from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import os

app = Flask(__name__)

def get_face_encoding(image_path):
    # Load the image
    image = face_recognition.load_image_file(image_path)
    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Find face encodings in the image
    face_encodings = face_recognition.face_encodings(image_rgb)
    # Return the first face encoding found, if any
    if face_encodings:
        return face_encodings[0]
    return None

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please upload both images"}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    image1_path = os.path.join('uploads', image1.filename)
    image2_path = os.path.join('uploads', image2.filename)

    image1.save(image1_path)
    image2.save(image2_path)

    face_encoding1 = get_face_encoding(image1_path)
    face_encoding2 = get_face_encoding(image2_path)

    if face_encoding1 is None or face_encoding2 is None:
        return jsonify({"error": "Could not find a face in one or both images"}), 400

    # Compare the faces
    results = face_recognition.compare_faces([face_encoding1], face_encoding2)

    os.remove(image1_path)
    os.remove(image2_path)

    return jsonify({"match": results[0]})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
