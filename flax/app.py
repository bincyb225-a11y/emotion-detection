from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("emotion_cnn_model.keras")

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# Store current image path
current_image = None


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not read image")

    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)

    return img


@app.route("/", methods=["GET", "POST"])
def index():
    global current_image

    prediction = None
    image_path = current_image

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "":

            # Delete all previous uploaded images
            for old_file in os.listdir(UPLOAD_FOLDER):
                old_path = os.path.join(UPLOAD_FOLDER, old_file)

                if os.path.isfile(old_path):
                    os.remove(old_path)

            # Save new image
            filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            # Process image
            processed_image = preprocess_image(save_path)

            # Predict
            result = model.predict(processed_image, verbose=0)

            emotion_index = np.argmax(result)
            prediction = emotion_labels[emotion_index]

            # Keep latest image for display
            current_image = "/" + save_path.replace("\\", "/")
            image_path = current_image

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

