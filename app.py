from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)

# -----------------------------
# LOAD MODELS
# -----------------------------
eye_model = tf.keras.models.load_model(
    "models/eye_small_best.keras"
)

yawn_model = tf.keras.models.load_model(
    "models/yawn_cnn_best.keras"
)


# -----------------------------
# PREPARE EYE
# -----------------------------
def prepare_eye(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (64, 64)
    )

    gray = gray.astype(np.float32)

    gray = np.expand_dims(
        gray,
        axis=-1
    )

    return gray


# -----------------------------
# PREPARE YAWN
# -----------------------------
def prepare_yawn(image):

    image = cv2.resize(
        image,
        (96, 96)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32)

    return image


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICTION
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    # Remove data:image/jpeg;base64,
    encoded = data.split(",")[1]

    image_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "error": "Invalid frame"
        })

    h, w, _ = frame.shape

    # --------------------------------
    # Approximate eye crops
    # --------------------------------

    eye_top = int(h * 0.28)
    eye_bottom = int(h * 0.43)

    left_eye = frame[
        eye_top:eye_bottom,
        int(w * 0.15):int(w * 0.47)
    ]

    right_eye = frame[
        eye_top:eye_bottom,
        int(w * 0.53):int(w * 0.85)
    ]

    eye_state = "UNKNOWN"
    eye_score = 0.0

    if (
        left_eye.size > 0
        and right_eye.size > 0
    ):

        left_input = prepare_eye(
            left_eye
        )

        right_input = prepare_eye(
            right_eye
        )

        batch = np.array([
            left_input,
            right_input
        ])

        predictions = eye_model(
            batch,
            training=False
        ).numpy().flatten()

        eye_score = float(
            (
                predictions[0]
                + predictions[1]
            ) / 2
        )

        if eye_score >= 0.5:
            eye_state = "OPEN"
        else:
            eye_state = "CLOSED"

    # --------------------------------
    # Yawn prediction
    # Use full camera frame initially
    # --------------------------------

    yawn_input = prepare_yawn(frame)

    yawn_input = np.expand_dims(
        yawn_input,
        axis=0
    )

    yawn_score = float(
        yawn_model(
            yawn_input,
            training=False
        ).numpy()[0][0]
    )

    if yawn_score >= 0.5:
        yawn_state = "YAWN"
    else:
        yawn_state = "NO YAWN"

    return jsonify({
        "eye_state": eye_state,
        "eye_score": eye_score,
        "yawn_state": yawn_state,
        "yawn_score": yawn_score
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )