import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "models/eye_small_best.keras"
)

def test_image(path):

    img = Image.open(path).convert("L")
    img = img.resize((64, 64))

    img = np.array(img, dtype=np.float32)

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    prediction = model(
        img,
        training=False
    ).numpy()[0][0]

    if prediction >= 0.5:
        result = "OPEN"
    else:
        result = "CLOSED"

    print("\nImage:", path)
    print("Score:", prediction)
    print("Prediction:", result)


test_image(
    "dataset_eye/Closed_Eyes/s0001_00016_0_0_0_0_0_01.png"
)

test_image(
    "dataset_eye/Open_Eyes/open_left_2.jpg"
)