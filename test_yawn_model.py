import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "models/yawn_cnn_best.keras"
)

def test_image(path):

    img = Image.open(path).convert("RGB")
    img = img.resize((96, 96))

    img = np.array(img, dtype=np.float32)

    # Model already contains Rescaling(1/255),
    # so DO NOT divide by 255 here.

    img = np.expand_dims(img, axis=0)

    prediction = model(
        img,
        training=False
    ).numpy()[0][0]

    # no_yawn = 0
    # yawn = 1
    if prediction >= 0.5:
        result = "YAWN"
    else:
        result = "NO YAWN"

    print("\nImage:", path)
    print("Score:", prediction)
    print("Prediction:", result)


# Replace these filenames with actual images
test_image(
    "dataset_yawn_new/yawn/18.jpg"
)

test_image(
    "dataset_yawn_new/no_yawn/2766.jpg"
)