# Driver Drowsiness Detection

This project detects driver drowsiness using AI and machine learning.

It uses two trained CNN models:

- Eye Detection: Open / Closed
- Yawn Detection: Yawn / No Yawn

If the driver's eyes remain closed for 5 seconds, the system gives a drowsiness warning.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- HTML, CSS, JavaScript

## Main Models

- `eye_small_best.keras`
- `yawn_cnn_best.keras`

## Run the Project

Install required packages and run:

```bash
python app.py
