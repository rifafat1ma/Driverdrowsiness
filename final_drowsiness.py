import cv2
import time
import winsound
import threading
import numpy as np
import tensorflow as tf
from collections import deque


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

eye_model = tf.keras.models.load_model(
    "models/eye_small_best.keras"
)

yawn_model = tf.keras.models.load_model(
    "models/yawn_cnn_best.keras"
)


# =====================================================
# FACE DETECTOR
# =====================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Reduces camera buffering / lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# =====================================================
# SETTINGS
# =====================================================

FACE_DETECT_EVERY = 4

# Run eye model every 2 frames
EYE_PREDICT_EVERY = 2

# Run yawn model less frequently
YAWN_PREDICT_EVERY = 4

# --------------------------------
# DROWSINESS THRESHOLD
# --------------------------------

# Eyes must stay closed for 5 seconds
DROWSY_TIME = 5.0

# CNN classification thresholds
EYE_THRESHOLD = 0.50
YAWN_THRESHOLD = 0.50

# Store previous predictions
eye_history = deque(maxlen=10)
yawn_history = deque(maxlen=8)

frame_count = 0

last_face = None

eye_state = "UNKNOWN"
yawn_state = "NO YAWN"

status = "ALERT"

closed_start = None

last_eye_score = 0.0
last_yawn_score = 0.0

alarm_active = False


# =====================================================
# ALARM FUNCTION
# =====================================================

def play_alarm():

    global alarm_active

    if alarm_active:
        return

    alarm_active = True

    def alarm():

        # 4 warning beeps
        for _ in range(4):

            winsound.Beep(
                1500,
                450
            )

            time.sleep(0.15)

        global alarm_active
        alarm_active = False

    threading.Thread(
        target=alarm,
        daemon=True
    ).start()


# =====================================================
# PREPARE EYE IMAGE
#
# Eye CNN was trained using:
# 64 x 64 grayscale images
# =====================================================

def prepare_eye(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (64, 64)
    )

    gray = gray.astype(
        np.float32
    )

    # Add channel dimension:
    # (64,64) -> (64,64,1)
    gray = np.expand_dims(
        gray,
        axis=-1
    )

    return gray


# =====================================================
# PREPARE YAWN IMAGE
#
# Yawn CNN was trained using:
# 96 x 96 RGB images
# =====================================================

def prepare_yawn(image):

    image = cv2.resize(
        image,
        (96, 96)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    )

    # DO NOT divide by 255 here.
    # The trained yawn CNN already contains
    # Rescaling(1/255).

    return image


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam.")
        break

    frame_count += 1

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # =================================================
    # FACE DETECTION
    # =================================================

    if frame_count % FACE_DETECT_EVERY == 0:

        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        if len(faces) > 0:

            # Use largest detected face
            last_face = max(
                faces,
                key=lambda f:
                f[2] * f[3]
            )

        else:

            last_face = None


    # =================================================
    # PROCESS DETECTED FACE
    # =================================================

    if last_face is not None:

        x, y, w, h = last_face

        # Keep coordinates safe
        x = max(0, x)
        y = max(0, y)

        face_crop = frame[
            y:y+h,
            x:x+w
        ]

        if face_crop.size == 0:
            continue


        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )


        # =================================================
        # TIGHTER EYE REGIONS
        # =================================================

        # Vertical eye position
        eye_top = int(
            y + h * 0.31
        )

        eye_bottom = int(
            y + h * 0.41
        )


        # Left eye
        left_x1 = int(
            x + w * 0.12
        )

        left_x2 = int(
            x + w * 0.46
        )


        # Right eye
        right_x1 = int(
            x + w * 0.54
        )

        right_x2 = int(
            x + w * 0.88
        )


        left_eye = frame[
            eye_top:eye_bottom,
            left_x1:left_x2
        ]

        right_eye = frame[
            eye_top:eye_bottom,
            right_x1:right_x2
        ]


        # Draw green eye boxes
        cv2.rectangle(
            frame,
            (left_x1, eye_top),
            (left_x2, eye_bottom),
            (0, 255, 0),
            1
        )

        cv2.rectangle(
            frame,
            (right_x1, eye_top),
            (right_x2, eye_bottom),
            (0, 255, 0),
            1
        )


        # =================================================
        # EYE CNN PREDICTION
        # =================================================

        if (
            frame_count %
            EYE_PREDICT_EVERY == 0
            and left_eye.size > 0
            and right_eye.size > 0
        ):

            left_input = prepare_eye(
                left_eye
            )

            right_input = prepare_eye(
                right_eye
            )


            eye_batch = np.array([
                left_input,
                right_input
            ])


            predictions = eye_model(
                eye_batch,
                training=False
            ).numpy().flatten()


            left_score = float(
                predictions[0]
            )

            right_score = float(
                predictions[1]
            )


            average_eye_score = (
                left_score +
                right_score
            ) / 2


            last_eye_score = (
                average_eye_score
            )


            # -----------------------------
            # MODEL LABELS
            #
            # CLOSED = 0
            # OPEN   = 1
            # -----------------------------

            if (
                average_eye_score
                >= EYE_THRESHOLD
            ):

                eye_history.append(1)

            else:

                eye_history.append(0)


            # -----------------------------
            # SMOOTH PREDICTION
            # -----------------------------

            if len(eye_history) >= 6:

                open_ratio = (
                    sum(eye_history) /
                    len(eye_history)
                )

                # More than half recent
                # predictions must indicate OPEN
                if open_ratio >= 0.50:

                    eye_state = "OPEN"

                else:

                    eye_state = "CLOSED"


        # =================================================
        # EYE-CLOSURE TIMER
        # =================================================

        if eye_state == "CLOSED":

            if closed_start is None:

                closed_start = (
                    time.time()
                )

        else:

            # Reset immediately when
            # eyes are reliably open
            closed_start = None


        # =================================================
        # YAWN MODEL
        #
        # IMPORTANT:
        # Use FULL FACE, not the old mouth crop.
        # =================================================

        if (
            frame_count %
            YAWN_PREDICT_EVERY == 0
        ):

            yawn_input = prepare_yawn(
                face_crop
            )

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


            last_yawn_score = (
                yawn_score
            )


            # -----------------------------
            # MODEL LABELS
            #
            # NO_YAWN = 0
            # YAWN    = 1
            # -----------------------------

            if (
                yawn_score
                >= YAWN_THRESHOLD
            ):

                yawn_history.append(1)

            else:

                yawn_history.append(0)


            # -----------------------------
            # SMOOTH YAWN PREDICTIONS
            # -----------------------------

            if len(yawn_history) >= 5:

                yawn_ratio = (
                    sum(yawn_history) /
                    len(yawn_history)
                )

                # Require majority of recent
                # predictions to be yawning
                if yawn_ratio >= 0.70:

                    yawn_state = "YAWN"

                else:

                    yawn_state = (
                        "NO YAWN"
                    )


        # =================================================
        # FINAL DRIVER STATUS
        # =================================================

        status = "ALERT"


        # ---------------------------------
        # EYES CLOSED FOR 5 SECONDS
        # ---------------------------------

        if closed_start is not None:

            closed_duration = (
                time.time()
                - closed_start
            )

            if (
                closed_duration
                >= DROWSY_TIME
            ):

                status = "DROWSY"


        # ---------------------------------
        # YAWNING
        # ---------------------------------

        if (
            status == "ALERT"
            and yawn_state == "YAWN"
        ):

            status = (
                "FATIGUE WARNING"
            )


        # =================================================
        # ALARM
        #
        # Only DROWSY triggers alarm.
        # A yawn alone does NOT trigger alarm.
        # =================================================

        if status == "DROWSY":

            play_alarm()


    # =================================================
    # NO FACE
    # =================================================

    else:

        eye_state = "UNKNOWN"
        yawn_state = "UNKNOWN"

        status = "NO FACE"

        closed_start = None

        eye_history.clear()
        yawn_history.clear()


    # =================================================
    # DISPLAY COLOR
    # =================================================

    if status == "ALERT":

        color = (
            0,
            255,
            0
        )

    elif status == "FATIGUE WARNING":

        color = (
            0,
            255,
            255
        )

    elif status == "DROWSY":

        color = (
            0,
            0,
            255
        )

    else:

        color = (
            255,
            255,
            255
        )


    # =================================================
    # DISPLAY INFORMATION
    # =================================================

    cv2.putText(
        frame,
        f"Eyes: {eye_state}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


    cv2.putText(
        frame,
        f"Yawn: {yawn_state}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


    # Scores are useful for testing.
    # You can remove these later.

    cv2.putText(
        frame,
        f"Eye score: {last_eye_score:.2f}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"Yawn score: {last_yawn_score:.2f}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    # =================================================
    # WAKE-UP WARNING
    # =================================================

    if status == "DROWSY":

        cv2.putText(
            frame,
            "WAKE UP!",
            (180, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3
        )


    # =================================================
    # SHOW CAMERA
    # =================================================

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    # Press Q to quit
    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# =====================================================
# CLEANUP
# =====================================================

cap.release()

cv2.destroyAllWindows()