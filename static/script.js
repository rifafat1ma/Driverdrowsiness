const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const startButton =
    document.getElementById("startButton");

const eyesText =
    document.getElementById("eyes");

const yawnText =
    document.getElementById("yawn");

const statusText =
    document.getElementById("status");

const warning =
    document.getElementById("warning");


let detectionRunning = false;

let closedStart = null;

const DROWSY_TIME = 5000;


// ----------------------------------
// START CAMERA
// ----------------------------------

async function startCamera() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user"
                },
                audio: false
            });

        video.srcObject = stream;

    }

    catch (error) {

        alert(
            "Camera permission failed: "
            + error
        );

    }
}


// ----------------------------------
// SEND FRAME
// ----------------------------------

async function sendFrame() {

    if (!detectionRunning)
        return;

    if (
        video.videoWidth === 0
        || video.videoHeight === 0
    ) {

        setTimeout(
            sendFrame,
            500
        );

        return;

    }

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    const context =
        canvas.getContext("2d");

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    const imageData =
        canvas.toDataURL(
            "image/jpeg",
            0.65
        );

    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        image: imageData
                    })
                }
            );

        const data =
            await response.json();

        eyesText.textContent =
            data.eye_state;

        yawnText.textContent =
            data.yawn_state;


        // ----------------------------------
        // EYE CLOSURE TIMER
        // ----------------------------------

        if (
            data.eye_state ===
            "CLOSED"
        ) {

            if (
                closedStart === null
            ) {

                closedStart =
                    Date.now();

            }

        }

        else {

            closedStart = null;

        }


        let status = "ALERT";


        // ----------------------------------
        // 5 SECOND DROWSINESS RULE
        // ----------------------------------

        if (
            closedStart !== null
            &&
            Date.now()
            - closedStart
            >= DROWSY_TIME
        ) {

            status = "DROWSY";

        }


        // ----------------------------------
        // YAWN
        // ----------------------------------

        else if (
            data.yawn_state ===
            "YAWN"
        ) {

            status =
                "FATIGUE WARNING";

        }


        statusText.textContent =
            status;


        // ----------------------------------
        // WARNING
        // ----------------------------------

        if (
            status === "DROWSY"
        ) {

            warning.style.display =
                "block";

            // Phone vibration if supported
            if (
                navigator.vibrate
            ) {

                navigator.vibrate([
                    500,
                    200,
                    500
                ]);

            }

        }

        else {

            warning.style.display =
                "none";

        }

    }

    catch (error) {

        console.log(error);

    }


    // About 2 predictions per second
    setTimeout(
        sendFrame,
        500
    );
}


// ----------------------------------
// START BUTTON
// ----------------------------------

startButton.addEventListener(
    "click",
    () => {

        if (!detectionRunning) {

            detectionRunning = true;

            startButton.textContent =
                "Detection Running";

            sendFrame();

        }

    }
);


startCamera();