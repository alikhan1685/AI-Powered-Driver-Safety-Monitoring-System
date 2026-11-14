# Driver Monitoring System

AI-Powered Truck Driver Safety Monitoring System: real-time drowsiness detection in the browser.

## Overview

This project detects driver drowsiness in real time using computer vision and deep learning. A webcam stream is analysed by MediaPipe face landmarks and two convolutional neural networks (CNNs), and the results are shown on a live dashboard with audio alerts. All processing runs in the browser, so video never leaves the user's device.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Technologies Used](#technologies-used)

## Features

**Drowsiness detection**

- Eye state classification: a CNN classifies each eye as open or closed and is combined with the Eye Aspect Ratio (EAR).
- Face drowsiness classification: a second CNN analyses the whole face and estimates a drowsiness probability.
- Temporal logic: eyes must stay closed for a set time before an alert is raised, which avoids false alarms from normal blinks.
- Smoothed predictions: the face model output is decided by a vote over the last six predictions.
- Audio alerts: a low tone for sustained eye closure and a high tone for the drowsiness model.
- Email alert (optional): sent at Stage 2 through EmailJS.

**Dashboard**

- Real-time metrics: attention score, EAR, MAR, closure duration, blink rate, drowsiness percentage, frame rate and latency.
- Live event log with timestamps.
- Three alert levels: normal, Stage 1 and Stage 2.
- Facial landmark overlay that can be switched on and off.
- Light and dark themes with a responsive layout for desktop and mobile.

## System Architecture

```
Webcam Feed
    |
    v
index.html -- MediaPipe Face Landmarker (face_landmarker.task)
    |             |-- Eye CNN (TensorFlow.js, 64x64 grayscale)
    |             |-- Face CNN (TensorFlow.js, 96x96)
    |             +-- EAR and MAR from landmarks
    |
    |-- Alert logic (closure timer, six-vote smoothing)
    |
    +-- Dashboard, audio alerts, optional email (EmailJS)
```

## Project Structure

```
driver-monitoring-system/
|
|-- index.html                  Web application (dashboard and detection logic)
|-- face_landmarker.task        MediaPipe face landmark model
|
|-- model_eye_state_cnn/        Eye CNN in TensorFlow.js format
|   |-- model.json
|   +-- group1-shard1of1.bin
|
|-- model_drowsiness_cnn/       Face drowsiness CNN in TensorFlow.js format
|   |-- model.json
|   +-- group1-shard*.bin
|
|-- drowsiness_detection.py     Desktop version (OpenCV window)
|-- driver_face_detection.py    Face detector demo
|-- eye_landmarks.py            Eye landmark experiments
|-- organize_dataset.py         Dataset preparation
|-- blaze_face_short_range.tflite
|-- eye_state_cnn.h5            Original Keras eye model
|-- drowsiness_cnn.h5           Original Keras drowsiness model
+-- README.md
```

## Installation

### Prerequisites

- Python 3.9 or higher (used only to serve the files locally)
- A webcam
- Google Chrome or Microsoft Edge

### Steps

1. Confirm the following are present next to `index.html`.

   ```
   model_eye_state_cnn/        model.json and .bin files
   model_drowsiness_cnn/       model.json and .bin files
   face_landmarker.task
   ```

2. For the desktop version only, install the dependencies.

   ```bash
   pip install opencv-python mediapipe tensorflow numpy
   ```

The web application itself requires no installation.

## Usage

### Web application

1. Start a local server in the project folder.

   ```bash
   python -m http.server 8000
   ```

2. Open `http://localhost:8000` in Chrome or Edge and press Ctrl + F5.

3. Select **Start camera** and allow camera access.

4. Confirm in the event log that the models loaded:

   ```
   Eye model loaded
   Drowsiness model loaded
   Both models loaded successfully
   Model ready
   ```

Do not open `index.html` directly from the file system. Camera access and model loading require `http://localhost` or HTTPS.

### Diagnostic options

Append these to the address to compare behaviour with the desktop version.

| Address | Effect |
|---|---|
| `/?bgr=0` | Pass RGB instead of BGR pixels to the face model |
| `/?gamma=1` | Disable the brightening step |
| `/?debug=1` | Display the exact eye and face crops passed to the models |

### Desktop version

```bash
python drowsiness_detection.py
```

Press Q in the OpenCV window to quit. The script loads `eye_state_cnn` and `drowsiness_cnn`; update the file names in the script if the models use the `.h5` extension.

## Models

| Model | Input | Output |
|---|---|---|
| Eye state CNN | 64 x 64 grayscale eye image | One sigmoid value: probability that the eye is open |
| Face drowsiness CNN | 96 x 96 color face image | One sigmoid value: 0.5 or higher indicates drowsiness |

Both models were trained in Keras and exported to TensorFlow.js format (`model.json` and `.bin` files) for use in the browser.

**Datasets**

- Eye state CNN: [dataset name and size to be added]
- Face drowsiness CNN: [dataset name and size to be added]

**Results**

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Eye state CNN | [to be added] | [to be added] | [to be added] |
| Face drowsiness CNN | [to be added] | [to be added] | [to be added] |

Measured in the browser: frame rate [to be added], latency [to be added] ms.

## Dashboard

| Section | Description |
|---|---|
| Live Driver Monitoring | Camera feed with landmark overlay, recording timer and frame rate |
| Driver Status | Attention score ring and current state (Attentive or Drowsy) |
| Attention Score | Combined score from eye closure and the face model |
| Eye State | Open or Closed, from the hybrid EAR and CNN rule |
| Eye Openness (EAR) | Eye aspect ratio from landmarks |
| Mouth Openness (MAR) | Mouth aspect ratio from landmarks |
| Closure Duration | Seconds both eyes have remained closed |
| Blink Rate | Blinks in the last minute |
| Drowsiness Model (face) | Smoothed drowsiness probability from the face CNN |
| Inference Latency | Processing time per frame |
| Alert System | Normal, Stage 1 and Stage 2 indicators |
| Event Log | Timestamped history of events and alerts |

## Configuration

Key settings are constants near the top of the script in `index.html`.

```js
// Eye decision (same rule as the desktop version)
const EAR_CLOSED   = 0.18;    // Below this value the eye is closed
const EAR_OPEN     = 0.25;    // Above this value the eye is open
const CNN_OPEN_MIN = 0.6;     // In between: CNN open probability required for "open"

// Alerts
const EYE_BEEP_AFTER = 1.0;   // Seconds the eyes must stay closed before the eye beep
const DROWSY_BEEP_AT = 0.50;  // Drowsiness value that starts the drowsy beep
const DROWSY_EVERY   = 3;     // Run the face model on every third frame

// Image preparation
const GAMMA = 0.6;            // Brightening step (set to 1 to disable)
```

**Alert stages**

| Stage | Trigger |
|---|---|
| Stage 1 | Eyes closed for 1 second, or the face model votes drowsy |
| Stage 2 | Eyes closed for 2 seconds (also sends the optional email) |

**Alert sounds**

| Sound | Trigger |
|---|---|
| Low tone (600 Hz) | Both eyes closed for 1 second or more |
| High tone (1000 Hz) | Drowsiness value above 50 percent |

### Email alerts (optional)

1. Create an account at [emailjs.com](https://www.emailjs.com), connect an email service, and create a template using the variables `{{to_email}}`, `{{stage}}`, `{{closure}}`, `{{time}}` and `{{message}}`.
2. In `index.html`, complete the `EMAIL` block with the Public Key, Service ID, Template ID and recipient, and set `enabled: true`.

Emails are sent at most once every five minutes. Because the public key is visible in the page source, restrict the allowed domains in the EmailJS dashboard.

## Technologies Used

| Technology | Purpose |
|---|---|
| HTML, CSS, JavaScript | Web dashboard and detection logic |
| MediaPipe Tasks Vision | Face landmark detection |
| TensorFlow.js | Running the CNN models in the browser |
| TensorFlow / Keras | Training the CNN models |
| OpenCV | Camera feed in the desktop version |
| Web Audio API | Audio alerts |
| EmailJS | Optional email alerts |
| GitHub Pages or Netlify | Hosting |
