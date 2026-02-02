# neurolinguaRT
# Raspberry Pi Real-Time Sleep Stage Forecasting Server

This repository contains the **Raspberry Pi runtime code** used in the *NeuroLingua / NeuroLingua-RT* framework for **real-time sleep stage prediction and forecasting** from EEG and EOG signals.

The script implements an **online, low-latency inference server** that receives physiological signals from an external acquisition system, processes them continuously, and performs neural network inference using a pre-trained ONNX model. The code is designed for **on-device execution** and was used for the real-time and embedded experiments reported in the paper.

---

## What This Code Does

The script `raspberry_sleep_final_webserver_2_forPaper.py` runs as a **continuous real-time service** on a Raspberry Pi and performs the following tasks:

1. Receives EEG/EOG samples from an external device via UART  
2. Verifies incoming data using CRC checks  
3. Applies signal preprocessing  
4. Segments the signal into **3-second subwindows**  
5. Maintains a rolling temporal buffer equivalent to 30-second epochs  
6. Executes a **pre-trained ONNX sleep-staging model**  
7. Predicts the current sleep stage and the upcoming sleep stage  
8. Optionally triggers GPIO relays based on predictions  
9. Serves predictions through a lightweight web interface  

All processing is performed **locally** on the Raspberry Pi.

---

## Intended Use

This code is intended for:

- Real-time sleep stage prediction experiments  
- Embedded and edge-AI validation of sleep staging models  
- Closed-loop sleep monitoring and actuation setups  
- Reproducibility of the NeuroLingua-RT runtime pipeline  

The code **does not** include model training or dataset handling.  
It assumes a **pre-trained model** is already available in ONNX format.

---

## Runtime Characteristics

- Subwindow duration: **3 seconds**  
- Epoch context: **rolling 30-second history**  
- Execution mode: **online / streaming**  
- Inference backend: **ONNX Runtime**  
- Typical latency: **tens of milliseconds per update** on Raspberry Pi 5  

Subwindows are used as internal modeling units; labels are defined at the epoch level during evaluation.

---

## Code File
neurolinguaRT_raspberry.py

The script is self-contained and includes:

- UART communication and framing  
- Signal preprocessing  
- Temporal buffering logic  
- ONNX model inference  
- Optional GPIO relay control  
- Web-based visualization server  

The implementation reflects the exact runtime configuration used in the paper and has not been simplified for demonstration purposes.

---

## Dependencies
numpy
scipy
onnxruntime
flask
pyserial
RPi.GPIO
The code requires Python 3.8+ and the following libraries:


GPIO support is optional and can be disabled if not needed.

---

## Model Requirement

A trained sleep-staging model exported to **ONNX format** must be provided.  
The model path is specified inside the script.

Optional INT8 quantization can be used to reduce latency and memory usage.

---

## Execution

To start the real-time server:

neurolinguaRT_raspberry.py


Once running, the script continuously processes incoming data and updates predictions in real time.

---

## Relation to the Paper

This code corresponds to the **embedded, real-time runtime layer** of the NeuroLingua / NeuroLingua-RT framework and was used for the Raspberry Pi experiments reported in the paper. It is released to support transparency and reproducibility.

---

## License

This code is provided for **research and academic use**.  
Commercial use requires permission from the authors.


