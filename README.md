# neurolinguaRT
NeuroLingua-RT: Raspberry Pi Real-Time Sleep Stage Forecasting Server

This repository contains the Raspberry Pi runtime implementation of NeuroLingua-RT, a lightweight, embedded-capable framework for real-time sleep stage forecasting using EEG/EOG signals.

The code implements a full online pipeline, including:

streaming EEG/EOG acquisition via UART,

real-time preprocessing and subwindow buffering,

ONNX-based neural inference,

next-epoch sleep stage forecasting,

optional relay-based closed-loop actuation,

and a lightweight web interface for visualization.

This implementation accompanies the NeuroLingua-RT paper and is intended for reproducibility and experimental validation, rather than offline model training.
1. System Overview

The system operates continuously on 3-second subwindows of EEG/EOG data and maintains a rolling buffer corresponding to past 30-second epochs.
After each new subwindow arrives, the model predicts:

the current sleep stage, and

the next (upcoming) sleep stage, enabling anticipatory operation.
This single script contains:

UART communication

Frame parsing and CRC verification

ACK/NACK handshake

Signal preprocessing

Filtering and normalization

Temporal buffering

Dataset-compatible 3-second windowing

Neural inference

ONNX Runtime execution

Softmax post-processing

Real-time control

Optional GPIO relay triggering

Web server

Live visualization of predictions

All components are designed to run fully on-device.
