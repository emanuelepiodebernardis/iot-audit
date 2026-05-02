# IoT/IIoT Intrusion Detection System using Machine Learning

Author: Emanuele Pio De Bernardis  
Affiliation: Università eCampus  
License: MIT  

---

## Overview
This repository implements a Machine Learning-based Intrusion Detection System (IDS) designed for IoT and IIoT network environments, with a focus on scalable deployment, adversarial robustness, and computational efficiency.

The system analyzes network flow telemetry derived from the TON_IoT dataset, simulating realistic IoT/IIoT smart infrastructure traffic under both benign and adversarial conditions.

The IDS supports two complementary detection paradigms:

Binary classification: distinguishing normal vs malicious traffic
Multiclass classification: identifying specific attack families (e.g., DoS, DDoS, MITM, ransomware, scanning, injection attacks)

The objective is to build a reproducible, benchmark-driven intrusion detection pipeline that jointly optimizes:

predictive performance
inference latency
model memory footprint
cross-domain generalization capability
---

## Research Context
Modern IoT/IIoT ecosystems introduce significant cybersecurity challenges due to:

heterogeneous and resource-constrained devices
large-scale distributed attack surfaces
high variability in network traffic distributions
difficulty in generalizing detection models across environments

This project addresses these challenges by focusing on:

Detection of network-based cyberattacks, including:
DoS / DDoS attacks

Man-in-the-Middle (MITM) attacks

botnet activity

injection and reconnaissance-based attacks

ransomware-related traffic patterns

evaluation under class imbalance conditions typical of real-world security datasets

robustness assessment under cross-domain shift (TON → CIC IoT 2023)

benchmarking of computational efficiency (latency + model size) for deployable IDS scenarios integration of model interpretability via SHAP-based explanations
---

## Pipeline Overview
The system follows a structured pipeline:

The proposed IDS follows a modular and reproducible pipeline:

Data preprocessing and cleaning
handling missing values and invalid flows
normalization of numerical features
encoding of categorical variables
Feature engineering
unified representation of network flow features
alignment across heterogeneous datasets (TON and CIC IoT 2023)
Supervised model training
multiple ML classifiers trained under identical preprocessing conditions
Evaluation framework
classification metrics (Accuracy, Precision, Recall, F1-score)
threshold-independent metrics (ROC-AUC, PR-AUC)
cross-validation (Stratified K-Fold)
Efficiency analysis
inference latency (ms per 1000 samples)
model size on disk (MB)
trade-off analysis between accuracy and computational cost
Explainability layer
SHAP-based global and local feature attribution
class-specific interpretability for attack categories (e.g., MITM detection analysis)

---

## Models and Results
The following supervised learning models are evaluated under identical experimental conditions on the held-out test set (38,095 samples, 20% stratified split):
Binary Classification (normal vs attack)
ModelF1ROC-AUCLatency (ms/1k)Size (MB)LightGBM0.9992>0.999977.61.375Random Forest0.9990>0.9999303.221.908XGBoost0.9989>0.999987.10.765MLP (DL baseline)0.99590.9993—0.119Decision Tree (d=5)0.99430.985647.20.019Logistic Regression0.99000.994555.50.016
A 5-fold stratified cross-validation confirms stable performance across splits
(LightGBM: F1 = 0.9993 ± 0.0001, variance < 0.0006 for all ensemble models).
Multiclass Classification (10 attack classes)
ModelAccuracyMacro F1Latency (ms/1k)XGBoost0.98960.9694129.4LightGBM0.98970.9693550.9Random Forest0.98800.9678317.1Logistic Regression0.85260.790343.8Decision Tree (d=5)0.81480.714542.8
The MITM class (208 test samples, 0.55% of test set) is the most challenging,
with XGBoost achieving F1 = 0.786 on this class.

## Quantization and Embedded Export
Three quantization pipelines are implemented for MCU deployment:
ModelFormatOrig. (KB)Quant. (KB)CRF1TargetLogistic RegressionC via m2cgen4.443.201.38x0.9900Arduino Mega / ESP32-C3Decision Tree (d=5)C via m2cgen8.074.761.69x0.9943Arduino Mega / ESP32-C3MLPTFLite Micro INT8121.6913.039.34x0.9959ESP32-C3XGBoostINT8 binary (.bin)771.12369.522.09x0.9989ESP32-C3LightGBMINT8 binary (.bin)1396.2473.8518.91x0.9992ESP32-C3
Quantized models are in quant_outputs/. The serialization/deserialization
code for the custom INT8 binary format is in embedded_model_io.py.

## Physical Hardware Benchmarks
All five quantized models were deployed and benchmarked on real hardware
using PlatformIO firmware (source in src/, configuration in platformio.ini).
ModelBoardMean latency (µs)SRAM used (B)SRAM limit (B)Logistic RegressionArduino Mega 25601,2707,1808,192Decision Tree d=5Arduino Mega 2560357,1828,192Logistic RegressionESP32-C3 SuperMini144—409,600Decision Tree d=5ESP32-C3 SuperMini3—409,600MLP TFLite INT8ESP32-C3 SuperMini8051,676409,600LightGBM INT8ESP32-C3 SuperMini5,564—409,600XGBoost INT8ESP32-C3 SuperMini8,240—409,600
The ESP32-C3 SuperMini (RISC-V 160 MHz) is 8.8–11.7× faster than the
Arduino Mega 2560 (AVR 16 MHz) on identical compiled C code.
All 40 predictions per session were correct (accuracy = 1.0 on test vectors).

## Cross-Domain Evaluation
Models trained on TON_IoT are evaluated against CIC-IoT-Dataset2023 using a
10-feature harmonised representation. Normalised distribution shift δ is computed
per feature; five of ten features exceed the δ > 1 threshold, with packet
asymmetry reaching δ = 5.95 — nearly six source standard deviations.

## Related Publication
This repository accompanies the following paper currently under submission:

De Bernardis, E. P., & Kuznetsov, O. (2026).
Lightweight Machine Learning Intrusion Detection for IoT/IIoT Networks:
Quantization Strategies and Physical Deployment on Resource-Constrained
Microcontrollers. Submitted to MDPI.

All code, model export scripts, PlatformIO firmware, and benchmark results
are publicly available in this repository under the MIT licence.

---

## Dataset: TON_IoT Network Dataset
Name: TON_IoT Network Dataset — IoT/IIoT network traffic for intrusion detection
Provider: Cyber Range & IoT Labs, UNSW Canberra (SEIT) — TON_IoT dataset collection
Official page: https://research.unsw.edu.au/projects/toniot-datasets
License: Creative Commons Attribution 4.0 International (CC BY 4.0) (see the TON_IoT site for details)

This repository uses the train/test network flows subset often distributed as train_test_network.csv (~29.9 MB; 44 columns). The flows were captured in realistic IoT/IIoT smart-environment scenarios using tools such as Argus and Bro (Zeek). The dataset contains benign and malicious traffic and is suitable for intrusion detection, anomaly detection, and ML benchmarking.
Download it from:
https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset

## Dataset: CIC-IoT Dataset 2023
Name: CIC IoT Dataset 2023
Provider: Canadian Institute for Cybersecurity (CIC), University of New Brunswick
Official page: https://www.unb.ca/cic/datasets/iotdataset-2023.html
License: Research/academic use (as defined by CIC dataset policy)

This dataset contains modern IoT network traffic capturing:

real-world IoT communications
diverse attack scenarios
updated attack patterns compared to older CIC datasets

It is used in this project for:

external validation (cross-domain testing)
robustness evaluation of trained models
domain shift analysis between TON_IoT and CIC-IoT environments

evaluating model generalization beyond the training distribution
