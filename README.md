# IoT/IIoT Intrusion Detection System using Machine Learning

Author: Emanuele Pio De Bernardis  
Affiliation: Università eCampus  
License: MIT  

---

## Overview
This project presents a Machine Learning-based Intrusion Detection System (IDS) for IoT and IIoT environments.

The system analyzes network flow data from the TON_IoT Network Dataset and supports both:

- Binary classification (normal vs attack)
- Multiclass classification (attack type identification)

The goal is to design a **reproducible, efficient, and deployable IDS pipeline**, balancing predictive performance and computational cost.

---

## Research Context
IoT and IIoT systems are highly vulnerable due to device heterogeneity and limited computational resources.

This project focuses on:

- Detection of network-based cyber-attacks (DoS, DDoS, MITM, ransomware, etc.)
- Evaluation of model robustness and class imbalance
- Analysis of computational efficiency (latency, model size)
- Reproducible ML pipeline for IDS systems

---

## Pipeline Overview
The system follows a structured pipeline:

1. Data preprocessing and cleaning
2. Feature encoding (numerical + categorical)
3. Model training (supervised learning)
4. Evaluation using advanced metrics
5. Efficiency analysis (latency and memory)

---

## Models Used
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM

---

## Dataset: TON_IoT Network Dataset
Name: TON_IoT Network Dataset — IoT/IIoT network traffic for intrusion detection
Provider: Cyber Range & IoT Labs, UNSW Canberra (SEIT) — TON_IoT dataset collection
Official page: https://research.unsw.edu.au/projects/toniot-datasets
License: Creative Commons Attribution 4.0 International (CC BY 4.0) (see the TON_IoT site for details)

This repository uses the train/test network flows subset often distributed as train_test_network.csv (~29.9 MB; 44 columns). The flows were captured in realistic IoT/IIoT smart-environment scenarios using tools such as Argus and Bro (Zeek). The dataset contains benign and malicious traffic and is suitable for intrusion detection, anomaly detection, and ML benchmarking.
Download it from:
https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset
