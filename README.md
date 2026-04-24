# IoT/IIoT Intrusion Detection System using Machine Learning

Author: Emanuele De Bernardis  
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

## Dataset
Dataset used: TON_IoT Network Dataset  
https://research.unsw.edu.au/projects/toniot-datasets
Due to size limitations, the dataset is not included in this repository.

Download it from:
https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset
