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

detection of network-based cyberattacks, including:
DoS / DDoS attacks
Man-in-the-Middle (MITM) attacks
botnet activity
injection and reconnaissance-based attacks
ransomware-related traffic patterns
evaluation under class imbalance conditions typical of real-world security datasets
robustness assessment under cross-domain shift (TON → CIC IoT 2023)
benchmarking of computational efficiency (latency + model size) for deployable IDS scenarios
integration of model interpretability via SHAP-based explanations
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

## Models Used
The following supervised learning models are evaluated under identical experimental conditions:

Logistic Regression (baseline linear classifier)
Random Forest (ensemble tree-based model)
XGBoost (gradient boosting framework)
LightGBM (optimized gradient boosting)
MLP Neural Network (deep learning baseline)

Each model is assessed in terms of:

detection performance
robustness under domain shift
computational efficiency

---

## Dataset: TON_IoT Network Dataset
Name: TON_IoT Network Dataset — IoT/IIoT network traffic for intrusion detection
Provider: Cyber Range & IoT Labs, UNSW Canberra (SEIT) — TON_IoT dataset collection
Official page: https://research.unsw.edu.au/projects/toniot-datasets
License: Creative Commons Attribution 4.0 International (CC BY 4.0) (see the TON_IoT site for details)

This repository uses the train/test network flows subset often distributed as train_test_network.csv (~29.9 MB; 44 columns). The flows were captured in realistic IoT/IIoT smart-environment scenarios using tools such as Argus and Bro (Zeek). The dataset contains benign and malicious traffic and is suitable for intrusion detection, anomaly detection, and ML benchmarking.
Download it from:
https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset
