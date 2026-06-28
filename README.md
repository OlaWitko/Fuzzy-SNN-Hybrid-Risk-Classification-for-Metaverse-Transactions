# Fuzzy-SNN-Hybrid-Risk-Classification-for-Metaverse-Transactions

A hybrid machine learning system that combines Fuzzy C-Means clustering with a neural network (FuzzySNN) to classify the risk level of blockchain financial transactions in an Open Metaverse environment. The model not only assigns risk categories but also produces soft membership scores, capturing the uncertainty inherent in real-world financial data.

What it does

Rather than forcing each transaction into a single hard risk class, this project uses fuzzy logic to represent degrees of risk — a transaction can partially belong to multiple risk levels simultaneously. The pipeline works in two stages:


Fuzzy C-Means (FCM) — unsupervised clustering that assigns each transaction a continuous membership score across three risk clusters (high, moderate, low)
FuzzySNN — a neural network trained to predict those fuzzy membership scores for new, unseen transactions


A KNN classifier is included as a baseline for comparison.

Dataset

Metaverse Financial Transactions Dataset — Kaggle

78,600 blockchain transactions with features including:

amount - Transaction amount
risk_score - Pre-computed risk score
login_frequency - User login frequency
session_duration - Session duration
hour_of_day - Time of transaction
transaction_type - Type of transaction
location_region - Geographic region
purchase_pattern - User purchase behaviour
age_group - User age group
anomaly - Risk label — high / moderate / low


Download metaverse_transactions_dataset.csv from Kaggle and place it in the same directory as the notebook before running.

Approach

EDA — class distribution, transaction amount histogram, correlation analysis
Preprocessing — label encoding for categorical features, StandardScaler standardization, removal of non-informative columns (timestamp, addresses)
Fuzzy C-Means — applied to 5 numerical features, producing a 3-cluster soft membership matrix; visualized via Risk Score vs Amount scatter plot
FuzzySNN architecture — fully connected network (input → 64 → 32 → output), ReLU activations, Sigmoid output, MSE loss, Adam optimizer, 50 epochs
Evaluation — confusion matrix, membership bar charts, radar chart for individual samples, MAE histogram, PCA projection of model output
KNN baseline — k=5, trained on hard labels for direct accuracy comparison

Results

FuzzySNN - Test MSE ~0.000012
FuzzySNN - Mean membership MAE ~0.0034
FuzzySNN - Classification accuracy (argmax) >99%
KNN (baseline) - Classification accuracy - moderate

The PCA projection of FuzzySNN outputs shows well-separated clusters in 2D, confirming the model has learned a meaningful internal representation — not just minimizing loss.

Requirements

torch
scikit-learn
scikit-fuzzy
pandas
numpy
matplotlib
seaborn

Install with:

bashpip install torch scikit-learn scikit-fuzzy pandas numpy matplotlib seaborn

Running the notebook

bashjupyter notebook FuzzySNN.ipynb

Key concepts demonstrated


Fuzzy C-Means unsupervised clustering
Hybrid fuzzy-neural architecture (soft label supervision)
Neural network training with PyTorch
Uncertainty-aware classification
PCA for model output visualization
Baseline comparison with KNN
