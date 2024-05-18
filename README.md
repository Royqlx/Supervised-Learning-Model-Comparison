# Supervised Learning Model Comparison and Feature Engineering
This repository contains a comprehensive machine learning pipeline to analyze a dataset, compare the performance of different classifiers, and perform feature engineering to improve model performance. The main steps include data exploration, visualization, model training, evaluation, and comparison. Additionally, feature engineering is performed to reduce dimensionality and improve the computational efficiency of the models.

## Introduction
This project aims to explore a dataset, visualize its distribution, train multiple machine learning models, and compare their performance using various evaluation metrics. Additionally, feature engineering is performed to reduce dimensionality and assess the impact on model performance.

## Setup
To run the code, you need to have the following libraries installed:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
```


