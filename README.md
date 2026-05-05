# Code Error Predictor + Fix Assistant

An intelligent debugging assistant that predicts programming errors and provides fixes with clear explanations using Machine Learning and Large Language Models.

---

## Problem Statement

Debugging is one of the most time-consuming tasks in software development. Developers, especially beginners, often struggle to understand error messages and identify the correct fixes.

This results in:
- Increased development time
- Reduced productivity
- Difficulty in learning and understanding concepts

Traditional rule-based systems are not effective because error patterns are diverse, context-dependent, and continuously evolving. A scalable and adaptive solution is required.

---

## Solution

This project builds a system that combines Machine Learning and LLMs to:

- Predict the type of error from code snippets or error logs
- Suggest possible fixes
- Provide clear, human-readable explanations

The system helps developers debug faster and understand errors more effectively.

---

## Tech Stack

### Backend
- Python
- FastAPI

### Machine Learning
- scikit-learn
- Pandas
- NumPy
- TF-IDF Vectorization

### LLM Integration
- OpenAI API / OpenRouter

### Frontend
- React.js

---

## Repository Structure Explanation

The repository follows a strict separation of concerns to ensure maintainability and reproducibility:

- **`data/`**: Divided into `raw/` (immutable source data) and `processed/` (data generated during the pipeline).
- **`models/`**: Stores serialized model and preprocessing artifacts (e.g., `.pkl` files).
- **`reports/`**: Contains evaluation metrics and performance reports.
- **`src/`**: The core source code, modularized by responsibility:
  - `data_loader.py`: Handles file I/O and basic data validation.
  - `train.py`: Orchestrates the training pipeline, including splitting, fitting, and artifact saving.
  - `predict.py`: Standalone inference script for generating predictions on new data.
  - `config.py`: Centralized path management and hyperparameter configuration.
- **`notebooks/`**: Reserved for exploratory data analysis (EDA) and prototyping.

## Data Flow Mapping

1. **Ingestion**: Raw data is loaded from `data/raw/` via the `data_loader` module.
2. **Training Pipeline**:
   - Data is split into training and testing sets.
   - The `FeatureEngineer` is fitted **only** on the training set.
   - The model is trained on the transformed training data.
   - Performance is evaluated on the held-out test set.
   - Artifacts (model, preprocessor) and reports are saved to `models/` and `reports/`.
3. **Inference**:
   - The `predict` module loads the saved artifacts.
   - New input data is validated and transformed using the saved preprocessor (no re-fitting).
   - The model generates predictions and confidence scores.

## Design Justification

The separation of **raw and processed data** ensures that original datasets are never modified, maintaining a "single source of truth." By isolating **notebooks** from the `src/` directory, we prevent experimental code from being accidentally included in the production pipeline. **Model artifacts** are saved externally to keep the source code lightweight and facilitate versioning of binary files. The **independence of logs and reports** allows for performance tracking without cluttering the operational environment. Finally, the **centralized configuration** in `config.py` ensures that all modules use consistent paths and parameters, making the project portable and easy to maintain.

---

## Project Setup Instructions

### Prerequisites
- **Python Version**: 3.13.x

### 1. Create the Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Environment
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project
- **Training and Evaluation:**
  ```bash
  python -m src.train
  ```
- **Prediction (Inference):**
  ```bash
  python -m src.predict --input data/sample_input.csv
  ```

---

## How It Works

1. The user inputs an error message or code snippet.
2. The input is processed and converted into numerical features using TF-IDF.
3. A trained machine learning model predicts the error category.
4. The predicted error and input are sent to an LLM API.
5. The LLM generates:
   - Suggested fixes
   - Clear explanations
6. The results are displayed through a React-based user interface.

---

## Features

- Error type prediction using machine learning
- Automatic fix suggestions
- Simple and understandable explanations
- Works with unseen error inputs
- FastAPI-based backend for efficient processing
- React-based user interface

---

## Team and Responsibilities

| Name     | Responsibility |
|----------|---------------|
| Abishek  | Machine Learning model development, FastAPI backend integration |
| Lakshmi  | Data collection, preprocessing, feature engineering |
| Lakshmi Shankar & Abishek | Frontend development using React, UI/UX design |



