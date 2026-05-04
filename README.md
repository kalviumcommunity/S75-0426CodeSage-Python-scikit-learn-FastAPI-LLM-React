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

The repository is organized into distinct directories to maintain a clean separation of concerns and ensure scalability. The `data/` directory is partitioned into `raw/` and `processed/` folders to preserve the integrity of the original datasets while tracking the transformations applied during the pipeline. The `src/` folder contains the core logic of the application, modularized into specific scripts for preprocessing, feature engineering, training, and evaluation. Machine learning models and transformation artifacts are stored in the `models/` directory, while `logs/` and `reports/` provide a centralized location for tracking system behavior and performance metrics respectively. Additionally, the `notebooks/` directory is reserved for exploratory data analysis and prototyping, keeping the production-ready code in `src/` free from experimental clutter.

## Data Flow Mapping

The system follows a linear and reproducible data pipeline. Data begins as raw files in `data/raw/`, which are then loaded and cleaned by the `data_preprocessing` module. Once the data is cleaned, it is passed to the `feature_engineering` module, where it undergoes transformation into model-ready inputs. These transformed features are used by the `train` module to fit the machine learning model. After training, the `evaluate` module assesses the model's performance on a held-out test set, and detailed results are saved to the `reports/` directory. Finally, the trained model and feature engineering artifacts are saved to the `models/` directory, where they can be loaded by the `predict` module to generate inferences on new, unseen data.

## Design Justification

The separation of raw and processed data is a fundamental best practice in data engineering, ensuring that the original data source remains immutable and reproducible. By keeping notebooks separate from the source code, we prevent experimental scripts from interfering with the production pipeline, ensuring that `src/` contains only tested and versioned logic. Storing models outside of the source directory allows for easier versioning of large binary artifacts and prevents the codebase from becoming bloated. The independence of logs and reports ensures that diagnostic information and performance metrics are easily accessible without cluttering the execution environment. Centralizing all path management within `config.py` eliminates hardcoded paths, making the project portable across different environments and operating systems while simplifying maintenance.

---

## Project Setup Instructions

### Prerequisites
- **Python Version**: 3.13.x

### 1. Create the Virtual Environment
From the project root directory, run:
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
  Run the main workflow to train the model and evaluate its performance:
  ```bash
  python -m src.main
  ```
- **Prediction:**
  Run a sample prediction using the trained model:
  ```bash
  python -m src.predict
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



