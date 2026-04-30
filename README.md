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



