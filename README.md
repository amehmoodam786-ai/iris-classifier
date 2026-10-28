
# Iris Classifier (Decision Tree)

## Overview

End-to-end machine learning example that builds a decision-tree classifier on the classic Iris dataset using scikit-learn.

The model predicts the species of an Iris flower using four measurements:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Quick start

### 1. Activate the virtual environment (Windows PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Train the model

```powershell
python src/train.py --test-size 0.2 --random-state 42
```

The script prints the model accuracy and saves the confusion matrix to `outputs/confusion_matrix.png`.

## Project structure

```text
iris-classifier/
├── data/
├── notebooks/
│   └── iris_model.ipynb
├── outputs/
│   └── confusion_matrix.png
├── src/
│   └── train.py
├── tests/
│   └── test_train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Results

The model is evaluated using accuracy and a confusion matrix. Results may vary if the dataset split or model settings change.

## Testing

Run the tests with:

```powershell
python -m pytest
```

## License

This project is licensed under the MIT License.
