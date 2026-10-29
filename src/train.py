
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def main():
    parser = argparse.ArgumentParser(description="Train an Iris classifier")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print("Training decision tree...")
    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy:.1%})")

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=iris.target_names,
        cmap="Blues",
    )
    plt.title("Iris Classification - Confusion Matrix")
    plt.tight_layout()

    output_file = output_dir / "confusion_matrix.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to: {output_file.resolve()}")

    model_file = output_dir / "model.joblib"
    import joblib; joblib.dump(model, model_file)
    print(f"Trained model saved to: {model_file.resolve()}")


if __name__ == "__main__":
    main()

