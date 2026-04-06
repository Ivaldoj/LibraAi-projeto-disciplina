import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# base do diretorio projeto fora da pasta models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "librai_rf.joblib")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    # Le e carrega o csv para um dataframe
    df = pd.read_csv(DATASET_PATH)

    # Checagem de balanceamento das amostras
    counts = df["label"].value_counts().sort_index()
    print("Amostras por classe:")
    print(counts)
    print()

    # Separa features de rotulos
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(str)

    # Splita os dados para teste e treino
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Define o modelo RandomForest
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    # Treina e avalia
    print("Treinando RandomForest...")
    model.fit(X_train, y_train)

    print("\nAvaliando...")
    y_pred = model.predict(X_test)

    labels_sorted = sorted(np.unique(y))
    report_str = classification_report(y_test, y_pred, labels=labels_sorted)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print("\nClassification report:")
    print(report_str)
    print("\nConfusion matrix:")
    print("Labels:", labels_sorted)
    print(cm)

    # Salva modelo
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")

    # ── REPORTS ──────────────────────────────────────────────

    # 1) Classification report em texto
    report_txt_path = os.path.join(REPORT_DIR, "static_classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("LibrAI -- RandomForest Estatico | Classification Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Amostras por classe:\n{counts.to_string()}\n\n")
        f.write(report_str)
    print(f"Report salvo em: {report_txt_path}")

    # 2) Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion Matrix -- RandomForest Estatico")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "static_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Matriz de confusao salva em: {cm_path}")

    # 3) Importancia das features (top 20)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    ax_imp.bar(range(20), importances[indices], color="#00E5FF", edgecolor="#252B3B")
    ax_imp.set_xticks(range(20))
    ax_imp.set_xticklabels([f"f{i}" for i in indices], rotation=45)
    ax_imp.set_title("Top 20 Feature Importances -- RandomForest")
    ax_imp.set_xlabel("Feature index")
    ax_imp.set_ylabel("Importancia")
    ax_imp.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    imp_path = os.path.join(REPORT_DIR, "static_feature_importance.png")
    plt.savefig(imp_path, dpi=150)
    print(f"Feature importance salva em: {imp_path}")

    plt.close("all")
    print("\n[OK] Treinamento concluido!")


if __name__ == "__main__":
    main()