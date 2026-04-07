"""
LibrAI — Treinamento do modelo GRU para letras dinâmicas de LIBRAS.

Lê sequências .npy de data/dynamic/<classe>/ e treina um modelo GRU.
Salva o melhor modelo em models/dynamic_gru.h5

NÃO altera dataset.csv nem o pipeline estático.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem GUI para salvar figuras
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# CONFIG 
DYNAMIC_LETTERS = ["H", "J", "K", "X", "Y", "Z"]
SEQ_LEN = 30                # timesteps por sequência (deve bater com dynamic_collect)
NUM_FEATURES = 63            # 21 landmarks × 3 coords
ADD_DELTAS = True            # adicionar velocidades entre frames como features extras
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15                # early stopping
VAL_SPLIT = 0.2
RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "dynamic")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_gru.h5")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "dynamic_labels.npy")
REPORT_DIR = os.path.join(BASE_DIR, "reports")


# DATA LOADING 
def load_sequences():
    """
    Carrega todos os .npy de data/dynamic/<classe>/.
    Retorna X (n_samples, SEQ_LEN, features), y (n_samples,), label_map.
    """
    X_list = []
    y_list = []
    classes_found = []

    for letter in DYNAMIC_LETTERS:
        class_dir = os.path.join(DATA_DIR, letter)
        files = sorted(glob.glob(os.path.join(class_dir, "sample_*.npy")))

        if not files:
            print(f" Nenhuma amostra encontrada para '{letter}' pulando.")
            continue

        classes_found.append(letter)
        for fpath in files:
            seq = np.load(fpath)  # (timesteps, features)
            X_list.append(seq)
            y_list.append(letter)

    if not X_list:
        raise RuntimeError("Nenhum dado encontrado em data/dynamic/. Execute dynamic_collect.py primeiro.")

    print(f"\nClasses encontradas: {classes_found}")
    print(f"Total de amostras: {len(X_list)}")

    # Contagem por classe
    unique, counts = np.unique(y_list, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} amostras")

    return X_list, y_list, classes_found


def pad_or_truncate(seq, target_len):
    """Padroniza comprimento da sequência: trunca ou faz padding com zeros."""
    current_len = seq.shape[0]
    n_feats = seq.shape[1]

    if current_len >= target_len:
        return seq[:target_len]
    else:
        pad = np.zeros((target_len - current_len, n_feats), dtype=np.float32)
        return np.vstack([seq, pad])


def add_delta_features(X):
    """
    Adiciona diferenças entre frames consecutivos como features extras.
    Input shape:  (n_samples, SEQ_LEN, features)
    Output shape: (n_samples, SEQ_LEN, features * 2)
    """
    # delta[t] = X[t] - X[t-1], delta[0] = zeros
    deltas = np.zeros_like(X)
    deltas[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, deltas], axis=-1)


# MODEL 
def build_model(input_shape, num_classes):
    """Constrói modelo GRU com Keras."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # GRU layers
        layers.GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.GRU(32, dropout=0.3, recurrent_dropout=0.2),

        # Classificação
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# TRAINING 
def main():
    print("=" * 50)
    print("  LibrAI | Treinamento GRU — Gestos Dinâmicos")
    print("=" * 50)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1) Carregar dados
    X_list, y_list, classes_found = load_sequences()

    # 2) Padronizar tamanho
    X_padded = np.array([pad_or_truncate(seq, SEQ_LEN) for seq in X_list], dtype=np.float32)
    print(f"\nShape após padding: {X_padded.shape}")  # (n, SEQ_LEN, 63)

    # 3) Deltas (opcional)
    if ADD_DELTAS:
        X_padded = add_delta_features(X_padded)
        print(f"Shape com deltas:   {X_padded.shape}")  # (n, SEQ_LEN, 126)

    # 4) Encode labels
    label_map = {letter: i for i, letter in enumerate(sorted(classes_found))}
    inv_label_map = {i: letter for letter, i in label_map.items()}
    y_encoded = np.array([label_map[lbl] for lbl in y_list], dtype=np.int32)
    num_classes = len(label_map)

    print(f"\nLabel map: {label_map}")
    print(f"Num classes: {num_classes}")

    # Salvar label map para inferência
    np.save(LABEL_MAP_PATH, label_map)
    print(f"Label map salvo em: {LABEL_MAP_PATH}")

    # 5) Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y_encoded,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"\nTrain: {X_train.shape[0]}  |  Val: {X_val.shape[0]}")

    # 6) Construir modelo
    import tensorflow as tf
    from tensorflow import keras

    input_shape = (X_padded.shape[1], X_padded.shape[2])  # (SEQ_LEN, features)
    model = build_model(input_shape, num_classes)
    model.summary()

    # 7) Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 8) Treinar
    print("\nTreinando...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 9) Avaliar
    print("\n" + "=" * 50)
    print("  AVALIAÇÃO")
    print("=" * 50)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nVal Loss: {val_loss:.4f}  |  Val Accuracy: {val_acc:.4f}")

    y_pred = np.argmax(model.predict(X_val), axis=1)
    label_names = [inv_label_map[i] for i in range(num_classes)]

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_names))

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(f"Labels: {label_names}")
    print(cm)

    # 10) Salvar modelo
    model.save(MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")

    # 11) Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plot_path = os.path.join(REPORT_DIR, "dynamic_training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Gráficos salvos em: {plot_path}")

    # Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax_cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix — GRU Dinâmico")
    cm_path = os.path.join(REPORT_DIR, "dynamic_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    print(f"Matriz de confusão salva em: {cm_path}")

    plt.close("all")
    print("\n Treinamento concludo!")


if __name__ == "__main__":
    main()
