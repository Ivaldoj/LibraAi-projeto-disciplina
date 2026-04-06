"""
LibrAI — Módulo de inferência em tempo real para gestos dinâmicos.

Mantém um buffer circular de frames (deque) e prediz quando o buffer está cheio.
Pode ser importado pelo app.py ou executado standalone para teste.

USO:
    from dynamic_predict import DynamicPredictor
    predictor = DynamicPredictor()
    predictor.feed(hand_landmarks)   # alimentar a cada frame
    result = predictor.predict()     # retorna (classe, confiança) ou None
"""

import os
import numpy as np
from collections import deque

# ??? CONFIG ??????????????????????????????????????????????????????????????????
SEQ_LEN = 30
NUM_FEATURES = 63     # 21 pontos × 3 coords
ADD_DELTAS = True     # deve bater com dynamic_train.py
CONFIDENCE_THRESHOLD = 0.5   # confiança mínima para retornar predição

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dynamic_gru.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "dynamic_labels.npy")


# ??? FEATURE EXTRACTION ?????????????????????????????????????????????????????
def extract_dynamic_features(hand_landmarks):
    """
    Extrai 63 features normalizadas (21 pontos × 3 coords).
    Idêntica à versão em dynamic_collect.py.
    """
    pulso = hand_landmarks[0]
    base_dedomeio = hand_landmarks[9]

    hand_size = np.sqrt(
        (base_dedomeio.x - pulso.x) ** 2 +
        (base_dedomeio.y - pulso.y) ** 2 +
        (base_dedomeio.z - pulso.z) ** 2
    )
    hand_size = max(hand_size, 1e-6)

    features = []
    for lm in hand_landmarks:
        features.append((lm.x - pulso.x) / hand_size)
        features.append((lm.y - pulso.y) / hand_size)
        features.append((lm.z - pulso.z) / hand_size)

    return np.array(features, dtype=np.float32)


# ??? PREDICTOR CLASS ?????????????????????????????????????????????????????????
class DynamicPredictor:
    """
    Preditor em tempo real para gestos dinâmicos.
    
    Fluxo:
        1. Chame feed(hand_landmarks) a cada frame com mão detectada
        2. Chame predict() para obter resultado quando buffer estiver cheio
        3. Chame clear() para resetar o buffer
    """

    def __init__(self, model_path=None, label_map_path=None, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.buffer = deque(maxlen=seq_len)
        self.model = None
        self.label_map = None       # {letra: index}
        self.inv_label_map = None   # {index: letra}
        self.loaded = False

        _model_path = model_path or MODEL_PATH
        _label_path = label_map_path or LABEL_MAP_PATH

        self._load_model(_model_path, _label_path)

    def _load_model(self, model_path, label_map_path):
        """Carrega modelo GRU e label map."""
        if not os.path.exists(model_path):
            print(f"  ? Modelo dinâmico não encontrado: {model_path}")
            print("    Execute dynamic_train.py primeiro.")
            return

        if not os.path.exists(label_map_path):
            print(f"  ? Label map não encontrado: {label_map_path}")
            return

        try:
            import tensorflow as tf
            # Silenciar logs do TF
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            tf.get_logger().setLevel("ERROR")

            self.model = tf.keras.models.load_model(model_path)
            self.label_map = np.load(label_map_path, allow_pickle=True).item()
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            self.loaded = True
            print(f"  [OK] Modelo dinamico carregado: {model_path}")
            print(f"     Classes: {list(self.label_map.keys())}")
        except Exception as e:
            print(f"  [ERRO] Erro ao carregar modelo dinamico: {e}")
            self.loaded = False

    def feed(self, hand_landmarks):
        """
        Adiciona um frame ao buffer.
        Recebe hand_landmarks do MediaPipe.
        """
        feats = extract_dynamic_features(hand_landmarks)
        self.buffer.append(feats)

    def is_ready(self):
        """Retorna True se o buffer está cheio e o modelo está carregado."""
        return self.loaded and len(self.buffer) == self.seq_len

    def predict(self):
        """
        Prediz a classe com base no buffer atual.
        
        Retorna:
            (classe: str, confiança: float) se buffer cheio e confiança > threshold
            None caso contrário
        """
        if not self.is_ready():
            return None

        # Montar sequência
        seq = np.array(list(self.buffer), dtype=np.float32)  # (SEQ_LEN, 63)

        # Adicionar deltas se configurado
        if ADD_DELTAS:
            deltas = np.zeros_like(seq)
            deltas[1:, :] = seq[1:, :] - seq[:-1, :]
            seq = np.concatenate([seq, deltas], axis=-1)  # (SEQ_LEN, 126)

        # Batch dimension
        seq_batch = np.expand_dims(seq, axis=0)  # (1, SEQ_LEN, features)

        # Predição
        proba = self.model.predict(seq_batch, verbose=0)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        predicted_class = self.inv_label_map.get(idx, "?")

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        return predicted_class, confidence

    def predict_and_reset(self):
        """Prediz e limpa o buffer (útil para gestos pontuais)."""
        result = self.predict()
        self.clear()
        return result

    def clear(self):
        """Limpa o buffer."""
        self.buffer.clear()

    def buffer_progress(self):
        """Retorna progresso do buffer (0.0 a 1.0)."""
        return len(self.buffer) / self.seq_len

    @property
    def dynamic_classes(self):
        """Retorna lista de classes do modelo dinâmico."""
        if self.label_map:
            return list(self.label_map.keys())
        return []


# ??? STANDALONE TEST ?????????????????????????????????????????????????????????
def main():
    """Teste standalone: abre webcam e mostra predições do GRU em tempo real."""
    import cv2
    import mediapipe as mp_lib
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    print("=" * 50)
    print("  LibrAI | Teste de Predição Dinâmica (GRU)")
    print("=" * 50)

    predictor = DynamicPredictor()
    if not predictor.loaded:
        print("Modelo não disponível. Saindo.")
        return

    # Câmera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # MediaPipe
    task_path = os.path.join(BASE_DIR, "hand_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)

    print("\nControles:")
    print("  C ? limpar buffer")
    print("  ESC ? sair")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                predictor.feed(hand_landmarks)

            # Progresso do buffer
            progress = predictor.buffer_progress()
            bar_w = int(progress * (w - 40))
            cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 25), (0, 200, 255), -1)
            cv2.rectangle(frame, (20, h - 40), (w - 20, h - 25), (255, 255, 255), 2)

            # Predição
            pred = predictor.predict()
            if pred:
                cls, conf = pred
                text = f"GRU: {cls} ({conf*100:.0f}%)"
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Buffer: {len(predictor.buffer)}/{predictor.seq_len}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow("LibrAI - Dynamic Predict Test", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == ord("c"):
                predictor.clear()
                print("Buffer limpo.")

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
