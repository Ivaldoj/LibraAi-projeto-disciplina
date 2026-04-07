import os
import time
import csv
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

AUTO_CAPTURE_HZ = 10  

# Letras estáticas de LIBRAS — letras que exigem movimento (H, J, K, X, Y, Z)
# ficam fora daqui e são tratadas no dynamic_collect.py
LETTERS = ["A","B","C","D","E","F","G","I","L","M","N","O","P","Q","R","S","T","U","V","W"]

TARGET_SAMPLES = 200
sample_count = {letter: 0 for letter in LETTERS}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)


def extract_features(hand_landmarks):
    # Normalização em dois passos: centraliza em relação ao pulso e divide pelo
    # tamanho da mão (distância pulso → base do dedo médio). Sem isso, a mesma
    # letra feita perto ou longe da câmera geraria vetores diferentes, confundindo
    # o modelo na hora do treino.
    pulso = hand_landmarks[0]
    base_dedomeio = hand_landmarks[9]

    hand_size = np.hypot(base_dedomeio.x - pulso.x, base_dedomeio.y - pulso.y)
    hand_size = hand_size if hand_size > 1e-6 else 1e-6  # evita divisão por zero

    features = []
    for lm in hand_landmarks:
        features.append((lm.x - pulso.x) / hand_size)
        features.append((lm.y - pulso.y) / hand_size)

    # Retorna 42 valores (21 pontos × x e y). Z omitido pois letras estáticas
    # não dependem de profundidade — o dynamic_collect.py inclui Z por isso.
    return features


def save_sample(hand_landmarks, label):
    features = extract_features(hand_landmarks)

    # Escreve cabeçalho apenas se o arquivo ainda não existe
    write_header = not os.path.exists(CSV_PATH)

    # Modo append para nunca sobrescrever dados já coletados
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [f"f{i}" for i in range(len(features))] + ["label"]
            writer.writerow(header)
        writer.writerow(features + [label])

    sample_count[label] += 1

    if sample_count[label] == TARGET_SAMPLES:
        print(f"o label {label} atingiu {TARGET_SAMPLES} amostras, siga para a proxima (pressione ] )")


def main():
    print("=== LibrAI | Coleta de Dataset ===")
    print("Letras disponíveis:", LETTERS)
    print(f"Teclas: A,B,C,D,E,F,G,I,L,M,N,O,P,Q,R,S,T,U,V,W (pressione pra trocar a label) | - (salvar 1) | 5 (auto {AUTO_CAPTURE_HZ}/s) |  4 (sair)")
    print("Dataset será salvo em:", CSV_PATH)

    current_label = LETTERS[0]
    auto_capture = False
    last_capture_time = 0.0

    # CAP_DSHOW é um backend mais estável no Windows; se falhar, cai pro padrão
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("nao foi possivel abrir sua camera .")

    # num_hands=1 porque o alfabeto de LIBRAS usa uma mão —
    # detectar duas aumentaria o custo computacional sem benefício
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame da câmera.")
                break

            frame = cv2.flip(frame, 1)  # espelha a imagem, mais intuitivo para o usuário
            h, w, _ = frame.shape

            # MediaPipe exige RGB; OpenCV captura em BGR — conversão obrigatória
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            hand_landmarks = None
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                # Desenha os 21 pontos sobre a mão para feedback visual
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Auto-capture: salva na frequência definida sem precisar apertar tecla
            now = time.time()
            if auto_capture and hand_landmarks is not None:
                if now - last_capture_time >= 1.0 / AUTO_CAPTURE_HZ:
                    save_sample(hand_landmarks, current_label)
                    last_capture_time = now

            # UI mínima na tela: label atual, status do auto-capture e contador
            cv2.putText(frame, f"Label: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Auto-capture: {'ON' if auto_capture else 'OFF'} ({AUTO_CAPTURE_HZ}/s)",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if auto_capture else (0, 0, 255), 2)

            count = sample_count[current_label]
            done = count >= TARGET_SAMPLES
            cv2.putText(frame, f"Samples: {count} / {TARGET_SAMPLES}{'  ✔' if done else ''}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if done else (255, 255, 255), 2)
            cv2.putText(frame, "Keys: [A,S,E,F,L,N,P,Q,S,T,U,V]=label  [-]=save  [5]=auto  [4]=quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("LibrAI - Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("4"):
                break
            if key == ord("5"):
                auto_capture = not auto_capture
                print("Auto-capture:", "ON" if auto_capture else "OFF")
            if key == ord("-") and hand_landmarks is not None:
                save_sample(hand_landmarks, current_label)
            for letter in LETTERS:
                if key == ord(letter.lower()):
                    current_label = letter
                    print("Label atual:", current_label)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
