import os
import time
import csv
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#configuração de auto-save (captura de frames por segundo automaticamente), e letras que serão captadas para treinamento do modelo
AUTO_CAPTURE_HZ = 10  
# Letras estáticas de LIBRAS (letras dinâmicas intencionalmente excluídas)
LETTERS = ["A","B","C","D","E","F","G","I","L","M","N","O","P","Q","R","S","T","U","V","W"]
#contagem de amostra de capturas para cada letra, almejando pelo menos 200 para cada
TARGET_SAMPLES = 200
sample_count = {letter: 0 for letter in LETTERS}
#procura de diretorio, arquivo csv, task, e etc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)



def extract_features(hand_landmarks):
   #função pra extrair os pontos da mão e normalizar para depois conseguir usá-los como um vetor
    pulso = hand_landmarks[0] #ponto do pulso
    base_dedomeio = hand_landmarks[9] #base do dedo do meio

    # faz uma escala com abse no tamanho da mão, tirado pela distancia euclidiana do pulso e base do dedo do meio
    hand_size = np.hypot(base_dedomeio.x - pulso.x, base_dedomeio.y - pulso.y)
    #so pra evitar divisao por zero
    hand_size = hand_size if hand_size > 1e-6 else 1e-6

    features = []
    for lm in hand_landmarks: #salva cada feature da mão, centralizada com base na posição e tmanho da mão
        features.append((lm.x - pulso.x) / hand_size)
        features.append((lm.y - pulso.y) / hand_size)

    return features


# ===== SAVE =====
def save_sample(hand_landmarks, label):
    features = extract_features(hand_landmarks)

    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            header = [f"f{i}" for i in range(len(features))] + ["label"]
            writer.writerow(header)

        writer.writerow(features + [label])

    # atualiza contador na memória quando
    sample_count[label] += 1

    # printa no terminal quando bater o alvo
    if sample_count[label] == TARGET_SAMPLES:
        print(f"o label {label} atingiu {TARGET_SAMPLES} amostras, siga para a proxima (pressione ] )")


# ===== MAIN =====
def main():
    print("=== LibrAI | Coleta de Dataset ===")
    print("Letras disponíveis:", LETTERS)
    # Letras estáticas de LIBRAS (letras dinâmicas intencionalmente excluídas)
    print(f"Teclas: A,B,C,D,E,F,G,I,L,M,N,O,P,Q,R,S,T,U,V,W (pressione pra trocar a label) | - (salvar 1) | 5 (auto {AUTO_CAPTURE_HZ}/s) |  4 (sair)")
    print("Dataset será salvo em:", CSV_PATH)

    current_label = LETTERS[0]
    auto_capture = False
    last_capture_time = 0.0
#começar a captura de camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("nao foi possivel abrir sua camera .")
#usa o modelo base com o .task de landmarker do mediapipe, e usa uma mao no maximo por frame, atualmente
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
#cria o objeto detector pra os pontos da mao para salvar os frames da camera ate que aperte o botao de parar
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame da câmera.")
                break
                #espelha imagem
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
                #converte de BGR (padrao) pra RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #transforma o resultado no formato que o TASKS api do mediapipe quer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            hand_landmarks = None
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                # desenha pontos
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # ===== AUTO-CAPTURE (numero definido no codigo/s) =====
            now = time.time()
            if auto_capture and hand_landmarks is not None:
                if now - last_capture_time >= 1.0 / AUTO_CAPTURE_HZ:
                    save_sample(hand_landmarks, current_label)
                    last_capture_time = now

            # ===== UI simples atual pra coleta de amostras=====
            cv2.putText(
                frame,
                f"Label: {current_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Auto-capture: {'ON' if auto_capture else 'OFF'} ({AUTO_CAPTURE_HZ}/s)",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if auto_capture else (0, 0, 255),
                2
            )
                #so pra manter contado o numero de amostras
            count = sample_count[current_label]
            status = f"{count} / {TARGET_SAMPLES}"
            done = count >= TARGET_SAMPLES
                #avisar quando atingir 200 amostras
            cv2.putText(
                frame,
                f"Samples: {status}{'  ✔' if done else ''}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if done else (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Keys: [A,S,E,F,L,N,P,Q,S,T,U,V]=label  [-]=save  [5]=auto  [4]=quit",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1
            )

            cv2.imshow("LibrAI - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            
            # ===== KEYS =====
            #fecha se pressionado
            if key == ord("4"):
                break
                #autosave
            if key == ord("5"):
                auto_capture = not auto_capture
                print("Auto-capture:", "ON" if auto_capture else "OFF")
                #salva 1 se pressionado
            if key == ord("-") and hand_landmarks is not None:
                save_sample(hand_landmarks, current_label)
                    #atualiza label
            for letter in LETTERS:
                if key == ord(letter.lower()):
                    current_label = letter
                    print("Label atual:", current_label)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
