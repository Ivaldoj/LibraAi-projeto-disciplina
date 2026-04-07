import os
import time
import glob

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Letras que em LIBRAS exigem movimento — não podem ser capturadas
# com um único frame estático, por isso têm pipeline próprio aqui
SEQ_LEN = 30
DYNAMIC_LETTERS = ["H", "J", "K", "X", "Y", "Z"]
COUNTDOWN_SEC = 3
AUTO_SEQUENCES = 5
FPS_CAPTURE = 15            # taxa de captura durante a gravação

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DATA_DIR = os.path.join(BASE_DIR, "data", "dynamic")


def extract_dynamic_features(hand_landmarks):
    # Mesma lógica de normalização do collect.py, mas inclui Z (profundidade)
    # porque gestos dinâmicos podem envolver aproximar/afastar a mão da câmera.
    # Isso gera 63 features (21 × 3) em vez das 42 (21 × 2) do pipeline estático.
    pulso = hand_landmarks[0]
    base_dedomeio = hand_landmarks[9]

    # Distância euclidiana 3D para o tamanho da mão
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


def next_sample_index(class_dir):
    # Determina o próximo número disponível para evitar sobrescrever amostras existentes
    existing = glob.glob(os.path.join(class_dir, "sample_*.npy"))
    if not existing:
        return 0
    indices = []
    for path in existing:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            indices.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(indices) + 1 if indices else 0


def count_samples(class_dir):
    return len(glob.glob(os.path.join(class_dir, "sample_*.npy")))


def draw_info(frame, lines, start_y=30, scale=0.7, color=(255, 255, 255)):
    # Utilitário para escrever várias linhas no frame sem repetir cv2.putText
    for i, text in enumerate(lines):
        y = start_y + i * 35
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)


def main():
    print("=" * 50)
    print("  LibrAI | Coleta de Gestos Dinâmicos")
    print("=" * 50)
    print(f"Classes: {DYNAMIC_LETTERS}")
    print(f"Frames por sequência: {SEQ_LEN}")
    print(f"Dados salvos em: {DATA_DIR}")
    print()
    print("Controles:")
    print("  H/J/K/X/Y/Z  → selecionar classe")
    print("  SPACE           → iniciar gravação (com contagem regressiva)")
    print("  R               → repetir última amostra (apaga e regrava)")
    print("  A               → modo auto (grava várias sequências)")
    print("  ESC / 4         → sair")
    print()

    for letter in DYNAMIC_LETTERS:
        os.makedirs(os.path.join(DATA_DIR, letter), exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)

    current_label = DYNAMIC_LETTERS[0]

    # Máquina de estados: controla em qual fase da coleta o programa está.
    # IDLE → COUNTDOWN → RECORDING → SAVING → IDLE
    state = "IDLE"
    countdown_start = 0
    sequence_buffer = []
    last_saved_path = None
    auto_mode = False
    auto_remaining = 0
    frame_interval = 1.0 / FPS_CAPTURE  # intervalo mínimo entre capturas
    last_frame_time = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            hand_detected = False
            hand_landmarks = None

            if result.hand_landmarks:
                hand_detected = True
                hand_landmarks = result.hand_landmarks[0]
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            class_dir = os.path.join(DATA_DIR, current_label)
            n_samples = count_samples(class_dir)
            now = time.time()

            # --- IDLE: aguarda comando do usuário ---
            if state == "IDLE":
                info = [
                    f"Classe: {current_label}  |  Amostras: {n_samples}",
                    f"Mao: {'OK' if hand_detected else 'NAO DETECTADA'}",
                    "SPACE=gravar  A=auto  R=repetir  ESC=sair",
                ]
                if auto_mode:
                    info[2] = f"AUTO: restam {auto_remaining} sequencias"
                draw_info(frame, info)

                # No modo auto, dispara a próxima gravação assim que a mão for detectada
                if auto_mode and auto_remaining > 0 and hand_detected:
                    state = "COUNTDOWN"
                    countdown_start = now

            # --- COUNTDOWN: dá tempo pro usuário se preparar ---
            elif state == "COUNTDOWN":
                elapsed = now - countdown_start
                remaining = COUNTDOWN_SEC - int(elapsed)
                if remaining <= 0:
                    state = "RECORDING"
                    sequence_buffer = []
                    last_frame_time = now
                else:
                    cv2.putText(frame, str(remaining), (w // 2 - 40, h // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                    draw_info(frame, [
                        f"Classe: {current_label}  |  Preparando...",
                        f"Gravando em {remaining}s..."
                    ])

            # --- RECORDING: captura frames na taxa definida por FPS_CAPTURE ---
            elif state == "RECORDING":
                if hand_detected and (now - last_frame_time >= frame_interval):
                    feats = extract_dynamic_features(hand_landmarks)
                    sequence_buffer.append(feats)
                    last_frame_time = now

                progress = len(sequence_buffer)

                # Barra de progresso no rodapé da tela
                bar_w = int((progress / SEQ_LEN) * (w - 40))
                cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 30), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, h - 50), (w - 20, h - 30), (255, 255, 255), 2)

                color = (0, 0, 255) if not hand_detected else (0, 255, 0)
                draw_info(frame, [
                    f"GRAVANDO: {current_label}  [{progress}/{SEQ_LEN}]",
                    f"Mao: {'OK' if hand_detected else 'PERDIDA - mantenha a mao visivel!'}",
                ], color=color)

                if progress >= SEQ_LEN:
                    state = "SAVING"

            # --- SAVING: empilha os frames e salva como .npy ---
            elif state == "SAVING":
                # .npy é mais adequado que CSV aqui porque cada amostra é um array 2D (30×63)
                seq_array = np.array(sequence_buffer, dtype=np.float32)
                idx = next_sample_index(class_dir)
                save_path = os.path.join(class_dir, f"sample_{idx}.npy")
                np.save(save_path, seq_array)
                last_saved_path = save_path

                n_samples = count_samples(class_dir)
                print(f"  ✅ Salvo: {save_path}  (shape={seq_array.shape})  |  Total {current_label}: {n_samples}")

                sequence_buffer = []
                state = "IDLE"

                if auto_mode:
                    auto_remaining -= 1
                    if auto_remaining <= 0:
                        auto_mode = False
                        print("  Auto-mode finalizado.")

            cv2.imshow("LibrAI - Coleta Dinamica", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("4"):
                break

            for letter in DYNAMIC_LETTERS:
                if key == ord(letter.lower()):
                    if state == "IDLE":
                        current_label = letter
                        auto_mode = False
                        print(f"  Classe selecionada: {current_label}")

            if key == ord(" ") and state == "IDLE":
                if hand_detected:
                    state = "COUNTDOWN"
                    countdown_start = now
                    auto_mode = False
                else:
                    print("  ⚠ Mão não detectada. Posicione a mão e tente novamente.")

            # R apaga a última amostra para que o usuário refaça caso tenha errado o gesto
            if key == ord("r") and state == "IDLE":
                if last_saved_path and os.path.exists(last_saved_path):
                    os.remove(last_saved_path)
                    print(f"  🔄 Removido: {last_saved_path}")
                    last_saved_path = None
                else:
                    print("  Nenhuma amostra para repetir.")

            if key == ord("a") and state == "IDLE":
                auto_mode = True
                auto_remaining = AUTO_SEQUENCES
                print(f"  Auto-mode: gravando {AUTO_SEQUENCES} sequências de '{current_label}'")

    finally:
        # finally garante que câmera e detector são fechados mesmo se ocorrer erro
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    print("\nColeta finalizada.")
    print("\nResumo:")
    for letter in DYNAMIC_LETTERS:
        d = os.path.join(DATA_DIR, letter)
        n = count_samples(d)
        print(f"  {letter}: {n} amostras")


if __name__ == "__main__":
    main()
