"""
LibrAI — Coleta de dados temporais para letras dinâmicas de LIBRAS.

Cada amostra = sequência de frames (landmarks da mão ao longo do tempo).
Salva como .npy em data/dynamic/<classe>/sample_X.npy
Shape de cada arquivo: (timesteps, 63)  →  21 pontos × 3 coords (x, y, z)

NÃO altera dataset.csv nem collect.py.
"""

import os
import time
import glob

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SEQ_LEN = 30                # frames por sequência
DYNAMIC_LETTERS = ["H", "J", "K", "X", "Y", "Z"]
COUNTDOWN_SEC = 3           # contagem regressiva antes de gravar
AUTO_SEQUENCES = 5          # quantas sequências gravar em modo automático
FPS_CAPTURE = 15            # taxa de captura (frames/s) durante gravação

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DATA_DIR = os.path.join(BASE_DIR, "data", "dynamic")


# ─── FEATURE EXTRACTION ─────────────────────────────────────────────────────
def extract_dynamic_features(hand_landmarks):
    """
    Extrai 63 features normalizadas (21 pontos × 3 coords: x, y, z).
    Normalização relativa ao pulso + escala pelo tamanho da mão.
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


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def next_sample_index(class_dir):
    """Retorna o próximo índice disponível para sample_X.npy."""
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
    """Conta quantas amostras existem para uma classe."""
    return len(glob.glob(os.path.join(class_dir, "sample_*.npy")))


def draw_info(frame, lines, start_y=30, scale=0.7, color=(255, 255, 255)):
    """Desenha múltiplas linhas de texto no frame."""
    for i, text in enumerate(lines):
        y = start_y + i * 35
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)


# ─── MAIN ────────────────────────────────────────────────────────────────────
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

    # Cria diretórios
    for letter in DYNAMIC_LETTERS:
        os.makedirs(os.path.join(DATA_DIR, letter), exist_ok=True)

    # Câmera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    # MediaPipe
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)

    current_label = DYNAMIC_LETTERS[0]
    state = "IDLE"            # IDLE, COUNTDOWN, RECORDING, SAVING
    countdown_start = 0
    sequence_buffer = []
    last_saved_path = None
    auto_mode = False
    auto_remaining = 0
    frame_interval = 1.0 / FPS_CAPTURE
    last_frame_time = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Detectar mão
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

            # ─── STATE MACHINE ───
            now = time.time()

            if state == "IDLE":
                info = [
                    f"Classe: {current_label}  |  Amostras: {n_samples}",
                    f"Mao: {'OK' if hand_detected else 'NAO DETECTADA'}",
                    "SPACE=gravar  A=auto  R=repetir  ESC=sair",
                ]
                if auto_mode:
                    info[2] = f"AUTO: restam {auto_remaining} sequencias"
                draw_info(frame, info)

                # Se auto mode, iniciar próxima gravação
                if auto_mode and auto_remaining > 0 and hand_detected:
                    state = "COUNTDOWN"
                    countdown_start = now

            elif state == "COUNTDOWN":
                elapsed = now - countdown_start
                remaining = COUNTDOWN_SEC - int(elapsed)
                if remaining <= 0:
                    state = "RECORDING"
                    sequence_buffer = []
                    last_frame_time = now
                else:
                    # Mostrar contagem regressiva grande
                    cv2.putText(frame, str(remaining), (w // 2 - 40, h // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                    info = [
                        f"Classe: {current_label}  |  Preparando...",
                        f"Gravando em {remaining}s..."
                    ]
                    draw_info(frame, info)

            elif state == "RECORDING":
                frames_needed = SEQ_LEN - len(sequence_buffer)

                if hand_detected and (now - last_frame_time >= frame_interval):
                    feats = extract_dynamic_features(hand_landmarks)
                    sequence_buffer.append(feats)
                    last_frame_time = now

                progress = len(sequence_buffer)
                bar_w = int((progress / SEQ_LEN) * (w - 40))

                # Barra de progresso
                cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 30), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, h - 50), (w - 20, h - 30), (255, 255, 255), 2)

                info = [
                    f"GRAVANDO: {current_label}  [{progress}/{SEQ_LEN}]",
                    f"Mao: {'OK' if hand_detected else 'PERDIDA - mantenha a mao visivel!'}",
                ]
                color = (0, 0, 255) if not hand_detected else (0, 255, 0)
                draw_info(frame, info, color=color)

                if progress >= SEQ_LEN:
                    state = "SAVING"

            elif state == "SAVING":
                # Salvar sequência
                seq_array = np.array(sequence_buffer, dtype=np.float32)  # (SEQ_LEN, 63)
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

            # ─── KEY HANDLING ───
            cv2.imshow("LibrAI - Coleta Dinamica", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("4"):  # ESC ou 4
                break

            # Selecionar classe
            for letter in DYNAMIC_LETTERS:
                if key == ord(letter.lower()):
                    if state == "IDLE":
                        current_label = letter
                        auto_mode = False
                        print(f"  Classe selecionada: {current_label}")

            # Gravar (SPACE)
            if key == ord(" ") and state == "IDLE":
                if hand_detected:
                    state = "COUNTDOWN"
                    countdown_start = now
                    auto_mode = False
                else:
                    print("  ⚠ Mão não detectada. Posicione a mão e tente novamente.")

            # Repetir (R) — apaga última e regrava
            if key == ord("r") and state == "IDLE":
                if last_saved_path and os.path.exists(last_saved_path):
                    os.remove(last_saved_path)
                    print(f"  🔄 Removido: {last_saved_path}")
                    last_saved_path = None
                else:
                    print("  Nenhuma amostra para repetir.")

            # Auto mode (A)
            if key == ord("a") and state == "IDLE":
                auto_mode = True
                auto_remaining = AUTO_SEQUENCES
                print(f"  Auto-mode: gravando {AUTO_SEQUENCES} sequências de '{current_label}'")

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    print("\nColeta finalizada.")
    # Resumo
    print("\nResumo:")
    for letter in DYNAMIC_LETTERS:
        d = os.path.join(DATA_DIR, letter)
        n = count_samples(d)
        print(f"  {letter}: {n} amostras")


if __name__ == "__main__":
    main()
