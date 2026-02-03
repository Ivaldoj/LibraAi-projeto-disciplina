import os
import time
import random
from collections import deque

import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# pygame pra feedback de dopamina
import pygame
pygame.mixer.init()


ALLOWED_LETTERS = ["A","B","C","D","E","F","G","I","L","M","N","O","P","Q","R","S","T","U","V","W"]

HARD_WORDS = [
    "OI","OLA","ALO","SOL","MAR","LUA","SOM","MAO","LUZ",
    "BOLA","AMOR","VIDA","CASA","RUAO","VILA","SINO","MALA","NOME","GOLA",
    "BALAO","LIVRO","AMIGO","AMIGA","SINAL","VENTO","NUVEM","RUMO","MANO",
    "IARA","DIOGO","GIGI","DAI","MEI","IRIS",
    "BRASIL","AMIGOS","CASAO","BALSA","VIGOR","MOLAS","NAVIO","GOMAS","SALVO"
]



# =========================
# CONFIG
# =========================
CAM_INDEX = 0
USE_CAP_DSHOW = True               

HISTORY_N = 12                     # buffer do voto
STABLE_RATIO = 0.50                # >=50% no voto para considerar estável
MIN_CONF = 0.40                    # confiança mínima (se modelo suportar proba)

ROUND_SECONDS_EASY = 15.0          # tempo por letra no EASY
ROUND_SECONDS_HARD = 9.0           # tempo por letra no HARD
COOLDOWN_NEXT = 0.7                # evita avanço repetido

REF_BOX_W = 400                    # tamanho max da imagem de referência
REF_BOX_H = 250

#  CONFIG DO FLASH pra dar dopamina apos acertar
FLASH_DURATION_MS = 70            # duração de cada piscada (ms)
FLASH_COLOR = "#00FF00"            # verde neon
FLASH_ALPHA = 0.4                  # intensidade do overlay (0-1)


FORCE_ALLOWED_CLASSES = ["A","B","C","D","E","F","G","I","L","M","N","O","P","Q","R","S","T","U","V","W"]


# PATHS de cada arquivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_PATH = os.path.join(BASE_DIR, "models", "librai_rf.joblib")
REF_DIR = os.path.join(BASE_DIR, "assets", "references")

#  PATHS DOS SONS 
SOUND_LETTER_PATH = os.path.join(BASE_DIR, "assets", "sounds", "correctletter.mp3")
SOUND_WORD_PATH = os.path.join(BASE_DIR, "assets", "sounds", "correctword.wav")



# FEATURES (MESMA do collect.py)
def extract_features(hand_landmarks):
    """Função pra extrair os pontos da mão e normalizar para depois conseguir usá-los como um vetor"""
    pulso = hand_landmarks[0]
    base_dedomeio = hand_landmarks[9]

    hand_size = np.hypot(base_dedomeio.x - pulso.x, base_dedomeio.y - pulso.y)
    hand_size = hand_size if hand_size > 1e-6 else 1e-6

    features = []
    for lm in hand_landmarks:
        features.append((lm.x - pulso.x) / hand_size)
        features.append((lm.y - pulso.y) / hand_size)
    return np.array(features, dtype=np.float32)


def majority_vote(history: deque):
    """Com base no histórico de letras detectadas, pega uma maioria"""
    if len(history) == 0:
        return None, 0.0
    values, counts = np.unique(np.array(history), return_counts=True)
    idx = int(np.argmax(counts))
    voted = str(values[idx])
    ratio = float(counts[idx]) / float(len(history))
    return voted, ratio


def fit_image_to_box(img_bgr, box_w, box_h):
    """Redimensiona a imagem mantendo aspect ratio"""
    if img_bgr is None:
        return None
    ih, iw = img_bgr.shape[:2]
    scale = min(box_w / iw, box_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    if nw <= 0 or nh <= 0:
        return None
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)



# APP em si
class LibrAIApp(tk.Tk):
    def __init__(self):
        super().__init__()
                #titulo grande la em cima
        self.title("LibrAI Challenge (Static LIBRAS Letters)")
        self.geometry("1100x650")
        self.resizable(True, True)

        # carrega os sons pra quando acertar
        self.sound_letter = None
        self.sound_word = None
        self._load_sounds()
        

        # ----- modelo com as classes que podem e n podem ser usadas
        self.model = joblib.load(MODEL_PATH)
        self.model_classes = [str(c) for c in getattr(self.model, "classes_", [])]
        if not self.model_classes:
            raise RuntimeError("modelo sem classes")

        if FORCE_ALLOWED_CLASSES is None:
            self.allowed = list(self.model_classes)
        else:
            self.allowed = [c for c in FORCE_ALLOWED_CLASSES if c in self.model_classes]
            if not self.allowed:
                raise RuntimeError("as classes permitidas nao batem com a classe do modelo")

        # camera
        api = cv2.CAP_DSHOW if USE_CAP_DSHOW else 0
        self.cap = cv2.VideoCapture(CAM_INDEX, api)
        if not self.cap.isOpened():
            raise RuntimeError("erro ao abrir a camera")

        # mediapipe landmarker
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # salva o gamestate
        self.history = deque(maxlen=HISTORY_N)
        self.running = False
        self.game_over = False

        self.mode = "EASY"

        self.score = 0
        self.round_start = None

        self.target_letter = None

        self.target_word = random.choice(HARD_WORDS)
        self.hard_pos = 0

        self.last_next_time = 0.2

        # cache das imagens pra nao ficar lendo toda vez
        self.ref_cache = {}
        self.ref_img_tk = None

        # estado do flash
        self.flash_overlay = None      # referência ao canvas overlay
        self.flash_active = False      # controle de piscada
        

        # UI
        self._build_ui()

        # inicia o loop
        self.after(15, self.update_loop)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_sounds(self):
        #Carrega os arquivos de som
        if os.path.exists(SOUND_LETTER_PATH):
            self.sound_letter = pygame.mixer.Sound(SOUND_LETTER_PATH)
            if os.path.exists(SOUND_WORD_PATH):
                self.sound_word = pygame.mixer.Sound(SOUND_WORD_PATH)

    # ========== FEEDBACK DE DOPAMINA ==========
    def _play_letter_sound(self):
        """Toca o som de acerto de letra"""
        if self.sound_letter:
            try:
                self.sound_letter.play()
            except Exception as e:
                print(f"Erro ao tocar som: {e}")

    def _play_word_sound(self):
        """Toca o som de palavra completa"""
        if self.sound_word:
            try:
                self.sound_word.play()
            except Exception as e:
                print(f"Erro ao tocar som: {e}")

    def _flash_screen(self, times=1):
        #pisca a tela quando acerta, dependendo de ser palavra ou uma letra so
        if self.flash_active:
            return  # já está piscando

        self.flash_active = True
        self._do_flash(times, times)

    def _do_flash(self, remaining, total):
        """Executa uma sequência de piscadas"""
        if remaining <= 0:
            self.flash_active = False
            return

        # Mostra overlay verde
        self._show_flash_overlay()

        # Esconde após FLASH_DURATION_MS
        self.after(FLASH_DURATION_MS, self._hide_flash_overlay)

        # Próxima piscada (com intervalo)
        interval = FLASH_DURATION_MS * 2
        self.after(interval, lambda: self._do_flash(remaining - 1, total))

    def _show_flash_overlay(self):
        #Mostra o overlay verde sobre o vídeo
        if self.flash_overlay is None:
            return
        self.flash_overlay.place(x=0, y=0, relwidth=1, relheight=1)

    def _hide_flash_overlay(self):
        #Esconde o overlay
        if self.flash_overlay is None:
            return
        self.flash_overlay.place_forget()

    def _on_correct_letter(self):
        """Chamado quando acerta uma letra"""
        self._play_letter_sound()
        self._flash_screen(times=1)

    def _on_correct_word(self):
        """Chamado quando completa uma palavra"""
        self._play_word_sound()
        self._flash_screen(times=2)



    # UI do app em si
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        right = ttk.Frame(root)

        left.grid(row=0, column=0, sticky="n")
        right.grid(row=0, column=1, padx=(15, 0), sticky="n")

        # ========== CONTAINER DO VÍDEO COM OVERLAY ==========
        self.video_container = tk.Frame(left, bg="black")
        self.video_container.pack()

        # Camera label
        self.video_label = ttk.Label(self.video_container)
        self.video_label.pack()
            #overlay do flash verde
        self.flash_overlay = tk.Frame(self.video_container, bg=FLASH_COLOR)
        

        # painel la da direita
        title = ttk.Label(right, text="LibrAI Challenge", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            right,
            text="Faça a letra solicitada (LIBRAS estático) antes do tempo acabar.",
            font=("Segoe UI", 10)
        )
        subtitle.pack(anchor="w", pady=(0, 12))

        # Modo facil
        self.mode_var = tk.StringVar(value="Mode: EASY")
        mode_lbl = ttk.Label(right, textvariable=self.mode_var, font=("Segoe UI", 12, "bold"))
        mode_lbl.pack(anchor="w", pady=(0, 8))

        # placar
        self.score_var = tk.StringVar(value="Score: 0")
        score_lbl = ttk.Label(right, textvariable=self.score_var, font=("Segoe UI", 14, "bold"))
        score_lbl.pack(anchor="w", pady=(0, 10))

        # letra alvo e letra detectada
        self.target_var = tk.StringVar(value="Target: -")
        ttk.Label(right, textvariable=self.target_var, font=("Segoe UI", 13)).pack(anchor="w", pady=(0, 6))

        self.pred_var = tk.StringVar(value="Detected: -")
        ttk.Label(right, textvariable=self.pred_var, font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 6))
        #instrução para começar o jogo simples
        self.status_var = tk.StringVar(value="Press Start to play.")
        ttk.Label(right, textvariable=self.status_var, font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 12))

        # barra do timer
        ttk.Label(right, text="Time:", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.timer_canvas = tk.Canvas(right, width=380, height=22, bg="#222222", highlightthickness=0)
        self.timer_canvas.pack(anchor="w", pady=(4, 14))
        self.timer_rect = self.timer_canvas.create_rectangle(0, 0, 380, 22, fill="#00cc66", width=0)

        # botoes
        controls = ttk.Frame(right)
        controls.pack(anchor="w", pady=(6, 0))
            #botoes de inciiar quitar ou resetar
        self.start_btn = ttk.Button(controls, text="Start", command=self.start_game)
        self.reset_btn = ttk.Button(controls, text="Reset", command=self.reset_game)
        self.quit_btn = ttk.Button(controls, text="Quit", command=self.on_close)

        self.start_btn.grid(row=0, column=0, padx=(0, 8))
        self.reset_btn.grid(row=0, column=1, padx=(0, 8))
        self.quit_btn.grid(row=0, column=2, padx=(0, 12))

        # botoes de modo
        self.easy_btn = ttk.Button(controls, text="Easy", command=self.set_easy_mode)
        self.hard_btn = ttk.Button(controls, text="Hard", command=self.set_hard_mode)
        self.easy_btn.grid(row=0, column=3, padx=(0, 8))
        self.hard_btn.grid(row=0, column=4)

        # inserir palavra custom do modo dificil
        word_frame = ttk.Frame(right)
        word_frame.pack(anchor="w", pady=(10, 6))

        ttk.Label(word_frame, text="Word (HARD):", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.word_entry = ttk.Entry(word_frame, width=22)
        self.word_entry.grid(row=0, column=1, padx=(8, 8))
        self.word_entry.insert(0, self.target_word)

        self.set_word_btn = ttk.Button(word_frame, text="Set", command=self.set_hard_word)
        self.set_word_btn.grid(row=0, column=2)
#label da dica
        hint = ttk.Label(
            right,
            text="Dica: mantenha a mão estável por ~1s para confirmar.",
            font=("Segoe UI", 9)
        )
        hint.pack(anchor="w", pady=(8, 8))

        instruc = ttk.Label(
            right,
            text="USE A MÃO ESQUERDA COM A PALMA VIRADA PRA A SUA CAMERA!!!",
            font=("Segoe UI", 9)
        )
        instruc.pack(anchor="sw", pady=(10, 10))

        # label das imagens de referência para o modo fácil
        ttk.Label(right, text="Referência (modo fácil):", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 6))

        self.ref_container = ttk.Frame(right)
        self.ref_container.pack(anchor="w", fill="x")

        self.ref_img_label = ttk.Label(self.ref_container)
        self.ref_img_label.pack(anchor="w")

        self.ref_text_label = ttk.Label(self.ref_container, text="", font=("Segoe UI", 9))
        self.ref_text_label.pack(anchor="w", pady=(6, 0))

        # referencia inicial (estado)
        self._update_reference_image()

    
    # função de cada modo e de definição de letra
   
    def set_easy_mode(self):
        self.mode = "EASY"
        self.mode_var.set("Mode: EASY")
        self.status_var.set("Modo EASY: referência ativada.")
        self.history.clear()
        self._update_reference_image()
        self._refresh_target_text()

    def set_hard_mode(self):
        self.mode = "HARD"
        self.mode_var.set("Mode: HARD")
        self.status_var.set("Modo HARD: sem referência + sequência rápida de letras.")
        self.history.clear()
        self._update_reference_image()
        self._refresh_target_text()
    #funcao de definicao de palavra no modo dificil (customizada)
    def set_hard_word(self):
        word = self.word_entry.get().strip().upper()
        word = "".join([c for c in word if c.isalpha()])
        if not word:
            self.status_var.set("Digite uma palavra válida (A-Z).")
            return
        self.target_word = word
        self.hard_pos = 0
        self.status_var.set(f"Palavra do HARD definida: {self.target_word}")
        self._refresh_target_text()

    def _pick_new_word(self):
        #escolhe palavra nova no modo dificil
        if not HARD_WORDS:
            return "DIOGO"

        if len(HARD_WORDS) == 1:
            return HARD_WORDS[0]

        candidates = [w for w in HARD_WORDS if w != self.target_word]
        return random.choice(candidates) if candidates else random.choice(HARD_WORDS)
    # controle do jogo
    def start_game(self):
        if self.running:
            return
        self.running = True
        self.game_over = False
        self.score = 0
        self.score_var.set("Score: 0")
        self.history.clear()
        self.hard_pos = 0
        self._next_round()
    #status do jogo apos apertar botao de reset
    def reset_game(self):
        self.running = False
        self.game_over = False
        self.score = 0
        self.round_start = None
        self.target_letter = None
        self.hard_pos = 0
        self.history.clear()

        self.score_var.set("Score: 0")
        self.pred_var.set("Detected: -")
        self.status_var.set("Press Start to play.")
        self._set_timer_ratio(1.0)
        self._refresh_target_text()
        self._update_reference_image()
        #status do jogo apos acabar (so ao perder)
    def end_game(self):
        self.running = False
        self.game_over = True
        self.status_var.set(f"Game Over! Final score: {self.score}. Press Reset/Start.")
        self._set_timer_ratio(0.0)
    #funcao pra setar o tempo em segundos de duracao do round da letra
    def _get_round_seconds(self):
        return ROUND_SECONDS_EASY if self.mode == "EASY" else ROUND_SECONDS_HARD
    #FUNCAO PARA DEFINIR O ALVO DE ACORDO COM A DIFICULDADE DO MODO
    def _current_target(self):
        if self.mode == "EASY":
            return self.target_letter
        if self.hard_pos >= len(self.target_word):
            return None
        return self.target_word[self.hard_pos]
    #FUNCAO PARA IR PRA A PROXIMO ROUND (LETRA)
    def _next_round(self):
        self.round_start = time.time()
        self.history.clear()
        self.last_next_time = time.time()

        if self.mode == "EASY":
            self.target_letter = random.choice(self.allowed)

        self.status_var.set("Go!")
        self._refresh_target_text()
        self._update_reference_image()
#ATUALIZAR ALVO 
    def _refresh_target_text(self):
        if self.mode == "EASY":
            t = self.target_letter if self.target_letter else "-"
            self.target_var.set(f"Target: {t}")
            return

        if self.target_word:
            if self.hard_pos >= len(self.target_word):
                self.target_var.set(f"Target: {self.target_word} (complete ✅)")
            else:
                progress = self.target_word[:self.hard_pos] + "_" * (len(self.target_word) - self.hard_pos)
                current = self.target_word[self.hard_pos]
                self.target_var.set(f"Target: {self.target_word} | {progress} | now: {current}")
        else:
            self.target_var.set("Target: -")

            #CARREGAR IMAGENS DE REFERENCIA NO MODO FÁCIL
    def _load_ref_image(self, letter):
        if not letter:
            return None
        if letter in self.ref_cache:
            return self.ref_cache[letter]

        for ext in ("png",):
            path = os.path.join(REF_DIR, f"{letter}.{ext}")
            if os.path.exists(path):
                img = cv2.imread(path)
                self.ref_cache[letter] = img
                return img

        self.ref_cache[letter] = None
        return None

    def _update_reference_image(self):
        #ATUALIZAR REFERENCIA (IAMGEM) NO MODO FÁCIL, NO DIFÍCIL ESCONDE
        if self.mode != "EASY":
            self.ref_img_label.configure(image="")
            self.ref_text_label.configure(text="(Sem referência no modo difícil)")
            self.ref_img_tk = None
            return

        letter = self.target_letter
        if not letter:
            self.ref_img_label.configure(image="")
            self.ref_text_label.configure(text="(Press Start para iniciar)")
            self.ref_img_tk = None
            return

        img = self._load_ref_image(letter)

        ref_fit = fit_image_to_box(img, REF_BOX_W, REF_BOX_H)
            #CONVERTE A REFERENCIA DO PADRAO DO OPENCV PRA RGB
        ref_rgb = cv2.cvtColor(ref_fit, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(ref_rgb)
        self.ref_img_tk = ImageTk.PhotoImage(pil_img)

        self.ref_img_label.configure(image=self.ref_img_tk)
        self.ref_text_label.configure(text=f"Faça a letra: {letter}")

        
    def _set_timer_ratio(self, ratio):
        #controla a barra de tempo visual do jogo
        ratio = max(0.0, min(1.0, ratio)) #deixa o valor entre 0 e 1 sempre
        width = int(380 * ratio) #largura da barra proporcional ao tempo
        self.timer_canvas.coords(self.timer_rect, 0, 0, width, 22) #redesenha com o tamnho
        #cor de acordo com a porcentagem de tempo sobrando
        if ratio > 0.5:
            color = "#00cc66"
        elif ratio > 0.2:
            color = "#ffcc00"
        else:
            color = "#ff4444"
        self.timer_canvas.itemconfig(self.timer_rect, fill=color)

    #
    # MAIN LOOP
    
    def update_loop(self):
        #captura frame da camera, espelha, converte para RGB que é o que o mediapipe quer, e detecta
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            voted = None
            ratio = 0.0
            conf = None

            if result.hand_landmarks:
                #verifica se detectou mao, desenha na tela, e extrai as features
                hand_landmarks = result.hand_landmarks[0]

                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                feats = extract_features(hand_landmarks).reshape(1, -1)

                predicted = None
                if hasattr(self.model, "predict_proba"):
                    # pega classe com maior probabilidade e guarda a confiança
                    proba = self.model.predict_proba(feats)[0]
                    idx = int(np.argmax(proba))
                    predicted = str(self.model.classes_[idx])
                    conf = float(proba[idx])
                else:
                    #se nao tem proba, so predicta logo 
                    predicted = str(self.model.predict(feats)[0])

                if predicted in self.allowed:
                    #so adiciona ao hisoptrico se for uma letra permitida
                    self.history.append(predicted)

                voted, ratio = majority_vote(self.history)

            else:
                if len(self.history) > 0:
                    self.history.popleft()

            # atualiza a logica do jogo 
            if self.running and not self.game_over:
                elapsed = time.time() - self.round_start
                round_seconds = self._get_round_seconds()
                ratio_left = 1.0 - (elapsed / round_seconds)
                self._set_timer_ratio(ratio_left)

                if voted is None:
                    self.pred_var.set("Detected: -")
                else:
                    if conf is not None:
                        self.pred_var.set(f"Detected: {voted} (stable {ratio*100:.0f}% | conf ~ {conf*100:.0f}%)")
                    else:
                        self.pred_var.set(f"Detected: {voted} (stable {ratio*100:.0f}%)")

                if elapsed >= round_seconds:
                    #se o tempo passou do tempo do round, acaba
                    self.end_game()
                else:
                    #atualiza o alvo que deve ser feito
                    target = self._current_target()

                    stable_ok = (ratio >= STABLE_RATIO)
                    conf_ok = (conf is None) or (conf >= MIN_CONF)

                    if target is not None and voted == target and stable_ok and conf_ok:
                        #atuazliacao de acerto
                        if time.time() - self.last_next_time >= COOLDOWN_NEXT:
                            self.score += 1
                            self.score_var.set(f"Score: {self.score}")

                            if self.mode == "HARD":
                                self.hard_pos += 1

                                if self.hard_pos >= len(self.target_word):
                                    # feedback de palavra completa e vai pra proxima
                                    self._on_correct_word()
                                    
                                    self.score += 2
                                    self.score_var.set(f"Score: {self.score}")
                                    self.target_word = self._pick_new_word()
                                    self.hard_pos = 0
                                    self.status_var.set(f"✅ Palavra completa! +2 bônus | Próxima: {self.target_word}")
                                else:
                                    #  FEEDBACK: LETRA CORRETA 
                                    self._on_correct_letter()
                                    
                                    self.status_var.set(f"✅ Letra correta! Próxima: {self.target_word[self.hard_pos]}")
                            else:
                                 #  FEEDBACK: LETRA CORRETA 
                                
                                self.status_var.set("✅ Correto! proximo...")
                                self._on_correct_letter()

                            self._next_round()
            else:
                if not self.game_over:
                    #mantem a barra de tempo cheia se o game ta roando e nao acabou
                    self._set_timer_ratio(1.0)

            self._show_frame(frame)

        self.after(20, self.update_loop)

    def _show_frame(self, frame_bgr):
        #converte do opencv pra tkinter
        frame = cv2.resize(frame_bgr, (680, 510), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #converte pra RGB do pillow/tkinter

        pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil)
        #converte tudo

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        #guarda referencia e atualiza label

    def on_close(self):
        #encerra pygame ao finalizar o programa e encerra tudo
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = LibrAIApp()
    app.mainloop()