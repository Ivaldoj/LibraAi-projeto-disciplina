"""
LibrAI — app.py INTEGRADO com pipeline dinâmico (GRU).

Baseado na versão ATUAL do app.py (UI escura com widgets customizados).
Todas as linhas novas estão marcadas com # 🆕 DYNAMIC
"""

import os
import time
import random
from collections import deque

import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, font as tkfont
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pygame
pygame.mixer.init()

# 🆕 DYNAMIC — importar preditor dinâmico
from dynamic_predict import DynamicPredictor, extract_dynamic_features  # 🆕 DYNAMIC


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

HISTORY_N = 12
STABLE_RATIO = 0.50
MIN_CONF = 0.40

ROUND_SECONDS_EASY = 15.0
ROUND_SECONDS_HARD = 9.0
COOLDOWN_NEXT = 0.7

REF_BOX_W = 340
REF_BOX_H = 220

FLASH_DURATION_MS = 70
FLASH_COLOR = "#00FF7F"
FLASH_ALPHA = 0.4

FORCE_ALLOWED_CLASSES = ["A","B","C","D","E","F","G","I","L","M","N","O","P","Q","R","S","T","U","V","W"]

# 🆕 DYNAMIC — config do detector de movimento
MOVEMENT_THRESHOLD = 0.15                                                  # 🆕 DYNAMIC
DYNAMIC_LETTERS_SET = {"H", "K", "Q", "P", "G", "F", "T"}                 # 🆕 DYNAMIC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_PATH = os.path.join(BASE_DIR, "models", "librai_rf.joblib")
REF_DIR = os.path.join(BASE_DIR, "assets", "references")

SOUND_LETTER_PATH = os.path.join(BASE_DIR, "assets", "sounds", "correctletter.mp3")
SOUND_WORD_PATH = os.path.join(BASE_DIR, "assets", "sounds", "correctword.wav")

# =========================
# DESIGN TOKENS
# =========================
BG_DEEP    = "#0D0F14"
BG_PANEL   = "#13161E"
BG_CARD    = "#1A1E2A"
BG_BORDER  = "#252B3B"
ACCENT     = "#00E5FF"
ACCENT2    = "#7B61FF"
SUCCESS    = "#00FF7F"
WARNING    = "#FFD700"
DANGER     = "#FF4560"
TEXT_PRI   = "#F0F4FF"
TEXT_SEC   = "#8892A4"
TEXT_DIM   = "#4A5568"
TIMER_BG   = "#1A1E2A"


def extract_features(hand_landmarks):
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
    if len(history) == 0:
        return None, 0.0
    values, counts = np.unique(np.array(history), return_counts=True)
    idx = int(np.argmax(counts))
    voted = str(values[idx])
    ratio = float(counts[idx]) / float(len(history))
    return voted, ratio


def fit_image_to_box(img_bgr, box_w, box_h):
    if img_bgr is None:
        return None
    ih, iw = img_bgr.shape[:2]
    scale = min(box_w / iw, box_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    if nw <= 0 or nh <= 0:
        return None
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


# =========================
# CUSTOM WIDGETS
# =========================

class DarkButton(tk.Canvas):
    """Custom styled button with hover effect."""
    def __init__(self, parent, text, command=None, accent=False, danger=False, width=110, height=36, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=BG_PANEL, highlightthickness=0, cursor="hand2", **kwargs)
        self._text = text
        self._command = command
        self._accent = accent
        self._danger = danger
        self._btn_w = width
        self._btn_h = height
        self._hovered = False
        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)

    def _get_colors(self):
        if self._accent:
            bg = ACCENT if self._hovered else "#00B8CC"
            fg = BG_DEEP
            border = ACCENT
        elif self._danger:
            bg = "#2A1A1A" if not self._hovered else "#3A1A1A"
            fg = DANGER
            border = DANGER
        else:
            bg = BG_CARD if not self._hovered else BG_BORDER
            fg = TEXT_PRI
            border = BG_BORDER
        return bg, fg, border

    def _draw(self):
        self.delete("all")
        bg, fg, border = self._get_colors()
        r = 6
        w, h = self._btn_w, self._btn_h
        self.create_arc(0, 0, r*2, r*2, start=90, extent=90, fill=bg, outline=border)
        self.create_arc(w-r*2, 0, w, r*2, start=0, extent=90, fill=bg, outline=border)
        self.create_arc(0, h-r*2, r*2, h, start=180, extent=90, fill=bg, outline=border)
        self.create_arc(w-r*2, h-r*2, w, h, start=270, extent=90, fill=bg, outline=border)
        self.create_rectangle(r, 0, w-r, h, fill=bg, outline=bg)
        self.create_rectangle(0, r, w, h-r, fill=bg, outline=bg)
        self.create_line(r, 0, w-r, 0, fill=border)
        self.create_line(r, h, w-r, h, fill=border)
        self.create_line(0, r, 0, h-r, fill=border)
        self.create_line(w, r, w, h-r, fill=border)
        self.create_text(w//2, h//2, text=self._text, fill=fg,
                         font=("Consolas", 10, "bold"))

    def _on_enter(self, e):
        self._hovered = True
        self._draw()

    def _on_leave(self, e):
        self._hovered = False
        self._draw()

    def _on_click(self, e):
        if self._command:
            self._command()


class ToggleButton(tk.Canvas):
    """Mode toggle button (Easy / Hard)."""
    def __init__(self, parent, text, command=None, active=False, width=90, height=34, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=BG_PANEL, highlightthickness=0, cursor="hand2", **kwargs)
        self._text = text
        self._command = command
        self._active = active
        self._btn_w = width
        self._btn_h = height
        self._draw()
        self.bind("<ButtonRelease-1>", self._on_click)

    def set_active(self, val):
        self._active = val
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self._btn_w, self._btn_h
        if self._active:
            bg, fg, border = ACCENT2, TEXT_PRI, ACCENT2
        else:
            bg, fg, border = BG_CARD, TEXT_SEC, BG_BORDER
        r = h // 2
        self.create_oval(0, 0, h, h, fill=bg, outline=border)
        self.create_oval(w-h, 0, w, h, fill=bg, outline=border)
        self.create_rectangle(r, 0, w-r, h, fill=bg, outline=bg)
        self.create_line(r, 0, w-r, 0, fill=border)
        self.create_line(r, h, w-r, h, fill=border)
        self.create_text(w//2, h//2, text=self._text, fill=fg,
                         font=("Consolas", 10, "bold"))

    def _on_click(self, e):
        if self._command:
            self._command()


class SeparatorLine(tk.Canvas):
    def __init__(self, parent, width=360, **kwargs):
        super().__init__(parent, width=width, height=2,
                         bg=BG_PANEL, highlightthickness=0, **kwargs)
        self.create_line(0, 1, width, 1, fill=BG_BORDER)


class Badge(tk.Label):
    """Small info badge."""
    def __init__(self, parent, text, color=ACCENT, **kwargs):
        super().__init__(parent, text=text, fg=color, bg=BG_CARD,
                         font=("Consolas", 9), padx=6, pady=2, **kwargs)


# =========================
# MAIN APP
# =========================

class LibrAIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LibrAI — LIBRAS Sign Challenge")
        self.geometry("1180x700")
        self.minsize(1000, 620)
        self.resizable(True, True)
        self.configure(bg=BG_DEEP)

        # sounds
        self.sound_letter = None
        self.sound_word = None
        self._load_sounds()

        # model
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

        # 🆕 DYNAMIC — carregar preditor dinâmico (GRU)
        self.dynamic_predictor = DynamicPredictor()                        # 🆕 DYNAMIC
        self.last_features_dynamic = None                                  # 🆕 DYNAMIC
        self.dynamic_pred_text = ""                                        # 🆕 DYNAMIC
        self.use_dynamic = False                                           # 🆕 DYNAMIC

        # camera
        api = cv2.CAP_DSHOW if USE_CAP_DSHOW else 0
        self.cap = cv2.VideoCapture(CAM_INDEX, api)
        if not self.cap.isOpened():
            raise RuntimeError("erro ao abrir a camera")

        # mediapipe
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # game state
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
        self.ref_cache = {}
        self.ref_img_tk = None
        self.flash_overlay = None
        self.flash_active = False

        self._build_ui()
        self.after(15, self.update_loop)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ─────────────────────────────────────────
    # SOUNDS
    # ─────────────────────────────────────────
    def _load_sounds(self):
        if os.path.exists(SOUND_LETTER_PATH):
            self.sound_letter = pygame.mixer.Sound(SOUND_LETTER_PATH)
        if os.path.exists(SOUND_WORD_PATH):
            self.sound_word = pygame.mixer.Sound(SOUND_WORD_PATH)

    def _play_letter_sound(self):
        if self.sound_letter:
            try: self.sound_letter.play()
            except Exception as e: print(f"Sound error: {e}")

    def _play_word_sound(self):
        if self.sound_word:
            try: self.sound_word.play()
            except Exception as e: print(f"Sound error: {e}")

    # ─────────────────────────────────────────
    # FLASH FEEDBACK
    # ─────────────────────────────────────────
    def _flash_screen(self, times=1):
        if self.flash_active:
            return
        self.flash_active = True
        self._do_flash(times, times)

    def _do_flash(self, remaining, total):
        if remaining <= 0:
            self.flash_active = False
            return
        self._show_flash_overlay()
        self.after(FLASH_DURATION_MS, self._hide_flash_overlay)
        self.after(FLASH_DURATION_MS * 2, lambda: self._do_flash(remaining - 1, total))

    def _show_flash_overlay(self):
        if self.flash_overlay:
            self.flash_overlay.place(x=0, y=0, relwidth=1, relheight=1)

    def _hide_flash_overlay(self):
        if self.flash_overlay:
            self.flash_overlay.place_forget()

    def _on_correct_letter(self):
        self._play_letter_sound()
        self._flash_screen(times=1)

    def _on_correct_word(self):
        self._play_word_sound()
        self._flash_screen(times=2)

    # ─────────────────────────────────────────
    # BUILD UI
    # ─────────────────────────────────────────
    def _build_ui(self):
        # ── ROOT GRID ──
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── LEFT PANEL: Video feed ──
        left = tk.Frame(self, bg=BG_DEEP)
        left.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        # video header
        vid_header = tk.Frame(left, bg=BG_DEEP)
        vid_header.pack(fill="x", pady=(0, 6))
        tk.Label(vid_header, text="CÂMERA AO VIVO", fg=TEXT_DIM, bg=BG_DEEP,
                 font=("Consolas", 9, "bold")).pack(side="left")
        self._cam_dot = tk.Canvas(vid_header, width=10, height=10, bg=BG_DEEP,
                                   highlightthickness=0)
        self._cam_dot.pack(side="left", padx=(6, 0))
        self._cam_dot.create_oval(1, 1, 9, 9, fill=SUCCESS, outline="")

        # video container with overlay
        self.video_container = tk.Frame(left, bg="#000000",
                                         highlightbackground=BG_BORDER,
                                         highlightthickness=1)
        self.video_container.pack()

        self.video_label = tk.Label(self.video_container, bg="#000000")
        self.video_label.pack()

        self.flash_overlay = tk.Frame(self.video_container, bg=FLASH_COLOR)

        # hand hint
        tk.Label(left, text="⬅  Mão esquerda · palma voltada para a câmera",
                 fg=TEXT_DIM, bg=BG_DEEP, font=("Consolas", 8)).pack(pady=(8, 0))

        # ── RIGHT PANEL: Controls ──
        right = tk.Frame(self, bg=BG_PANEL,
                          highlightbackground=BG_BORDER, highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        right.grid_columnconfigure(0, weight=1)

        inner = tk.Frame(right, bg=BG_PANEL)
        inner.pack(fill="both", expand=True, padx=20, pady=18)

        # ── BRAND ──
        brand = tk.Frame(inner, bg=BG_PANEL)
        brand.pack(fill="x", pady=(0, 2))
        tk.Label(brand, text="LibrAI", fg=ACCENT, bg=BG_PANEL,
                 font=("Consolas", 22, "bold")).pack(side="left")
        tk.Label(brand, text=" Challenge", fg=TEXT_PRI, bg=BG_PANEL,
                 font=("Consolas", 22)).pack(side="left")
        tk.Label(inner,
                 text="Faça a letra solicitada em LIBRAS antes do tempo acabar.",
                 fg=TEXT_SEC, bg=BG_PANEL, font=("Consolas", 9),
                 wraplength=360, justify="left").pack(anchor="w")

        SeparatorLine(inner, width=380).pack(fill="x", pady=12)

        # ── SCORE + MODE ROW ──
        top_row = tk.Frame(inner, bg=BG_PANEL)
        top_row.pack(fill="x", pady=(0, 10))

        # Score card
        score_card = tk.Frame(top_row, bg=BG_CARD,
                               highlightbackground=BG_BORDER, highlightthickness=1)
        score_card.pack(side="left")
        tk.Label(score_card, text="SCORE", fg=TEXT_DIM, bg=BG_CARD,
                 font=("Consolas", 8, "bold"), padx=12, pady=4).pack()
        self.score_var = tk.StringVar(value="0")
        tk.Label(score_card, textvariable=self.score_var, fg=ACCENT, bg=BG_CARD,
                 font=("Consolas", 26, "bold"), padx=12).pack()

        # Mode toggle
        mode_card = tk.Frame(top_row, bg=BG_PANEL)
        mode_card.pack(side="right")
        tk.Label(mode_card, text="MODO", fg=TEXT_DIM, bg=BG_PANEL,
                 font=("Consolas", 8, "bold")).pack(anchor="e")
        toggle_row = tk.Frame(mode_card, bg=BG_PANEL)
        toggle_row.pack(pady=(4, 0))

        self._easy_toggle = ToggleButton(toggle_row, "EASY",
                                          command=self.set_easy_mode, active=True, width=80)
        self._easy_toggle.pack(side="left", padx=(0, 6))
        self._hard_toggle = ToggleButton(toggle_row, "HARD",
                                          command=self.set_hard_mode, active=False, width=80)
        self._hard_toggle.pack(side="left")

        # internal mode_var for compatibility
        self.mode_var = tk.StringVar(value="Mode: EASY")

        SeparatorLine(inner, width=380).pack(fill="x", pady=(4, 12))

        # ── TARGET DISPLAY ──
        target_card = tk.Frame(inner, bg=BG_CARD,
                                highlightbackground=BG_BORDER, highlightthickness=1)
        target_card.pack(fill="x", pady=(0, 10))

        target_inner = tk.Frame(target_card, bg=BG_CARD)
        target_inner.pack(fill="x", padx=14, pady=10)

        tk.Label(target_inner, text="ALVO ATUAL", fg=TEXT_DIM, bg=BG_CARD,
                 font=("Consolas", 8, "bold")).pack(anchor="w")
        self.target_var = tk.StringVar(value="—")
        tk.Label(target_inner, textvariable=self.target_var, fg=TEXT_PRI, bg=BG_CARD,
                 font=("Consolas", 13), wraplength=340, justify="left").pack(anchor="w", pady=(4, 0))

        # Detected row
        det_row = tk.Frame(target_inner, bg=BG_CARD)
        det_row.pack(fill="x", pady=(8, 0))
        tk.Label(det_row, text="DETECTADO:", fg=TEXT_DIM, bg=BG_CARD,
                 font=("Consolas", 8, "bold")).pack(side="left")
        self.pred_var = tk.StringVar(value="—")
        tk.Label(det_row, textvariable=self.pred_var, fg=ACCENT2, bg=BG_CARD,
                 font=("Consolas", 10)).pack(side="left", padx=(8, 0))

        # 🆕 DYNAMIC — linha de info do pipeline dinâmico
        dyn_row = tk.Frame(target_inner, bg=BG_CARD)                      # 🆕 DYNAMIC
        dyn_row.pack(fill="x", pady=(6, 0))                               # 🆕 DYNAMIC
        tk.Label(dyn_row, text="DINÂMICO:", fg=TEXT_DIM, bg=BG_CARD,      # 🆕 DYNAMIC
                 font=("Consolas", 8, "bold")).pack(side="left")           # 🆕 DYNAMIC
        self.dynamic_var = tk.StringVar(value="—")                         # 🆕 DYNAMIC
        tk.Label(dyn_row, textvariable=self.dynamic_var, fg=SUCCESS,       # 🆕 DYNAMIC
                 bg=BG_CARD, font=("Consolas", 9)).pack(                   # 🆕 DYNAMIC
                     side="left", padx=(8, 0))                             # 🆕 DYNAMIC

        # ── TIMER ──
        timer_section = tk.Frame(inner, bg=BG_PANEL)
        timer_section.pack(fill="x", pady=(0, 10))

        timer_header = tk.Frame(timer_section, bg=BG_PANEL)
        timer_header.pack(fill="x")
        tk.Label(timer_header, text="TEMPO", fg=TEXT_DIM, bg=BG_PANEL,
                 font=("Consolas", 8, "bold")).pack(side="left")

        self.timer_canvas = tk.Canvas(timer_section, width=380, height=10,
                                       bg=TIMER_BG, highlightthickness=0)
        self.timer_canvas.pack(fill="x", pady=(5, 0))
        self.timer_rect = self.timer_canvas.create_rectangle(
            0, 0, 380, 10, fill=SUCCESS, width=0)

        # ── STATUS ──
        status_card = tk.Frame(inner, bg=BG_CARD,
                                highlightbackground=BG_BORDER, highlightthickness=1)
        status_card.pack(fill="x", pady=(0, 12))
        self.status_var = tk.StringVar(value="Pressione Start para jogar.")
        tk.Label(status_card, textvariable=self.status_var, fg=TEXT_PRI, bg=BG_CARD,
                 font=("Consolas", 10), wraplength=340, justify="left",
                 padx=14, pady=8).pack(anchor="w")

        # ── CONTROL BUTTONS ──
        btn_row = tk.Frame(inner, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(0, 10))

        self.start_btn = DarkButton(btn_row, "▶  START", command=self.start_game,
                                     accent=True, width=120)
        self.start_btn.pack(side="left", padx=(0, 6))

        self.reset_btn = DarkButton(btn_row, "↺  RESET", command=self.reset_game, width=110)
        self.reset_btn.pack(side="left", padx=(0, 6))

        self.quit_btn = DarkButton(btn_row, "✕  QUIT", command=self.on_close,
                                    danger=True, width=100)
        self.quit_btn.pack(side="left")

        SeparatorLine(inner, width=380).pack(fill="x", pady=(2, 12))

        # ── HARD WORD ENTRY ──
        word_section = tk.Frame(inner, bg=BG_PANEL)
        word_section.pack(fill="x", pady=(0, 10))

        tk.Label(word_section, text="PALAVRA (MODO HARD)", fg=TEXT_DIM, bg=BG_PANEL,
                 font=("Consolas", 8, "bold")).pack(anchor="w", pady=(0, 5))

        word_entry_row = tk.Frame(word_section, bg=BG_PANEL)
        word_entry_row.pack(fill="x")

        # styled entry frame
        entry_frame = tk.Frame(word_entry_row, bg=BG_CARD,
                                highlightbackground=BG_BORDER, highlightthickness=1)
        entry_frame.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.word_entry = tk.Entry(entry_frame, bg=BG_CARD, fg=TEXT_PRI,
                                    insertbackground=ACCENT, relief="flat",
                                    font=("Consolas", 11), bd=4)
        self.word_entry.pack(fill="x")
        self.word_entry.insert(0, self.target_word)

        self.set_word_btn = DarkButton(word_entry_row, "SET", command=self.set_hard_word,
                                        width=70, height=34)
        self.set_word_btn.pack(side="left")

        # ── REFERENCE IMAGE (EASY mode) ──
        SeparatorLine(inner, width=380).pack(fill="x", pady=(4, 12))

        ref_header = tk.Frame(inner, bg=BG_PANEL)
        ref_header.pack(fill="x", pady=(0, 6))
        tk.Label(ref_header, text="REFERÊNCIA", fg=TEXT_DIM, bg=BG_PANEL,
                 font=("Consolas", 8, "bold")).pack(side="left")
        tk.Label(ref_header, text="(modo EASY)", fg=TEXT_DIM, bg=BG_PANEL,
                 font=("Consolas", 8)).pack(side="left", padx=(6, 0))

        self.ref_container = tk.Frame(inner, bg=BG_CARD,
                                       highlightbackground=BG_BORDER, highlightthickness=1)
        self.ref_container.pack(fill="x")

        self.ref_img_label = tk.Label(self.ref_container, bg=BG_CARD)
        self.ref_img_label.pack(pady=(8, 4))

        self.ref_text_label = tk.Label(self.ref_container, text="",
                                        fg=TEXT_SEC, bg=BG_CARD,
                                        font=("Consolas", 9), pady=6)
        self.ref_text_label.pack()

        self._update_reference_image()

    # ─────────────────────────────────────────
    # MODE MANAGEMENT
    # ─────────────────────────────────────────
    def set_easy_mode(self):
        self.mode = "EASY"
        self.mode_var.set("Mode: EASY")
        self._easy_toggle.set_active(True)
        self._hard_toggle.set_active(False)
        self.status_var.set("Modo EASY ativado — referência visível.")
        self.history.clear()
        self._update_reference_image()
        self._refresh_target_text()

    def set_hard_mode(self):
        self.mode = "HARD"
        self.mode_var.set("Mode: HARD")
        self._easy_toggle.set_active(False)
        self._hard_toggle.set_active(True)
        self.status_var.set("Modo HARD — soletrar palavras, sem referência.")
        self.history.clear()
        self._update_reference_image()
        self._refresh_target_text()

    def set_hard_word(self):
        word = self.word_entry.get().strip().upper()
        word = "".join([c for c in word if c.isalpha()])
        if not word:
            self.status_var.set("⚠  Digite uma palavra válida (A-Z).")
            return
        self.target_word = word
        self.hard_pos = 0
        self.status_var.set(f"Palavra definida: {self.target_word}")
        self._refresh_target_text()

    def _pick_new_word(self):
        if not HARD_WORDS:
            return "DIOGO"
        if len(HARD_WORDS) == 1:
            return HARD_WORDS[0]
        candidates = [w for w in HARD_WORDS if w != self.target_word]
        return random.choice(candidates) if candidates else random.choice(HARD_WORDS)

    # ─────────────────────────────────────────
    # GAME CONTROL
    # ─────────────────────────────────────────
    def start_game(self):
        if self.running:
            return
        self.running = True
        self.game_over = False
        self.score = 0
        self.score_var.set("0")
        self.history.clear()
        self.hard_pos = 0
        if self.dynamic_predictor.loaded:                                  # 🆕 DYNAMIC
            self.dynamic_predictor.clear()                                 # 🆕 DYNAMIC
        self._next_round()

    def reset_game(self):
        self.running = False
        self.game_over = False
        self.score = 0
        self.round_start = None
        self.target_letter = None
        self.hard_pos = 0
        self.history.clear()
        if self.dynamic_predictor.loaded:                                  # 🆕 DYNAMIC
            self.dynamic_predictor.clear()                                 # 🆕 DYNAMIC
        self.dynamic_var.set("—")                                          # 🆕 DYNAMIC
        self.score_var.set("0")
        self.pred_var.set("—")
        self.status_var.set("Pressione Start para jogar.")
        self._set_timer_ratio(1.0)
        self._refresh_target_text()
        self._update_reference_image()

    def end_game(self):
        self.running = False
        self.game_over = True
        self.status_var.set(f"GAME OVER  ·  Score final: {self.score}  ·  Pressione Reset.")
        self._set_timer_ratio(0.0)

    def _get_round_seconds(self):
        return ROUND_SECONDS_EASY if self.mode == "EASY" else ROUND_SECONDS_HARD

    def _current_target(self):
        if self.mode == "EASY":
            return self.target_letter
        if self.hard_pos >= len(self.target_word):
            return None
        return self.target_word[self.hard_pos]

    def _next_round(self):
        self.round_start = time.time()
        self.history.clear()
        self.last_next_time = time.time()
        if self.mode == "EASY":
            self.target_letter = random.choice(self.allowed)
        self.status_var.set("GO!")
        self._refresh_target_text()
        self._update_reference_image()

    # ─────────────────────────────────────────
    # TARGET TEXT
    # ─────────────────────────────────────────
    def _refresh_target_text(self):
        if self.mode == "EASY":
            t = self.target_letter if self.target_letter else "—"
            self.target_var.set(f"Letra:  {t}")
            return
        if self.target_word:
            if self.hard_pos >= len(self.target_word):
                self.target_var.set(f"{self.target_word}  ✅  completa!")
            else:
                done_part = self.target_word[:self.hard_pos]
                remaining = "_" * (len(self.target_word) - self.hard_pos)
                current = self.target_word[self.hard_pos]
                self.target_var.set(
                    f"Palavra: {self.target_word}\n"
                    f"Progresso: {done_part}{remaining}   →  agora: {current}"
                )
        else:
            self.target_var.set("—")

    # ─────────────────────────────────────────
    # REFERENCE IMAGE
    # ─────────────────────────────────────────
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
        if self.mode != "EASY":
            self.ref_img_label.configure(image="")
            self.ref_text_label.configure(text="Sem referência no modo HARD.", fg=TEXT_DIM)
            self.ref_img_tk = None
            return

        letter = self.target_letter
        if not letter:
            self.ref_img_label.configure(image="")
            self.ref_text_label.configure(text="Pressione Start para iniciar.", fg=TEXT_DIM)
            self.ref_img_tk = None
            return

        img = self._load_ref_image(letter)
        ref_fit = fit_image_to_box(img, REF_BOX_W, REF_BOX_H)
        if ref_fit is None:
            self.ref_img_label.configure(image="")
            self.ref_text_label.configure(text=f"Referência não encontrada: {letter}", fg=DANGER)
            self.ref_img_tk = None
            return

        ref_rgb = cv2.cvtColor(ref_fit, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(ref_rgb)
        self.ref_img_tk = ImageTk.PhotoImage(pil_img)
        self.ref_img_label.configure(image=self.ref_img_tk)
        self.ref_text_label.configure(text=f"Faça a letra:  {letter}", fg=ACCENT)

    # ─────────────────────────────────────────
    # TIMER BAR
    # ─────────────────────────────────────────
    def _set_timer_ratio(self, ratio):
        ratio = max(0.0, min(1.0, ratio))
        width = int(380 * ratio)
        self.timer_canvas.coords(self.timer_rect, 0, 0, width, 10)
        if ratio > 0.5:
            color = SUCCESS
        elif ratio > 0.2:
            color = WARNING
        else:
            color = DANGER
        self.timer_canvas.itemconfig(self.timer_rect, fill=color)

    # ═══════════════════════════════════════════════════════════════
    # 🆕 DYNAMIC — detectar movimento da mão entre frames
    # ═══════════════════════════════════════════════════════════════
    def _detect_movement(self, current_features):                          # 🆕 DYNAMIC
        """
        Compara features atuais com as do frame anterior.
        Retorna magnitude do movimento.
        """
        if self.last_features_dynamic is None:
            self.last_features_dynamic = current_features
            return 0.0

        movement = np.linalg.norm(current_features - self.last_features_dynamic)
        self.last_features_dynamic = current_features
        return movement

    # ─────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────
    def update_loop(self):
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
                hand_landmarks = result.hand_landmarks[0]
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 128), -1)

                # ── Pipeline ESTÁTICO (inalterado) ──
                feats = extract_features(hand_landmarks).reshape(1, -1)
                predicted = None
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(feats)[0]
                    idx = int(np.argmax(proba))
                    predicted = str(self.model.classes_[idx])
                    conf = float(proba[idx])
                else:
                    predicted = str(self.model.predict(feats)[0])

                if predicted in self.allowed:
                    self.history.append(predicted)

                voted, ratio = majority_vote(self.history)

                # ═══════════════════════════════════════════════════════
                # 🆕 DYNAMIC — pipeline paralelo
                # ═══════════════════════════════════════════════════════
                if self.dynamic_predictor.loaded:                          # 🆕 DYNAMIC
                    # Extrair features dinâmicas (63-dim com z)
                    dyn_feats = extract_dynamic_features(hand_landmarks)    # 🆕 DYNAMIC

                    # Detectar movimento
                    movement = self._detect_movement(dyn_feats)            # 🆕 DYNAMIC
                    self.use_dynamic = movement > MOVEMENT_THRESHOLD       # 🆕 DYNAMIC

                    # Alimentar buffer do GRU sempre
                    self.dynamic_predictor.feed(hand_landmarks)            # 🆕 DYNAMIC

                    # Predizer quando buffer está cheio
                    dyn_result = self.dynamic_predictor.predict()           # 🆕 DYNAMIC
                    if dyn_result:                                         # 🆕 DYNAMIC
                        dyn_cls, dyn_conf = dyn_result                    # 🆕 DYNAMIC
                        self.dynamic_pred_text = (                         # 🆕 DYNAMIC
                            f"{dyn_cls} ({dyn_conf*100:.0f}%)")           # 🆕 DYNAMIC

                        # Se detectou movimento E GRU prediz letra dinâmica,
                        # sobrescrever a predição estática
                        if (self.use_dynamic and                           # 🆕 DYNAMIC
                                dyn_cls in DYNAMIC_LETTERS_SET):           # 🆕 DYNAMIC
                            voted = dyn_cls                                # 🆕 DYNAMIC
                            conf = dyn_conf                                # 🆕 DYNAMIC
                            ratio = 1.0                                    # 🆕 DYNAMIC
                            self.history.clear()                           # 🆕 DYNAMIC
                            self.history.append(dyn_cls)                   # 🆕 DYNAMIC
                    else:                                                  # 🆕 DYNAMIC
                        prog = self.dynamic_predictor.buffer_progress()    # 🆕 DYNAMIC
                        self.dynamic_pred_text = (                         # 🆕 DYNAMIC
                            f"buffer {prog*100:.0f}%")                    # 🆕 DYNAMIC

                    # Atualizar label dinâmico na UI
                    pipe = "DYN" if self.use_dynamic else "STA"            # 🆕 DYNAMIC
                    self.dynamic_var.set(                                   # 🆕 DYNAMIC
                        f"[{pipe}] mov={movement:.2f} · "                  # 🆕 DYNAMIC
                        f"{self.dynamic_pred_text}")                       # 🆕 DYNAMIC
                # ═══════════════════════════════════════════════════════

            else:
                if len(self.history) > 0:
                    self.history.popleft()
                self.last_features_dynamic = None                          # 🆕 DYNAMIC

            # ── GAME LOGIC (inalterada) ──
            if self.running and not self.game_over:
                elapsed = time.time() - self.round_start
                round_seconds = self._get_round_seconds()
                ratio_left = 1.0 - (elapsed / round_seconds)
                self._set_timer_ratio(ratio_left)

                if voted is None:
                    self.pred_var.set("—")
                else:
                    if conf is not None:
                        self.pred_var.set(f"{voted}  (estável {ratio*100:.0f}%  conf {conf*100:.0f}%)")
                    else:
                        self.pred_var.set(f"{voted}  (estável {ratio*100:.0f}%)")

                if elapsed >= round_seconds:
                    self.end_game()
                else:
                    target = self._current_target()
                    stable_ok = (ratio >= STABLE_RATIO)
                    conf_ok = (conf is None) or (conf >= MIN_CONF)

                    if target is not None and voted == target and stable_ok and conf_ok:
                        if time.time() - self.last_next_time >= COOLDOWN_NEXT:
                            self.score += 1
                            self.score_var.set(str(self.score))

                            if self.mode == "HARD":
                                self.hard_pos += 1
                                if self.hard_pos >= len(self.target_word):
                                    self._on_correct_word()
                                    self.score += 2
                                    self.score_var.set(str(self.score))
                                    self.target_word = self._pick_new_word()
                                    self.hard_pos = 0
                                    self.status_var.set(
                                        f"✅ Palavra completa!  +2 bônus  →  Próxima: {self.target_word}")
                                else:
                                    self._on_correct_letter()
                                    self.status_var.set(
                                        f"✅ Correto!  Próxima letra: {self.target_word[self.hard_pos]}")
                            else:
                                self._on_correct_letter()
                                self.status_var.set("✅ Correto!  Próxima letra...")

                            self._next_round()
            else:
                if not self.game_over:
                    self._set_timer_ratio(1.0)

            self._show_frame(frame)

        self.after(20, self.update_loop)

    def _show_frame(self, frame_bgr):
        frame = cv2.resize(frame_bgr, (660, 495), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def on_close(self):
        try: pygame.mixer.quit()
        except Exception: pass
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception: pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception: pass
        self.destroy()


if __name__ == "__main__":
    app = LibrAIApp()
    app.mainloop()
