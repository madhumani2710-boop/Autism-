import cv2
import numpy as np
import time
import threading
import os
import math
import json
import csv
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

import pygame

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_pose  = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face  = mp.solutions.face_mesh
    mp_draw  = mp.solutions.drawing_utils
except ImportError:
    MP_AVAILABLE = False
    print("[WARN] mediapipe not found -> pip install mediapipe")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not found -> pip install pyserial")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[WARN] SpeechRecognition not found -> pip install SpeechRecognition pyaudio")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests not found -> pip install requests")

CHILD_NAME       = "friend"
AUDIO_FOLDER     = "audio_files"
LISTEN_SECS      = 10
FAST_THRESHOLD   = 3.0
REPORT_FOLDER    = "reports"

ESP32_CAM_URL    = "http://192.168.137.58/stream"   
SERIAL_PORT      = "COM6"                            
SERIAL_BAUD      = 115200

AUDIO_GREETING   = "001"
AUDIO_PRAISE     = "004"
AUDIO_ENCOURAGE  = "003"

# Fixed order: 9 -> 10 -> 11 -> 13 -> 15 -> 16
QUESTION_ORDER   = ["009", "010", "011", "013", "015", "016"]

C = {
    "bg":      (15,  15,  25),
    "panel":   (25,  28,  45),
    "accent":  (0,  220, 180),
    "warn":    (0,  160, 255),
    "danger":  (50,  50, 255),
    "success": (50, 220, 100),
    "text":    (230,230, 240),
    "muted":   (130,130, 150),
    "gold":    (30, 200, 255),
    "praise":  (0,  255, 150),
    "slow":    (0,  120, 255),
    "error":   (0,   0,  200),
}

@dataclass
class QuestionScore:
    q_index:            int   = 0
    audio_file:         str   = ""
    voice_text:         str   = ""
    voice_detected:     bool  = False
    response_time_secs: float = 0.0
    was_fast:           bool  = False
    reactive_audio:     str   = ""
    keyword_score:      float = 0.0
    clarity_score:      float = 0.0
    tone_score:         float = 0.0
    eye_contact_score:  float = 0.0
    hand_pattern_score: float = 0.0
    repetition_score:   float = 0.0
    gaze_score:         float = 0.0
    total_score:        float = 0.0

    def compute_total(self):
        rt_score = max(0.0, 100.0 - (self.response_time_secs / 8.0) * 100)
        weights = {
            "rt":         (rt_score,                   0.12),
            "keyword":    (self.keyword_score,         0.13),
            "clarity":    (self.clarity_score,         0.10),
            "tone":       (self.tone_score,            0.10),
            "eye":        (self.eye_contact_score,     0.15),
            "hand":       (self.hand_pattern_score,    0.10),
            "rep":        (self.repetition_score,      0.10),
            "gaze":       (self.gaze_score,            0.10),
            "fast_bonus": (100.0 if self.was_fast else 0.0, 0.10),
        }
        self.total_score = round(sum(v * w for v, w in weights.values()), 1)
        return self.total_score


@dataclass
class SmoothWindow:
    _buf: deque = field(default_factory=lambda: deque(maxlen=30))

    def push(self, v: float):
        self._buf.append(float(v))

    def mean(self) -> float:
        return sum(self._buf) / max(len(self._buf), 1)

class AudioEngine:
    def __init__(self):
        pygame.mixer.init()

    def play(self, name: str) -> bool:
        path = os.path.join(AUDIO_FOLDER, f"{name}.wav")
        if not os.path.exists(path):
            print(f"[Audio] NOT FOUND: {path}")
            return False
        print(f"[Audio] Playing: {name}.wav")
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

        time.sleep(0.3)
        return True

class ESP32CamCapture:
    def __init__(self, url):
        self.url       = url
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = False
        self.connected = False

    def open(self) -> bool:
        if not REQUESTS_AVAILABLE:
            print("[Camera] ERROR: 'requests' not installed")
            return False

        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

        print(f"[Camera] Connecting to ESP32-CAM at {self.url} ...")
        deadline = time.time() + 10
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    self.connected = True
                    print("[Camera] ESP32-CAM connected OK")
                    return True
            time.sleep(0.2)

        self._running = False
        print("[Camera] ERROR: ESP32-CAM not reachable")
        print(f"  -> Is camera powered on?")
        print(f"  -> Is this IP correct? {self.url}")
        return False

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, np.zeros((480, 640, 3), dtype=np.uint8)
            return True, self._frame.copy()

    def release(self):
        self._running = False

    def _loop(self):
        SOI, EOI = b'\xff\xd8', b'\xff\xd9'
        while self._running:
            try:
                resp = requests.get(self.url, stream=True, timeout=(8, None))
                buf  = b""
                for chunk in resp.iter_content(8192):
                    if not self._running:
                        break
                    buf += chunk
                    while True:
                        s = buf.find(SOI)
                        if s == -1: buf = b""; break
                        e = buf.find(EOI, s)
                        if e == -1: buf = buf[s:]; break
                        img = cv2.imdecode(
                            np.frombuffer(buf[s:e+2], np.uint8),
                            cv2.IMREAD_COLOR)
                        buf = buf[e+2:]
                        if img is not None:
                            with self._lock:
                                self._frame = img
            except Exception as ex:
                if self._running:
                    print(f"[Camera] Lost: {ex} -- reconnecting...")
                    time.sleep(2)

class SerialMicReader:
    SILENCE     = 2048
    NOISE_FLOOR = 150

    def __init__(self, port, baud=115200):
        self.port      = port
        self.baud      = baud
        self._ser      = None
        self._lock     = threading.Lock()
        self._vals: deque = deque(maxlen=200)
        self._running  = False
        self._onset: Optional[float] = None
        self.connected = False

    def open(self) -> bool:
        if not SERIAL_AVAILABLE:
            print("[Mic] ERROR: 'pyserial' not installed")
            return False
        try:
            self._ser      = serial.Serial(self.port, self.baud, timeout=1)
            self._running  = True
            self.connected = True
            threading.Thread(target=self._loop, daemon=True).start()
            print(f"[Mic] Serial connected: {self.port} @ {self.baud} OK")
            return True
        except serial.SerialException as ex:
            print(f"[Mic] ERROR: Cannot open {self.port}")
            print(f"  -> {ex}")
            print(f"  -> Is ESP32 plugged in via USB?")
            print(f"  -> Check COM port in Device Manager")
            return False

    def close(self):
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _loop(self):
        while self._running:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if line.isdigit():
                    amp = abs(int(line) - self.SILENCE)
                    with self._lock:
                        self._vals.append(amp)
                        if amp > self.NOISE_FLOOR and self._onset is None:
                            self._onset = time.time()
            except:
                time.sleep(0.01)

    def reset(self):
        with self._lock:
            self._onset = None
            self._vals.clear()

    def onset_time(self) -> Optional[float]:
        with self._lock:
            return self._onset

    def stats(self) -> dict:
        with self._lock:
            vals = list(self._vals)
        if not vals:
            return {"mean": 0, "peak": 0, "std": 0, "active": False}
        m = sum(vals) / len(vals)
        return {
            "mean":   m,
            "peak":   max(vals),
            "std":    math.sqrt(sum((v - m)**2 for v in vals) / len(vals)),
            "active": max(vals) > self.NOISE_FLOOR,
        }

def show_hardware_error(errors: list):
    w, h = 820, 500
    while True:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (18, 8, 8)

        cv2.putText(img, "HARDWARE ERROR -- CANNOT START",
                    (w//2-270, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 220), 2)
        cv2.putText(img, "Fix the issues below then restart the program.",
                    (w//2-260, 84), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (160, 160, 160), 1)
        cv2.line(img, (40, 100), (w-40, 100), (50, 50, 50), 1)

        for i, (device, problem, fix) in enumerate(errors):
            y = 130 + i * 115
            cv2.rectangle(img, (38, y), (240, y+26), (60, 15, 15), -1)
            cv2.putText(img, device, (44, y+19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 80, 255), 2)
            cv2.putText(img, f"Problem : {problem}",
                        (38, y+52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (100, 100, 255), 1)
            cv2.putText(img, f"Fix     : {fix}",
                        (38, y+78), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (50, 210, 90), 1)
            cv2.line(img, (38, y+100), (w-38, y+100), (40, 40, 40), 1)

        cv2.putText(img, "Press Q to close",
                    (w//2-80, h-22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (90, 90, 90), 1)

        cv2.imshow("Autism Toy -- Hardware Error", img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    sys.exit(1)

class SpeechEngine:
    def __init__(self):
        self.rec = self.mic = None
        if SR_AVAILABLE:
            try:
                self.rec = sr.Recognizer()
                self.mic = sr.Microphone()
                with self.mic as s:
                    self.rec.adjust_for_ambient_noise(s, duration=0.5)
                print("[STT] SpeechRecognition ready OK")
            except Exception as ex:
                print(f"[STT] Init error: {ex}")

    def listen(self, timeout=8) -> Optional[str]:
        if not (self.rec and self.mic):
            return None
        try:
            with self.mic as s:
                audio = self.rec.listen(s, timeout=timeout,
                                        phrase_time_limit=timeout)
            text = self.rec.recognize_google(audio)
            print(f"[STT] Recognized: '{text}'")
            return text
        except:
            return None

class PoseGazeAnalyzer:
    def __init__(self):
        if not MP_AVAILABLE:
            self.pose = self.face = self.hands_mp = None
            return
        self.pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6)
        self.face = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.hands_mp = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6)
        self._wrist_hist: deque = deque(maxlen=60)
        self._gaze_hist:  deque = deque(maxlen=30)

    def process(self, bgr):
        if not MP_AVAILABLE:
            return None, None, None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return (self.pose.process(rgb),
                self.face.process(rgb),
                self.hands_mp.process(rgb))

    def score_eye_contact(self, face_r) -> float:
        if not (face_r and face_r.multi_face_landmarks):
            return 0.0
        lm = face_r.multi_face_landmarks[0].landmark
        try:
            li, ri = lm[468], lm[473]
            nose   = lm[1]
            if abs(nose.x - 0.5) < 0.2 and li.z < 0.1 and ri.z < 0.1:
                dev   = math.sqrt(((li.x+ri.x)/2-0.5)**2+((li.y+ri.y)/2-0.5)**2)
                score = max(0.0, 1.0 - dev * 8)
                self._gaze_hist.append(score)
                return round(score * 100, 1)
        except IndexError:
            pass
        return 0.0

    def score_hand_patterns(self, pose_r) -> dict:
        if not (pose_r and pose_r.pose_landmarks):
            return {"hand_score": 0.0, "repetition_score": 50.0}
        lm = pose_r.pose_landmarks.landmark
        L  = mp_pose.PoseLandmark
        def xy(i): p = lm[i]; return np.array([p.x, p.y])
        lw, rw = xy(L.LEFT_WRIST),    xy(L.RIGHT_WRIST)
        ls, rs = xy(L.LEFT_SHOULDER), xy(L.RIGHT_SHOULDER)
        bw     = abs(ls[0] - rs[0]) + 1e-6
        lr, rr_ = (lw - ls) / bw, (rw - rs) / bw
        sym    = (1 - min(1, abs(lr[0]+rr_[0])))*0.6 + \
                 (1 - min(1, abs(lr[1]-rr_[1])))*0.4
        elev   = (int(lw[1] < lm[L.NOSE].y) + int(rw[1] < lm[L.NOSE].y)) / 2
        hs     = (sym * 0.5 + elev * 0.5) * 100
        self._wrist_hist.append((lw + rw) / 2)
        rep = 50.0
        if len(self._wrist_hist) >= 10:
            h   = list(self._wrist_hist)
            ds  = [np.linalg.norm(np.array(h[i]) - np.array(h[i-1]))
                   for i in range(1, len(h))]
            md  = sum(ds) / len(ds)
            std = math.sqrt(sum((d - md)**2 for d in ds) / len(ds))
            rep = 80.0 if md < 0.005 else (30.0 if std/(md+1e-6) < 0.3 else 70.0)
        return {"hand_score": round(hs, 1), "repetition_score": round(rep, 1)}

    def draw(self, frame, pose_r, face_r, hands_r):
        if not MP_AVAILABLE:
            return frame
        if pose_r and pose_r.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, pose_r.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,220,180), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(100,200,255), thickness=2))
        if hands_r and hands_r.multi_hand_landmarks:
            for hl in hands_r.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255,200,0), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(200,150,0), thickness=2))
        if face_r and face_r.multi_face_landmarks:
            for fl in face_r.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, fl, mp_face.FACEMESH_IRISES,
                    mp_draw.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=(0,200,200), thickness=1))
        return frame

class ScoringEngine:
    def compute(self, text, amp) -> dict:
        clarity = 0.0
        if amp["mean"] > 0:
            clarity = (1 - min(1, amp["std"] / (amp["mean"] + 1e-6))) * 100
        std  = amp["std"]
        tone = 40.0 if std < 50 else (80.0 if std < 200 else 55.0)

        if text and len(text.strip()) > 0:
            words = text.strip().split()
            kw = min(100.0, 40.0 + len(words) * 6.0)
        else:
            kw = 0.0

        return {
            "clarity_score": round(clarity, 1),
            "tone_score":    round(tone,    1),
            "keyword_score": round(kw,      1),
        }

class ReportWriter:
    def __init__(self):
        os.makedirs(REPORT_FOLDER, exist_ok=True)

    def save(self, scores: List[QuestionScore]):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(REPORT_FOLDER, f"{CHILD_NAME}_{ts}")
        with open(base + ".json", "w") as f:
            json.dump({"child": CHILD_NAME, "date": ts,
                       "scores": [vars(s) for s in scores]}, f, indent=2)
        if scores:
            with open(base + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=vars(scores[0]).keys())
                w.writeheader()
                for s in scores:
                    w.writerow(vars(s))
        print(f"[Report] Saved: {base}.json + .csv")

class AutismToyApp:

    S_INTRO   = "INTRO"
    S_PLAYING = "PLAYING"
    S_LISTEN  = "LISTENING"
    S_REACT   = "REACTING"
    S_SCORING = "SCORING"
    S_RESULTS = "RESULTS"

    def __init__(self):
        self.audio  = AudioEngine()
        self.speech = SpeechEngine()
        self.pose   = PoseGazeAnalyzer()
        self.scorer = ScoringEngine()
        self.report = ReportWriter()

        errors = []

        self.cam = ESP32CamCapture(ESP32_CAM_URL)
        if not self.cam.open():
            errors.append((
                "ESP32-CAM (WiFi)",
                f"No stream received from {ESP32_CAM_URL}",
                "Power on ESP32-CAM, connect WiFi, verify IP address"
            ))

        self.mic = SerialMicReader(SERIAL_PORT, SERIAL_BAUD)
        self.mic_ok = self.mic.open()
        if not self.mic_ok:
            errors.append((
                f"ESP32 Mic ({SERIAL_PORT})",
                f"Serial port {SERIAL_PORT} could not be opened",
                "Plug ESP32 USB into PC, check COM port in Device Manager"
            ))

        if errors:
            show_hardware_error(errors)

        self.state          = self.S_INTRO
        self.lock           = threading.Lock()
        self.q_scores:      List[QuestionScore] = []
        self.current_q_name = ""
        self.current_q_num  = 0
        self.countdown      = 0
        self.live_text      = ""
        self.live_amp       = {}
        self.reactive_label = ""
        self.reactive_col   = C["success"]
        self.all_eye:  List[float] = []
        self.all_hand: List[float] = []
        self.all_rep:  List[float] = []
        self.sm_eye    = SmoothWindow()
        self.sm_hand   = SmoothWindow()

        print("\n========================================")
        print("  AUTISM TOY -- INTERACTIVE MODE READY")
        print("========================================")
        print(f"  Greeting   : {AUDIO_GREETING}.wav")
        print(f"  Questions  : {' -> '.join(QUESTION_ORDER)}")
        print(f"  Fast(<=3s) : {AUDIO_PRAISE}.wav  (praise)")
        print(f"  Slow/None  : {AUDIO_ENCOURAGE}.wav  (encourage)")
        print(f"  Camera     : ESP32-CAM OK")
        print(f"  Mic        : {SERIAL_PORT} OK")
        print(f"  STT        : {'OK' if SR_AVAILABLE else 'NOT AVAILABLE'}")
        print(f"  MediaPipe  : {'OK' if MP_AVAILABLE else 'NOT AVAILABLE'}")
        print("========================================")
        print("  SPACE = Start  |  Q = Quit  |  R = Restart\n")

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _session(self):
        # ── Step 1: Greeting 
        self._set(state=self.S_PLAYING,
                  current_q_name=AUDIO_GREETING,
                  current_q_num=0,
                  reactive_label="")
        self.audio.play(AUDIO_GREETING)

        time.sleep(1.2)

        # ── Step 2: Questions 009 -> 010 -> 011 -> 013 -> 015 -> 016 ─────
        for q_num, q_audio in enumerate(QUESTION_ORDER, start=1):

            # --- Play question ---
            self._set(state=self.S_PLAYING,
                      current_q_name=q_audio,
                      current_q_num=q_num,
                      live_text="",
                      reactive_label="")
            self.audio.play(q_audio)
            audio_end = time.time()   
            self.mic.reset()         

            with self.lock:
                self.all_eye.clear()
                self.all_hand.clear()
                self.all_rep.clear()
                self.state = self.S_LISTEN   

            recognized = [None]
            stt_done   = threading.Event()

            def _stt():
                recognized[0] = self.speech.listen(timeout=LISTEN_SECS)
                with self.lock:
                    self.live_text = recognized[0] or "[no speech]"
                stt_done.set()

            threading.Thread(target=_stt, daemon=True).start()

            # Countdown ticker
            for remaining in range(LISTEN_SECS, 0, -1):
                with self.lock:
                    self.countdown = remaining
                time.sleep(1)
            self.countdown = 0

        
            STT_API_GRACE = 3.5
            stt_done.wait(timeout=STT_API_GRACE)

            with self.lock:
                snap_eye  = list(self.all_eye)
                snap_hand = list(self.all_hand)
                snap_rep  = list(self.all_rep)

          
            onset = self.mic.onset_time()

            if onset:
                resp_secs = max(0.0, onset - audio_end)
            else:
                resp_secs = float(LISTEN_SECS)

            is_fast = 0.0 < resp_secs <= FAST_THRESHOLD

            react_audio = AUDIO_PRAISE    if is_fast else AUDIO_ENCOURAGE
            react_label = f"FAST  ({resp_secs:.1f}s)" if is_fast \
                          else f"SLOW  ({resp_secs:.1f}s)"
            react_col   = C["praise"] if is_fast else C["slow"]

            self._set(state=self.S_REACT,
                      reactive_label=react_label,
                      reactive_col=react_col)

            print(f"[Q{q_num}] {q_audio}.wav  ->  {react_label}  ->  playing {react_audio}.wav")

            self.audio.play(react_audio)

            self._set(state=self.S_SCORING)
            amp = self.mic.stats()
            vs  = self.scorer.compute(recognized[0], amp)

            avg_eye  = sum(snap_eye)  / max(len(snap_eye),  1)
            avg_hand = sum(snap_hand) / max(len(snap_hand), 1)
            avg_rep  = sum(snap_rep)  / max(len(snap_rep),  1)

            if len(snap_eye) > 1:
                mean_e = avg_eye
                std_e  = math.sqrt(
                    sum((v - mean_e) ** 2 for v in snap_eye) / len(snap_eye))
                gaze_s = round(max(0.0, mean_e - std_e * 0.5), 1)
            else:
                gaze_s = round(avg_eye, 1)

            qs = QuestionScore(
                q_index            = q_num - 1,
                audio_file         = q_audio,
                voice_text         = recognized[0] or "",
                voice_detected     = bool(recognized[0]),
                response_time_secs = resp_secs,
                was_fast           = is_fast,
                reactive_audio     = react_audio,
                keyword_score      = vs["keyword_score"],
                clarity_score      = vs["clarity_score"],
                tone_score         = vs["tone_score"],
                eye_contact_score  = round(avg_eye,  1),
                hand_pattern_score = round(avg_hand, 1),
                repetition_score   = round(avg_rep,  1),
                gaze_score         = gaze_s,
            )
            qs.compute_total()
            with self.lock:
                self.q_scores.append(qs)

            print(f"       Score:{qs.total_score}/100  "
                  f"Eye:{avg_eye:.0f}  Hand:{avg_hand:.0f}  "
                  f"Gaze:{gaze_s:.0f}  "
                  f"STT:'{qs.voice_text[:20]}'")
            time.sleep(0.3)

        self.report.save(self.q_scores)
        self._set(state=self.S_RESULTS)
        self._print_report()

    def _panel(self, img, x1,y1,x2,y2, alpha=0.78):
        sub = img[y1:y2,x1:x2]
        cv2.addWeighted(np.full_like(sub,C["panel"]),alpha,sub,1-alpha,0,sub)
        img[y1:y2,x1:x2] = sub
        cv2.rectangle(img,(x1,y1),(x2,y2),C["accent"],1)

    def _txt(self, img, text, pos, scale=0.65, color=None, thick=2):
        col = color or C["text"]
        cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+2)
        cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,col,thick)

    def _bar(self, img, x,y,w,h,val,col=None):
        col = col or C["accent"]
        cv2.rectangle(img,(x,y),(x+w,y+h),(40,40,60),-1)
        fill = int(w*max(0.0,min(1.0,val)))
        if fill > 0:
            cv2.rectangle(img,(x,y),(x+fill,y+h),col,-1)
        cv2.rectangle(img,(x,y),(x+w,y+h),C["muted"],1)

    def _sc(self, s):
        return C["success"] if s>=70 else (C["warn"] if s>=40 else C["danger"])

    def _draw_intro(self, f):
        h,w = f.shape[:2]
        cv2.addWeighted(np.full_like(f,18),0.55,f,0.45,0,f)
        self._txt(f,"AUTISM INTERACTION TOY",(w//2-230,62),
                  scale=1.0,color=C["accent"],thick=3)
        self._txt(f,f"Hello, {CHILD_NAME}!",(w//2-90,100),
                  scale=0.75,color=C["gold"])
        self._panel(f,w//2-310,118,w//2+310,350)

        lines = [
            (f"Greeting   :  {AUDIO_GREETING}.wav  (plays first)",          C["text"]),
            (f"Q1 -> Q6   :  {' -> '.join(QUESTION_ORDER)}",                C["text"]),
            (f"Fast(<=3s) :  {AUDIO_PRAISE}.wav  (praise) then next Q",     C["praise"]),
            (f"Slow/None  :  {AUDIO_ENCOURAGE}.wav  (encourage) then next Q",C["slow"]),
            (f"Listen     :  {LISTEN_SECS}s per question",                   C["text"]),
            ( "Camera     :  ESP32-CAM   [connected]",                       C["success"]),
            (f"Mic        :  {SERIAL_PORT}  [connected]",                    C["success"]),
        ]
        for i,(l,col) in enumerate(lines):
            self._txt(f,l,(w//2-285,158+i*28),scale=0.56,color=col)

        if int(time.time()*2)%2:
            self._txt(f,">> PRESS SPACE TO START <<",(w//2-188,h-55),
                      scale=0.85,color=C["gold"],thick=2)

    def _draw_session(self, f):
        h,w = f.shape[:2]
        with self.lock:
            state     = self.state
            q_name    = self.current_q_name
            q_num     = self.current_q_num
            countdown = self.countdown
            text      = self.live_text
            amp       = dict(self.live_amp)
            rl        = self.reactive_label
            rc        = self.reactive_col

        total_q = len(QUESTION_ORDER)

        self._panel(f,0,0,w,72)
        labels = {
            self.S_PLAYING: ("PLAYING",   C["warn"]),
            self.S_LISTEN:  ("LISTENING", C["success"]),
            self.S_REACT:   ("REACTING",  C["gold"]),
            self.S_SCORING: ("SCORING",   C["accent"]),
        }
        lbl,col = labels.get(state,(state,C["muted"]))
        self._txt(f,f"[ {lbl} ]",(18,44),scale=0.9,color=col,thick=2)
        if q_num > 0:
            self._txt(f,f"Q{q_num}/{total_q}  --  Audio {q_name}.wav",
                      (w-310,44),scale=0.65,color=C["muted"])

        if state == self.S_REACT and rl:
            bw = 480
            self._panel(f,w//2-bw//2,h//2-55,w//2+bw//2,h//2+55)
            self._txt(f,rl,(w//2-175,h//2+14),scale=1.1,color=rc,thick=3)
            tag = (f"> Playing {AUDIO_PRAISE}.wav (praise)" if "FAST" in rl
                   else f"> Playing {AUDIO_ENCOURAGE}.wav (encourage)")
            self._txt(f,tag,(w//2-155,h//2-20),scale=0.62,color=C["muted"])

        if state == self.S_LISTEN and countdown > 0:
            frac = countdown / LISTEN_SECS
            c2   = C["success"] if frac>0.5 else (C["warn"] if frac>0.25 else C["danger"])
            self._bar(f,18,h-26,w-36,16,frac,col=c2)
            self._txt(f,f"Listening... {countdown}s remaining",
                      (20,h-32),scale=0.52,color=c2)

        px = w-238
        self._panel(f,px-8,84,w-8,84+220)
        self._txt(f,"LIVE SCORES",(px,108),scale=0.55,color=C["accent"])
        ey = self.sm_eye.mean()
        hv = self.sm_hand.mean()
        for i,(lb2,val) in enumerate([("Eye Contact",ey),("Hand Pattern",hv)]):
            yy = 126+i*52
            self._txt(f,lb2,(px,yy),scale=0.46,color=C["muted"])
            self._bar(f,px,yy+8,200,15,val/100,col=self._sc(val))
            self._txt(f,f"{val:.0f}%",(px+204,yy+19),scale=0.46,color=C["text"])
        if amp:
            peak   = min(1.0,amp.get("peak",0)/2048)
            active = amp.get("active",False)
            self._txt(f,"MIC",(px,242),scale=0.46,color=C["muted"])
            self._bar(f,px,250,200,15,peak,
                      col=C["success"] if active else C["danger"])
            self._txt(f,"ACTIVE" if active else "SILENT",
                      (px+204,262),scale=0.38,
                      color=C["success"] if active else C["danger"])

        with self.lock:
            done_scores = list(self.q_scores)
        for i in range(total_q):
            cx = 28+i*38; cy = h-14
            done = i < len(done_scores)
            cur  = (i == q_num-1)
            if done:
                sc = done_scores[i].total_score
                cv2.circle(f,(cx,cy),13,self._sc(sc),-1)
                self._txt(f,str(int(sc//10)),(cx-6,cy+5),scale=0.35,color=(10,10,10))
            elif cur:
                cv2.circle(f,(cx,cy),13,C["accent"],-1)
                self._txt(f,f"Q{i+1}",(cx-9,cy+5),scale=0.35,color=(10,10,10))
            else:
                cv2.circle(f,(cx,cy),13,C["muted"],2)
                self._txt(f,f"Q{i+1}",(cx-9,cy+5),scale=0.35,color=C["muted"])

        if text:
            self._panel(f,10,h-82,w-10,h-42)
            self._txt(f,f"STT: {text[:55]}{'...' if len(text)>55 else ''}",
                      (18,h-56),scale=0.54,color=C["success"])

    def _draw_results(self, f):
        h,w = f.shape[:2]
        cv2.addWeighted(np.full_like(f,12),0.82,f,0.18,0,f)
        self._txt(f,"SESSION COMPLETE",(w//2-180,50),
                  scale=1.0,color=C["gold"],thick=3)
        with self.lock:
            scores = list(self.q_scores)
        if not scores:
            return
        overall = sum(s.total_score for s in scores)/len(scores)
        fast_c  = sum(1 for s in scores if s.was_fast)

        oc,oy,rr = int(w*0.78),200,70
        cv2.circle(f,(oc,oy),rr+5,(30,30,50),-1)
        rc = self._sc(overall)
        for i in range(0,int(360*overall/100),3):
            rd = math.radians(i-90)
            cv2.circle(f,(int(oc+rr*math.cos(rd)),
                          int(oy+rr*math.sin(rd))),4,rc,-1)
        self._txt(f,f"{overall:.0f}",(oc-22,oy+10),scale=1.3,color=rc,thick=3)
        grade = "A" if overall>=90 else "B" if overall>=80 else \
                "C" if overall>=70 else "D" if overall>=55 else "F"
        self._txt(f,grade,(oc-14,oy-20),scale=1.5,color=rc,thick=4)
        self._txt(f,"OVERALL",(oc-38,oy+38),scale=0.48,color=C["muted"])

        self._panel(f,oc-85,oy+55,oc+85,oy+115)
        self._txt(f,f"FAST: {fast_c}/{len(scores)}",(oc-68,oy+80),
                  scale=0.62,color=C["praise"])
        self._txt(f,f"SLOW: {len(scores)-fast_c}/{len(scores)}",(oc-68,oy+104),
                  scale=0.62,color=C["slow"])

        self._panel(f,28,72,int(w*0.67),72+len(scores)*82+30)
        self._txt(f,"QUESTION BREAKDOWN",(48,100),scale=0.68,color=C["accent"])
        for i,qs in enumerate(scores):
            yy   = 125+i*82
            tag  = "FAST" if qs.was_fast else "SLOW"
            tcol = C["praise"] if qs.was_fast else C["slow"]
            self._txt(f,f"Q{qs.q_index+1}  Audio {qs.audio_file}.wav",
                      (48,yy),scale=0.6,color=C["text"])
            self._txt(f,tag,(320,yy),scale=0.6,color=tcol)
            self._txt(f,f"{qs.response_time_secs:.1f}s -> {qs.reactive_audio}.wav",
                      (380,yy),scale=0.5,color=C["muted"])
            self._bar(f,48,yy+12,270,14,qs.total_score/100,
                      col=self._sc(qs.total_score))
            self._txt(f,f"{qs.total_score:.0f}/100",
                      (328,yy+22),scale=0.55,color=self._sc(qs.total_score))
            dims = [("Eye", qs.eye_contact_score),
                    ("Hand",qs.hand_pattern_score),
                    ("Rep", qs.repetition_score),
                    ("KW",  qs.keyword_score),
                    ("Clr", qs.clarity_score),
                    ("Tone",qs.tone_score),
                    ("Gaz", qs.gaze_score)]
            for j,(dl,dv) in enumerate(dims):
                xx = 48+j*36
                self._txt(f,dl,(xx,yy+36),scale=0.28,color=C["muted"])
                self._bar(f,xx,yy+42,28,10,dv/100,col=self._sc(dv))
                self._txt(f,str(int(dv)),(xx,yy+62),scale=0.28,color=C["text"])
            if qs.voice_text:
                self._txt(f,f"'{qs.voice_text[:35]}'",(48,yy+68),
                          scale=0.36,color=C["success"])

        self._txt(f,"[R] New Session   [Q] Quit",
                  (w//2-148,h-28),scale=0.62,color=C["muted"])
    def run(self):
        started = False
        while True:
            ret, frame = self.cam.read()
            if not ret:
                time.sleep(0.03)
                continue

            h,w = frame.shape[:2]

            with self.lock:
                state = self.state

            # MediaPipe only during active listen window
            if state == self.S_LISTEN and MP_AVAILABLE:
                pr,fr,hr = self.pose.process(frame)
                frame    = self.pose.draw(frame,pr,fr,hr)
                eye_s    = self.pose.score_eye_contact(fr)
                hand_d   = self.pose.score_hand_patterns(pr)
                self.sm_eye.push(eye_s)
                self.sm_hand.push(hand_d["hand_score"])
                with self.lock:
                    self.all_eye.append(eye_s)
                    self.all_hand.append(hand_d["hand_score"])
                    self.all_rep.append(hand_d["repetition_score"])

            with self.lock:
                self.live_amp = self.mic.stats()

            if   state == self.S_INTRO:   self._draw_intro(frame)
            elif state == self.S_RESULTS: self._draw_results(frame)
            else:                         self._draw_session(frame)

            cv2.imshow("Autism Interaction Toy", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') and state == self.S_INTRO and not started:
                started = True
                threading.Thread(target=self._session, daemon=True).start()
            elif key == ord('r') and state == self.S_RESULTS:
                with self.lock:
                    self.q_scores.clear()
                    self.state = self.S_INTRO
                started = False

        # Cleanup
        self.cam.release()
        self.mic.close()
        if MP_AVAILABLE:
            self.pose.pose.close()
            self.pose.face.close()
            self.pose.hands_mp.close()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

    def _print_report(self):
        with self.lock:
            scores = list(self.q_scores)
        if not scores:
            return
        print("\n========================================")
        print("       FINAL SESSION REPORT")
        print("========================================")
        print(f"  Audio Flow: {AUDIO_GREETING} -> {' -> '.join(QUESTION_ORDER)}")
        print(f"  {'Q':<4} {'Audio':<7} {'Speed':<10} {'RT(s)':<7}"
              f" {'React':<8} {'Score':<8} {'Gaze'}")
        print("  " + "-"*60)
        for s in scores:
            spd = "FAST" if s.was_fast else "SLOW"
            print(f"  Q{s.q_index+1:<3} {s.audio_file:<7} {spd:<10}"
                  f" {s.response_time_secs:<7.1f} {s.reactive_audio:<8}"
                  f" {s.total_score:<8} {s.gaze_score:.1f}")
        overall = sum(s.total_score for s in scores)/len(scores)
        fast_c  = sum(1 for s in scores if s.was_fast)
        print("  " + "-"*60)
        print(f"  OVERALL : {overall:.1f}/100")
        print(f"  FAST    : {fast_c}/{len(scores)}")
        print(f"  SLOW    : {len(scores)-fast_c}/{len(scores)}")
        print("========================================")

    def get_dashboard_summary(self):
        """Returns a simplified version of the results for the web dashboard."""
        if not self.q_scores:
            return None
            
        total_eye = sum(s.eye_contact_score for s in self.q_scores) / len(self.q_scores)
        total_verbal = sum(s.keyword_score for s in self.q_scores) / len(self.q_scores)
        avg_rt = sum(s.response_time_secs for s in self.q_scores) / len(self.q_scores)

        return {
            "child_name": CHILD_NAME,
            "eye_gaze_avg": round(total_eye, 1),
            "verbal_clarity": round(total_verbal, 1),
            "avg_response_time": round(avg_rt, 2),
            "status": "Test Complete"
        }

if __name__ == "__main__":
    app = AutismToyApp()
    app.run()
