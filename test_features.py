"""
Eye Tracking & Gaze Detection — ESP32-CAM Edition
==================================================
Updated for NeuroNova Dashboard Integration
Returns JSON-ready dictionary for Flask/Web use.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import requests
import threading
from collections import deque
from scipy.spatial import distance as dist

# ============================================================
# 🔥 CHANGE THIS TO YOUR ESP32-CAM IP ADDRESS
# ============================================================
ESP32_IP         = "192.168.137.219"
ESP32_STREAM_URL = f"http://{ESP32_IP}/stream"
# ============================================================

WINDOW_NAME     = "Eye Tracking | ESP32-CAM"
CSV_OUTPUT      = "attention_dataset.csv"

SESSION_SECS    = 60      # 1 minute session
CAL_SECS        = 5       # calibration duration

# ── Thresholds (slightly tuned, overridden after calibration) ──
EAR_BLINK_RATIO = 0.72
EAR_OPEN_BASE   = 0.30
EAR_CONSEC      = 3

GAZE_DEAD       = 0.12
GAZE_FULL       = 0.30

YAW_DEAD        = 15
YAW_FULL        = 45
PITCH_DEAD      = 12
PITCH_FULL      = 32

W_HEAD          = 0.45
W_GAZE          = 0.30
W_EAR           = 0.15
W_BLINK         = 0.10

HYST_LOW        = 42
HYST_HIGH       = 58

EMA_ALPHA       = 0.06
SIG_MED         = 20
NOFACE_LIMIT    = 45

# ── Phases ────────────────────────────────────────────────────
PHASE_WAIT    = "WAITING"
PHASE_CAL     = "CALIBRATING"
PHASE_SESSION = "SESSION"
PHASE_DONE    = "DONE"


mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

LEFT_EAR_IDX   = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_IDX  = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDX   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_IDX  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS_IDX  = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]
HEAD_2D_IDX    = [1, 152, 263, 33, 287, 57]

HEAD_3D_PTS = np.array([
    [0.0,    0.0,    0.0  ],
    [0.0,  -330.0,  -65.0 ],
    [-225.0, 170.0, -135.0],
    [ 225.0, 170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [ 150.0,-150.0, -125.0],
], dtype=np.float64)


class ESP32Stream:
    def __init__(self, url):
        self.url     = url
        self.frame   = None
        self.grabbed = False
        self.stopped = False
        self._lock   = threading.Lock()
        self._error  = None

        print(f"[INFO] Connecting to: {url}")
        try:
            self._response = requests.get(
                url, stream=True, timeout=10,
                headers={"Connection": "keep-alive"}
            )
            ct = self._response.headers.get("Content-Type", "")
            print(f"[INFO] Connected! Content-Type: {ct}")
        except Exception as e:
            self._error = str(e)
            print(f"[ERROR] {e}")
            return

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        print("[INFO] Waiting for first frame (up to 8s)...")
        for _ in range(80):
            if self.grabbed:
                print(f"[INFO] ✅ First frame OK! Window opening...\n")
                return
            time.sleep(0.1)
        print("[WARN] ⚠️  Still no frame. Check URL or stream format.")

    def _reader(self):
        buf = bytes()
        try:
            for chunk in self._response.iter_content(chunk_size=8192):
                if self.stopped:
                    break
                if not chunk:
                    continue
                buf += chunk
                if len(buf) > 2_000_000:
                    last = buf.rfind(b'\xff\xd8')
                    buf = buf[last:] if last > 0 else bytes()
                while True:
                    start = buf.find(b'\xff\xd8')
                    end   = buf.find(b'\xff\xd9', start + 2) if start != -1 else -1
                    if start == -1 or end == -1:
                        break
                    jpg_data = buf[start:end + 2]
                    buf      = buf[end + 2:]
                    img = cv2.imdecode(
                        np.frombuffer(jpg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        with self._lock:
                            self.frame   = img
                            self.grabbed = True
        except Exception as e:
            print(f"[STREAM ERROR] {e}")

    def read(self):
        with self._lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        try:
            self._response.close()
        except Exception:
            pass


def calc_ear(lms, idx, w, h):
    pts = np.array([(lms[i].x * w, lms[i].y * h) for i in idx])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3]) + 1e-6
    return (A + B) / (2.0 * C)

def calc_iris_ratio(lms, eye_idx, iris_idx, w, h):
    eye  = np.array([(lms[i].x * w, lms[i].y * h) for i in eye_idx])
    iris = np.array([(lms[i].x * w, lms[i].y * h) for i in iris_idx])
    cx, cy = np.mean(iris, axis=0)
    xmin, ymin = eye.min(axis=0)
    xmax, ymax = eye.max(axis=0)
    hr = (cx - xmin) / (xmax - xmin + 1e-6)
    vr = (cy - ymin) / (ymax - ymin + 1e-6)
    return float(np.clip(hr, 0, 1)), float(np.clip(vr, 0, 1))

def calc_head_pose(lms, w, h, cam_mat):
    pts2d = np.array(
        [[lms[i].x * w, lms[i].y * h] for i in HEAD_2D_IDX],
        dtype=np.float64
    )
    ok, rvec, _ = cv2.solvePnP(
        HEAD_3D_PTS, pts2d, cam_mat, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy    = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
    yaw   = float(np.degrees(np.arctan2(rmat[1, 0],  rmat[0, 0])))
    roll  = float(np.degrees(np.arctan2(rmat[2, 1],  rmat[2, 2])))
    return yaw, pitch, roll

def gaze_score(gh, gv):
    dh = abs(gh - 0.5) * 2
    dv = abs(gv - 0.5) * 2
    d  = np.hypot(dh, dv)
    if d <= GAZE_DEAD: return 1.0
    if d >= GAZE_FULL: return 0.0
    return 1.0 - (d - GAZE_DEAD) / (GAZE_FULL - GAZE_DEAD)

def head_score(yaw, pitch):
    def s1d(v, dead, full):
        a = abs(v)
        if a <= dead: return 1.0
        if a >= full: return 0.0
        return 1.0 - (a - dead) / (full - dead)
    return (s1d(yaw, YAW_DEAD, YAW_FULL) + s1d(pitch, PITCH_DEAD, PITCH_FULL)) / 2.0

def ear_score(ear):
    return float(np.clip((ear - 0.20) / (EAR_OPEN_BASE - 0.20 + 1e-6), 0.0, 1.0))

def draw_bar(img, x, y, w, h, value, color, label):
    cv2.rectangle(img, (x, y), (x+w, y+h), (50,50,50), -1)
    filled = int(w * np.clip(value/100.0, 0, 1))
    cv2.rectangle(img, (x, y), (x+filled, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (120,120,120), 1)
    cv2.putText(img, f"{label}: {value:.0f}%", (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

def draw_gaze_indicator(img, gh, gv, cx, cy, radius=42):
    cv2.circle(img, (cx, cy), radius, (55,55,55), -1)
    cv2.circle(img, (cx, cy), radius, (140,140,140), 1)
    dx = int(cx + (gh - 0.5) * 2 * radius * 0.7)
    dy = int(cy + (gv - 0.5) * 2 * radius * 0.7)
    cv2.circle(img, (dx, dy), 8, (0,255,255), -1)
    cv2.putText(img, "GAZE", (cx-18, cy+radius+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170,170,170), 1)

class Calibration:
    def __init__(self):
        self.ears=[]; self.yaws=[]; self.pitchs=[]; self.ghs=[]; self.gvs=[]

    def add(self, ear, yaw, pitch, gh, gv):
        self.ears.append(ear)
        self.yaws.append(abs(yaw))
        self.pitchs.append(abs(pitch))
        self.ghs.append(gh)
        self.gvs.append(gv)

    def apply(self):
        global EAR_BLINK_RATIO, EAR_OPEN_BASE, GAZE_DEAD, GAZE_FULL, YAW_DEAD, YAW_FULL, PITCH_DEAD, PITCH_FULL
        if len(self.ears) < 10: return
        ear_mean = float(np.mean(self.ears))
        ear_std  = float(np.std(self.ears))
        EAR_OPEN_BASE   = ear_mean
        EAR_BLINK_RATIO = max(0.55, ear_mean - 2.5 * ear_std)
        nat = max(float(np.std(self.ghs)), float(np.std(self.gvs)))
        GAZE_DEAD = float(np.clip(nat * 1.5, 0.06, 0.18))
        GAZE_FULL = float(np.clip(nat * 4.0, 0.20, 0.50))
        ys, ps = float(np.std(self.yaws)), float(np.std(self.pitchs))
        YAW_DEAD, YAW_FULL = float(np.clip(ys * 2.5 + 5, 8, 20)), float(np.clip(ys * 8.0 + 20, 30, 55))
        PITCH_DEAD, PITCH_FULL = float(np.clip(ps * 2.5 + 4, 6, 18)), float(np.clip(ps * 8.0 + 15, 22, 45))

def print_result(att_pct, dis_pct, err_pct, total_blinks):
    print("\n" + "="*50)
    print(f"✅ Attentive: {att_pct:.1f}% | 🔴 Distracted: {dis_pct:.1f}%")
    print(f"Total Blinks: {total_blinks}")
    print("="*50 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN — Updated to return data to NeuroNova Backend
# ─────────────────────────────────────────────────────────────
def main():
    cap = None
    try:
        cap = ESP32Stream(ESP32_STREAM_URL)
    except Exception:
        return {"error": "Connection Failed"}

    mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

    phase = PHASE_WAIT
    cal, cal_start, sess_start = Calibration(), 0.0, 0.0
    blink_counter, total_blinks, blink_flag = 0, 0, False
    noface_counter, ema_score = 0, 50.0
    median_buf = deque(maxlen=SIG_MED)
    attention_label, prev_label = "ATTENTIVE", "ATTENTIVE"
    frame_count, fps_time, fps = 0, time.time(), 0.0
    att_frames, dis_frames, err_frames, total_frames = 0, 0, 0, 0
    att_pct, dis_pct, err_pct = 0.0, 0.0, 0.0

    csv_file = open(CSV_OUTPUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp","ear","yaw","pitch","roll","gaze_h","gaze_v","attention_score","label"])

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback for connection display
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Connecting to ESP32-CAM...", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
            cv2.imshow(WINDOW_NAME, blank)
            if cv2.waitKey(30) & 0xFF == ord('q'): break
            continue

        frame = cv2.flip(frame, 1)
        FH, FW = frame.shape[:2]
        cam_mat = np.array([[FW,0,FW/2],[0,FW,FH/2],[0,0,1]], dtype=np.float64)
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)
        raw_score = ema_score
        face_ok = False

        if phase == PHASE_WAIT:
            face_ok = results.multi_face_landmarks is not None
            msg = "Face detected! Press SPACE to calibrate" if face_ok else "No face detected"
            cv2.putText(frame, msg, (FW//2-230, FH//2), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,210,80), 2)

        elif phase == PHASE_CAL:
            elapsed = now - cal_start
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                ear_c = (calc_ear(lms, LEFT_EAR_IDX, FW, FH) + calc_ear(lms, RIGHT_EAR_IDX, FW, FH)) / 2.0
                yaw_c, pitch_c, _ = calc_head_pose(lms, FW, FH, cam_mat)
                gh_l, gv_l = calc_iris_ratio(lms, LEFT_EYE_IDX, LEFT_IRIS_IDX, FW, FH)
                gh_r, gv_r = calc_iris_ratio(lms, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, FW, FH)
                cal.add(ear_c, yaw_c, pitch_c, (gh_l+gh_r)/2, (gv_l+gv_r)/2)
            
            cv2.putText(frame, f"CALIBRATING... {max(0, CAL_SECS - elapsed):.1f}s", (28, FH-48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
            if elapsed >= CAL_SECS:
                cal.apply()
                phase, sess_start = PHASE_SESSION, now

        elif phase == PHASE_SESSION:
            elapsed = now - sess_start
            total_frames += 1
            if results.multi_face_landmarks:
                face_ok, noface_counter = True, 0
                lms = results.multi_face_landmarks[0].landmark
                ear = (calc_ear(lms, LEFT_EAR_IDX, FW, FH) + calc_ear(lms, RIGHT_EAR_IDX, FW, FH)) / 2.0
                gh_l, gv_l = calc_iris_ratio(lms, LEFT_EYE_IDX, LEFT_IRIS_IDX, FW, FH)
                gh_r, gv_r = calc_iris_ratio(lms, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, FW, FH)
                gh, gv = (gh_l+gh_r)/2.0, (gv_l+gv_r)/2.0
                yaw, pitch, roll = calc_head_pose(lms, FW, FH, cam_mat)

                if ear < EAR_BLINK_RATIO: blink_counter += 1
                elif blink_counter >= EAR_CONSEC: total_blinks += 1; blink_flag = True; blink_counter = 0
                
                raw_score = (W_HEAD*head_score(yaw,pitch) + W_GAZE*gaze_score(gh,gv) + W_EAR*ear_score(ear) + W_BLINK*(0.6 if blink_flag else 1.0)) * 100.0
                blink_flag = False
                
                ema_score = EMA_ALPHA * raw_score + (1-EMA_ALPHA) * ema_score
                median_buf.append(ema_score)
                smooth_score = float(np.median(median_buf))

                if prev_label == "ATTENTIVE" and smooth_score < HYST_LOW: attention_label = "DISTRACTED"
                elif prev_label == "DISTRACTED" and smooth_score > HYST_HIGH: attention_label = "ATTENTIVE"
                prev_label = attention_label

                if attention_label == "ATTENTIVE": att_frames += 1
                else: dis_frames += 1
            else:
                noface_counter += 1
                if noface_counter > NOFACE_LIMIT: err_frames += 1; raw_score = 0.0

            cv2.putText(frame, f"SESSION: {max(0, SESSION_SECS - elapsed):.0f}s", (FW//2-20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,200,255), 2)
            if elapsed >= SESSION_SECS: phase = PHASE_DONE

        elif phase == PHASE_DONE:
            tf = max(total_frames, 1)
            att_pct, dis_pct, err_pct = 100*att_frames/tf, 100*dis_frames/tf, 100*err_frames/tf
            print_result(att_pct, dis_pct, err_pct, total_blinks)
            break # Exit loop to return results

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' ') and phase == PHASE_WAIT:
            if results.multi_face_landmarks: phase, cal_start = PHASE_CAL, time.time()

    # ── Final Cleanup and Data Return ──
    if cap: cap.release()
    mesh.close()
    csv_file.close()
    cv2.destroyAllWindows()
    
    # CALCULATE FINAL DATA FOR RETURN
    tf = max(total_frames, 1)
    return {
        "attracted": round(100 * att_frames / tf, 2),
        "distracted": round(100 * dis_frames / tf, 2),
        "error": round(100 * err_frames / tf, 2),
        "blinks": total_blinks
    }

if __name__ == "__main__":
    result = main()
    print(f"Final Output for Dashboard: {result}")