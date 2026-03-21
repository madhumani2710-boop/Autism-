import cv2
import time

def main(duration=60):
    """
    Renamed to main() to match your app.py call.
    Runs webcam eye-tracking and returns gaze percentages.
    """
    # 0 = Default Webcam, change to 1 if using external ESP32-CAM via USB
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        return {"status": "error", "message": "Could not open webcam"}

    start_time = time.time()
    
    frames_attracted = 0
    frames_distracted = 0
    total_frames = 0
    blink_count = 0 # Added for extra diagnostic data

    print(f"Phase 2 Gaze Analysis Started for {duration}s...")

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # --- GAZE DETECTION LOGIC ---
        # In a full implementation, you'd use MediaPipe face_mesh here.
        # For the integration to work, we simulate the logic:
        # if child_is_looking_at_camera: frames_attracted += 1
        
        # SIMULATION: We'll assume they look 75% of the time for this demo
        # Replace this line with your actual MediaPipe detection result
        is_looking = True 
        
        if is_looking:
            frames_attracted += 1
        else:
            frames_distracted += 1

        # --- UI OVERLAY ---
        elapsed = int(time.time() - start_time)
        cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Phase 2: {elapsed}/{duration}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "KEEP CHILD IN FRAME", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('NeuroNova Phase 2: Eye-Gaze', frame)

        # Allow manual override/quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # --- FINAL SCORE CALCULATION ---
    if total_frames > 0:
        attracted_pct = round((frames_attracted / total_frames) * 100)
        distracted_pct = 100 - attracted_pct
    else:
        attracted_pct, distracted_pct = 0, 0

    return {
        "status": "success",
        "attracted": attracted_pct,
        "distracted": distracted_pct,
        "blinks": blink_count, # Matches your Dashboard's 'p.blinks' field
        "total_frames": total_frames
    }