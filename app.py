from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import test_features      # Phase 2 Eye Gaze script
from final import AutismToyApp  # Phase 3 Audio script

app = Flask(__name__)
CORS(app)

# Track the Phase 3 session globally
audio_session = None

# ---------------------------------------------------------
# PHASE 2: EYE-GAZE (Webcam)
# ---------------------------------------------------------
@app.route('/start-test', methods=['GET'])
def run_phase2():
    # Existing Eye-Gaze logic
    results = test_features.main() 
    return jsonify(results)

# ---------------------------------------------------------
# PHASE 3: AUDIO RESPONSE (ESP32 Mic + final.py)
# ---------------------------------------------------------
@app.route('/start-audio-phase', methods=['POST'])
def start_audio():
    global audio_session
    try:
        # 1. Initialize teammate's hardware class
        audio_session = AutismToyApp()
        
        # 2. Run in a thread so the Dashboard doesn't freeze
        # This will look for the ESP32 running your main.cpp
        thread = threading.Thread(target=audio_session.run, daemon=True)
        thread.start()
        
        return jsonify({"status": "Hardware Started", "message": "Phase 3 Initialized"})
    except Exception as e:
        return jsonify({"status": "Hardware Error", "error": str(e)}), 500

@app.route('/get-audio-results', methods=['GET'])
def get_audio_results():
    global audio_session
    if audio_session:
        # Check if the session reached the RESULTS state
        if audio_session.state == "RESULTS" or audio_session.state == "SCORING":
            summary = audio_session.get_dashboard_summary()
            if summary:
                return jsonify(summary)
        
        # If still running, return current status
        return jsonify({
            "status": "processing", 
            "current_q": audio_session.current_q_num,
            "live_text": getattr(audio_session, 'live_text', "")
        })
    return jsonify({"status": "not_started"}), 404

if __name__ == '__main__':
    # threaded=True is vital so Flask can answer "How is it going?" 
    # while the hardware is busy testing the child.
    app.run(port=5000, threaded=True)