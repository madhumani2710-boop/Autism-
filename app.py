from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import test_features      
from final import AutismToyApp  

app = Flask(__name__)
CORS(app)

audio_session = None

@app.route('/start-test', methods=['GET'])
def run_phase2():
    results = test_features.main() 
    return jsonify(results)

@app.route('/start-audio-phase', methods=['POST'])
def start_audio():
    global audio_session
    try:
        audio_session = AutismToyApp()

        thread = threading.Thread(target=audio_session.run, daemon=True)
        thread.start()
        
        return jsonify({"status": "Hardware Started", "message": "Phase 3 Initialized"})
    except Exception as e:
        return jsonify({"status": "Hardware Error", "error": str(e)}), 500

@app.route('/get-audio-results', methods=['GET'])
def get_audio_results():
    global audio_session
    if audio_session:

        if audio_session.state == "RESULTS" or audio_session.state == "SCORING":
            summary = audio_session.get_dashboard_summary()
            if summary:
                return jsonify(summary)
        
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
