# app.py
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment
import numpy as np
import os
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# Load Demucs model once at startup
model = get_model('htdemucs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def load_audio_pydub(file_path, target_sr=44100):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(target_sr).set_channels(2)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples.reshape(-1, 2).T
    samples /= np.iinfo(audio.array_type).max
    return torch.from_numpy(samples), target_sr

@app.route("/")
def home():
    return jsonify({"message": "Demucs API is running!"})

@app.route("/separate", methods=["POST"])
def separate_audio():
    # Expect an uploaded mp3 file
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = "input.mp3"
    file.save(filename)

    waveform, sr = load_audio_pydub(filename)
    sources = apply_model(model, waveform.unsqueeze(0), device=device, shifts=1, split=True, overlap=0.25)[0]

    os.makedirs("outputs", exist_ok=True)
    vocal_path = os.path.join("outputs", "vocals.wav")
    torchaudio.save(vocal_path, sources[3], sr)

    return send_file(vocal_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
