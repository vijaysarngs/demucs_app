# ✅ STEP 1: Fix for Python 3.13 (audioop removed)
import sys
try:
    import audioop
except ModuleNotFoundError:
    import pyaudioop as audioop
    sys.modules["audioop"] = audioop

# ✅ STEP 2: Imports
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment
import numpy as np
import os
from flask import Flask, request, jsonify, send_file
import tempfile

# ✅ Flask app setup
app = Flask(__name__)

# ✅ STEP 3: Audio loader using PyDub
def load_audio_pydub(file_path, target_sr=44100):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(target_sr).set_channels(2)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples.reshape(-1, 2).T  # [channels, samples]
    samples /= np.iinfo(audio.array_type).max
    return torch.from_numpy(samples), target_sr

# ✅ STEP 4: Load Demucs model once
print("🔹 Loading Demucs model...")
model = get_model('htdemucs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✅ Model loaded on {device}")

# ✅ STEP 5: API route
@app.route('/separate', methods=['POST'])
def separate_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        file.save(tmp.name)
        input_path = tmp.name

    print(f"🎧 Received file: {input_path}")
    waveform, sr = load_audio_pydub(input_path)

    print("🔹 Running separation...")
    sources = apply_model(model, waveform.unsqueeze(0).to(device),
                          device=device, shifts=1, split=True, overlap=0.25)[0]

    output_dir = tempfile.mkdtemp()
    stems = ['drums', 'bass', 'other', 'vocals']
    output_paths = {}

    for idx, name in enumerate(stems):
        output_path = os.path.join(output_dir, f"{name}.wav")
        torchaudio.save(output_path, sources[idx].cpu(), sr)
        output_paths[name] = output_path
        print(f"✅ Saved {name} stem")

    return jsonify({
        "message": "Separation complete!",
        "stems": list(output_paths.keys())
    })

# ✅ Health check route
@app.route('/')
def home():
    return jsonify({'message': 'Demucs separation API is live 🎶'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
