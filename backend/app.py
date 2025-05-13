from flask import Flask, request, jsonify
import os
import datetime
from model import recognize_speech
from subprocess import Popen

app = Flask(__name__)

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    file = request.files['audio']
    filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    filepath = os.path.join(DATA_DIR, filename)
    file.save(filepath)
    transcription = recognize_speech(filepath)
    return jsonify({'transcription': transcription})

@app.route('/api/train', methods=['POST'])
def train():
    audio_file = request.files['audio']
    phrase = request.form['phrase']
    filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{phrase}.wav"
    filepath = os.path.join(DATA_DIR, filename)
    audio_file.save(filepath)

    # Запустить тренировочный скрипт в фоне
    Popen(['python3', 'train.py'])
    return jsonify({'status': 'Обучение запущено'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)