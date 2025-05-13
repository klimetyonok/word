import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf

# Загрузка модели
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def recognize_speech(wav_path):
    speech, sr = sf.read(wav_path)
    input_values = tokenizer(speech, return_tensors='pt', padding='longest').input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription