# audio loader.
import librosa
#from transformers import WhisperFeatureExtractor, WhisperTokenizer
from safetensors.torch import load_file


if __name__ == "__main__":
    path = './q1.wav'
    SAMPLE_RATE = 16000

    audio_data, _ = librosa.load(path, sr=SAMPLE_RATE, offset=0, duration=None)
    print(audio_data, audio_data.shape)

    #audio_data, _ = librosa.load(path, sr=16000)