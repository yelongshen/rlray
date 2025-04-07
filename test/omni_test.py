# okay, let's do something fun. 
import librosa
from transformers import WhisperFeatureExtractor, WhisperTokenizer

if __name__ == "__main__":
    path = './question1.m4a'
    audio_data, _ = librosa.load(path, sr=16000)

    feature_extractor_1 = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    feature_extractor_2 = WhisperFeatureExtractor.from_pretrained('../../qwen2.5/Qwen2.5-Omni-7B')
    feature_extractor_3 = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

    inputs_1 = feature_extractor_1(audio_data, sampling_rate=16000, return_tensors="pt")
    print(inputs_1)

    inputs_2 = feature_extractor_2(audio_data, sampling_rate=16000, return_tensors="pt")
    print(inputs_2)

    inputs_3 = feature_extractor_3(audio_data, sampling_rate=16000, return_tensors="pt")
    print(inputs_3)
    


    
