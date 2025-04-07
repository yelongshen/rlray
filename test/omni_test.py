# okay, let's do something fun. 
import librosa
from transformers import WhisperFeatureExtractor, WhisperTokenizer

#  "chunk_length": 300,
#  "dither": 0.0,
#  "feature_extractor_type": "WhisperFeatureExtractor",
#  "feature_size": 128,
#  "hop_length": 160,
#  "image_mean": [
#    0.48145466,
#    0.4578275,
#    0.40821073
#  ],
#  "image_processor_type": "Qwen2VLImageProcessor",
#  "image_std": [
#    0.26862954,
#    0.26130258,
#    0.27577711
#  ],
#  "max_pixels": 12845056,
#  "merge_size": 2,
#  "min_pixels": 3136,
#  "n_fft": 400,
#  "n_samples": 4800000,
#  "nb_max_frames": 30000,
#  "padding_side": "right",
#  "padding_value": 0.0,
#  "patch_size": 14,
#  "processor_class": "Qwen2_5OmniProcessor",
#  "return_attention_mask": true,
#  "sampling_rate": 16000,
#  "temporal_patch_size": 2

if __name__ == "__main__":
    path = './question1.m4a'
    audio_data, _ = librosa.load(path, sr=16000)

    feature_extractor_1 = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    feature_extractor_2 = WhisperFeatureExtractor.from_pretrained('../../qwen2.5/Qwen2.5-Omni-7B')
    feature_extractor_3 = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

    inputs_1 = feature_extractor_1(audio_data, sampling_rate=16000, return_tensors="pt")
    inputs_3 = feature_extractor_3(audio_data, sampling_rate=16000, return_tensors="pt")

    d = inputs_1['input_features'] - inputs_3['input_features']
    print(d)
    print(sum(d*d)/len(d))
    
    #assert torch.allclose(o1, o2, atol=1e-1)
    #assert 
    #print(inputs_1['input_features'].shape)
    #inputs_2 = feature_extractor_2(audio_data, sampling_rate=16000, return_tensors="pt")
    #print(inputs_2['input_features'].shape)
    #print(inputs_3['input_features'].shape)
    
    

    
