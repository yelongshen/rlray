from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

import librosa

from io import BytesIO
from urllib.request import urlopen

model_path = "../../Qwen2.5-Omni-3B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


# @title inference function
def inference(audio_path):
    messages = [
        {"role": "system", "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],},
        {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)

    text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    audio = output[1]
    return text, audio

audio_path = "./q1.wav"

audio = librosa.load(audio_path, sr=16000)[0]
#display(Audio(audio, rate=16000))

## Use a local HuggingFace model to inference.
response = inference(audio_path)

print(response[0][0])

print(response[1], response[1].shape)
import torchaudio

torchaudio.save("a1.wav", response[1].unsqueeze(0).cpu(), 16_000)

#with open("./a1.wav", "wb") as f:
#    f.write(response[1])      # done—this is a valid WAV file
