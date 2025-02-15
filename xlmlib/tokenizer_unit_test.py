import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 


local_path = './Samba_3_8B_IT_long_new/'
tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True) 
tokenizer.model_max_length = 4096 
tokenizer.pad_token = tokenizer.unk_token # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) 
tokenizer.padding_side = 'right'     

text1 = "So the total number is 7 \u00d7 6 \u00d7 5 \u00d7 4 \u00d7 3 \u00d7 2 \u00d7 1 = 7!.\n\nYes, that's right. Therefore, the answer is C) 7!."
print(tokenizer(text1))

text2 = "7 \u00d7 6"
print(tokenizer(text2))
