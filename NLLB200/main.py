from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", token=token)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", token=token)

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt", src_lang="eng_Latn")

# Генерация на русском
generated_tokens = model.generate(
    **inputs, 
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("rus_Cyrl")
)
result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(result)