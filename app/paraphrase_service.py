from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

MODELS = {
    "en": "Vamsi/T5_Paraphrase_Paws",
    "fr": "plguillou/t5-base-fr-sum-cnndm",
    "de": "mrm8488/bert2bert_shared-german-finetuned-summarization",
    # Add more as needed
}

loaded_models = {}
loaded_tokenizers = {}

def get_model_and_tokenizer(lang):
    if lang not in loaded_models:
        model_name = MODELS.get(lang, MODELS["en"])
        loaded_tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
        loaded_models[lang] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return loaded_tokenizers[lang], loaded_models[lang]

class ParaphraseRequest(BaseModel):
    text: str
    lang: str = "en"

@app.post("/paraphrase")
def paraphrase(req: ParaphraseRequest):
    tokenizer, model = get_model_and_tokenizer(req.lang)
    input_text = f"paraphrase: {req.text} </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt")
    outputs = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=1
    )
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"paraphrase": paraphrased} 