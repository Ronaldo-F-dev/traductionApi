from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch

app = FastAPI()

# Charger les modèles et les tokenizers pour différentes directions de traduction
fr_fon_model_name = 'fine-tuned-model'  # Modèle pour la traduction du français vers le fongbe
fr_fon_tokenizer = MarianTokenizer.from_pretrained(fr_fon_model_name)
fr_fon_model = MarianMTModel.from_pretrained(fr_fon_model_name)

fon_fr_model_name = 'fine-tuned-model-fon-fr'  # Modèle pour la traduction du français vers le fongbe
fon_fr_tokenizer = MarianTokenizer.from_pretrained(fon_fr_model_name)
fon_fr_model = MarianMTModel.from_pretrained(fon_fr_model_name)

# Classe pour la réponse de traduction
class TranslationResponse(BaseModel):
    translated_text: str


# Fonction pour traduire le texte
def translate_text(text, tokenizer, model):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


@app.get("/translate/{lang_pair}/{text}", response_model=TranslationResponse)
def translate(lang_pair: str, text: str):
    try:
        if lang_pair == "fr-to-fon":
            translated_text = translate_text(text, fr_fon_tokenizer, fr_fon_model)
        elif lang_pair == "fon-to-fr":
            translated_text = translate_text(text, fon_fr_tokenizer, fon_fr_model)
        else:
            raise HTTPException(status_code=400, detail="Direction de traduction non prise en charge.")

        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Bienvenu sur l'API de traduction "}
