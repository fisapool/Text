# Paraphraser Microservice

A FastAPI-based microservice for advanced paraphrasing using HuggingFace Transformers. Supports multiple languages and styles.

## Usage

### Local (Python)
```bash
pip install -r app/requirements.txt
uvicorn paraphrase_service:app --host 0.0.0.0 --port 8001
```

### Docker
```bash
docker-compose up --build
```

### API Example
POST /paraphrase
```json
{
  "text": "Your text here.",
  "lang": "en"
}
```

## Supported Languages/Models
- English: Vamsi/T5_Paraphrase_Paws
- French: plguillou/t5-base-fr-sum-cnndm
- German: mrm8488/bert2bert_shared-german-finetuned-summarization
- (Add more as needed)

## Extending
To add more languages or styles, update the `MODELS` dictionary in `app/paraphrase_service.py`.

## Node.js Integration Example
```js
async function callParaphrasingService(text, lang = "en") {
  const response = await fetch('http://localhost:8001/paraphrase', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang }),
  });
  if (!response.ok) throw new Error('Paraphrasing service unavailable');
  const data = await response.json();
  return data.paraphrase;
}
``` 