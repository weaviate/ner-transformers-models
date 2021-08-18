# ner-transformers-models

The inference container for the NER (Named Entity Recognition or Token Classification) Weaviate module.

## Build Docker Container

```sh
LOCAL_REPO="ner-transformers" MODEL_NAME="dbmdz/bert-large-cased-finetuned-conll03-english" ./cicd/build.sh
```