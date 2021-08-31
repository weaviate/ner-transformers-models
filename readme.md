# ner-transformers-models

The inference container for the NER (Named Entity Recognition or Token Classification) Weaviate module.

## Information

This is an inference container to use as model for the [Weaviate NER Module](https://www.semi.technology/developers/weaviate/current/modules/). 

This model works with [NER (Token Classification) Transformer models from Huggingface](https://huggingface.co/models?pipeline_tag=token-classification), such as [bert-base-NER](https://huggingface.co/dslim/bert-base-NER). The code for the application in this repo works well with models that take in a text input like `My name is Sarah and I live in London` and return information in JSON format like this:

```json
[
  {
    "entity_group": "PER",
    "score": 0.9985478520393372,
    "word": "Sarah",
    "start": 11,
    "end": 16
  },
  {
    "entity_group": "LOC",
    "score": 0.999621570110321,
    "word": "London",
    "start": 31,
    "end": 37
  }
]
```

The Weaviate NER Module then takes this output and processes this to GraphQL output.

### Pre-built images

|Model Name|Image Name|
|---|---|
|`bert-large-cased-finetuned-conll03-english` ([Info](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english?text=Eric+H.+Taylor+writes+nothing))|`semitechnologies/bert-large-cased-finetuned-conll03-english`|
| `dslim-bert-base-NER` ([Info](https://huggingface.co/dslim/bert-base-NER)) | `semitechnologies/dslim-bert-base-NER | 

## Build Docker Container

```sh
LOCAL_REPO="ner-transformers" MODEL_NAME="dbmdz/bert-large-cased-finetuned-conll03-english" ./cicd/build.sh
```

## Choose your model

**Make sure that the model gives the output as stated above, otherwise a good-working model isn't ensured!** You can always make a [custom inference container](https://www.semi.technology/developers/weaviate/current/modules/custom-modules.html#a-replace-parts-of-an-existing-module) (and take this repo as an example) or [custom Weaviate model](https://www.semi.technology/developers/contributor-guide/current/weaviate-module-system/how-to-build-a-new-module.html) if your NER model gives different outputs or you want to change the GraphQL design of the NER module entirely. 

### Custom build with any token classification huggingface model

Create a new `Dockerfile` (you do not need to clone this repository, any folder
on your machine is fine), we will name it `camembert-ner.Dockerfile`. Add the
following lines to it:

```
FROM semitechnologies/ner-transformers:custom
RUN MODEL_NAME=Jean-Baptiste/camembert-ner ./download.py
```

Now you just need to build and tag your Dockerfile, we will tag it as
`jean-baptiste-camembert-ner`:

```
docker build -f camembert-ner.Dockerfile -t jean-baptiste-camembert-ner .
```

That's it! You can now push your image to your favorite registry or reference
it locally in your Weaviate `docker-compose.yaml` using the docker tag
`jean-baptiste-camembert-ner`.

### Custom build with a private / local model

In the following example, we are going to build a custom image for a non-public
model which we have locally stored at `./my-model`.

Create a new `Dockerfile` (you do not need to clone this repository, any folder
on your machine is fine), we will name it `my-model.Dockerfile`. Add the
following lines to it:

```
FROM semitechnologies/ner-transformers:custom
COPY ./my-model /app/models/model
```

The above will make sure that your model end ups in the image at
`/app/models/model`. This path is important, so that the application can find the
model.

Now you just need to build and tag your Dockerfile, we will tag it as
`my-model-inference`:

```
docker build -f my-model.Dockerfile -t my-model-inference .
```

That's it! You can now push your image to your favorite registry or reference
it locally in your Weaviate `docker-compose.yaml` using the docker tag
`my-model-inference`.

# More Information

For more information, visit the [official documentation](https://www.semi.technology/developers/weaviate/current/modules/).
