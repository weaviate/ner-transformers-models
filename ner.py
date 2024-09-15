from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from pydantic import BaseModel
from gliner import GLiNER
from typing import Optional

class NerInput(BaseModel):
    text: str
    labels: Optional[list]


class NerResult(BaseModel):
    entity: str
    word: str
    certainty: float
    startPosition: int
    endPosition: int


class Ner:
    def __init__(self, model_path: str, model_name: str, cuda_support: bool, cuda_core: str):
        if "gliner" in model_name:
            self.ner = GlinerModel(model_path, model_name, cuda_support, cuda_core)
        else:
            self.ner = BaseNer(model_path, cuda_support, cuda_core)

    async def do(self, input: NerInput):
        return self.ner.do(input)


class BaseNer:
    model: AutoModelForTokenClassification
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str

    def __init__(self, model_path: str, cuda_support: bool, cuda_core: str):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        device = -1
        if self.cuda:
            self.model.to(self.cuda_core)
            device = int(cuda_core[5:]) # form is e.g. cuda:3
        self.model.eval() # make sure we're in inference mode, not training

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=device)

    def do(self, input: NerInput):
        text = input.text
        if len(text) == 0:
            return None
        
        ner_results = self.nlp(text)

        '''
        this is how it looks like:
        ner_results = [{
            "entity": "LOC",
            "score": 0.9996141791343689,
            "word": "Berlin",
            "start": 38,
            "end": 44
        }]

        this is how it should look:
        ner_results = [{
            "entity": "LOC",
            "word": "string",
            "certainty": 0.9996141791343689,
            "startPosition": 38,
            "endPosition: 44
        }]
        '''
        
        for item in ner_results:
            # convert numpy.int64 values to native python int values
            item['certainty'] = float(item.pop('score'))
            item['startPosition'] = item.pop('start')
            item['endPosition'] = item.pop('end')
            del item['index']

        return ner_results


class GlinerModel:
    model: GLiNER
    cuda: bool
    cuda_core: str
    default_labels: list[str]

    def __init__(self, model_path: str, model_name: str, cuda_support: bool, cuda_core: str):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.default_labels = ["person", "organization", "location", "miscellaneous"]
        self.model = GLiNER.from_pretrained(model_name, cache_dir=model_path)
        if self.cuda:
            self.model.to(self.cuda_core)
        self.model.eval()

    def do(self, input: NerInput):
        labels = self.default_labels
        if input.labels is not None and len(input.labels) > 0:
            labels = input.labels

        entities = self.model.predict_entities(input.text, labels)

        ner_results = list()
        for entity in entities:
            res = NerResult(entity=entity["label"],
                            word=entity["text"],
                            certainty=entity["score"],
                            startPosition=entity["start"],
                            endPosition=entity["end"])
            ner_results.append(res)

        return ner_results
