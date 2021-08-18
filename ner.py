from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from pydantic import BaseModel
import torch

class NerInput(BaseModel):
    text: str

class Ner:
    model: AutoModelForTokenClassification
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str

    def __init__(self, model_path: str, cuda_support: bool, cuda_core: str):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        if self.cuda:
            self.model.to(self.cuda_core)
        self.model.eval() # make sure we're in inference mode, not training

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)


    async def do(self, input: NerInput):
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
            item["start"] = item["start"].item()
            item["end"] = item["end"].item()

            item['certainty'] = item.pop('score')
            item['startPosition'] = item.pop('start')
            item['endPosition'] = item.pop('end')
            del item['index']

        return ner_results
