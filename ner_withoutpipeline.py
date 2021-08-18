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

    async def do(self, input: NerInput):
        text = input.text
        if len(text) == 0:
            return None

        tokensScores=[]
        inputs = self.getTokenizedInputs(text)


        # TO DO:
        # remove '[CLS]' and '[SEP]'
        outputs = self.getNamedEntities(inputs)



        for input_value in inputs:
            answersScores.append(self.tryToAnswer(input_value[0], input_value[1]))

        response = ("", 0.0)
        if len(answersScores) > 0:
            for res in answersScores:
                if res[1] > response[1]:
                    response = res
                # print("answer: ", res[0], res[1])

        answer = response[0]
        certainty = response[1]
        if not self.isGoodAnswer(answer):
            return None

        return answer, certainty
