import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from ner import Ner, NerInput
from meta import Meta

app = FastAPI()
ner : Ner
meta_config : Meta
logger = getLogger('uvicorn')


@app.on_event("startup")
def startup_event():
    global ner
    global meta_config

    cuda_env = os.getenv("ENABLE_CUDA")
    cuda_support=False
    cuda_core=""

    if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
        cuda_support=True
        cuda_core = os.getenv("CUDA_CORE")
        if cuda_core is None or cuda_core == "":
            cuda_core = "cuda:0"
        logger.info(f"CUDA_CORE set to {cuda_core}")
    else:
        logger.info("Running on CPU")

    ner = Ner('./models/model', cuda_support, cuda_core)
    meta_config = Meta('./models/model')


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
    return meta_config.get()


@app.post("/ner/")
async def read_item(item: NerInput, response: Response):
    try:
        tokens = await ner.do(item)

        jsonToReturn = {
            "text": item.text,
            "tokens": tokens
        }

        '''
        tokens = [{
                "entity": "string",
                "word": "string",
                "certainty": 0.9     # float
                "startPosition": 1   # int
                "endPosition: 2      # int
            }] # Or None

        text = "string"
        '''

        return jsonToReturn

    except Exception as e:
        logger.exception(
            'Something went wrong while extracting data.'
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
