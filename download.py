#!/usr/bin/env python3

import os
import sys
import logging
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = os.getenv('MODEL_NAME')
if model_name is None or model_name == "":
    logging.error("Fatal: MODEL_NAME is required")
    sys.exit(1)

logging.info("Downloading model {} from huggingface model hub".format(model_name))

model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained('./models/model')
tokenizer.save_pretrained('./models/model')