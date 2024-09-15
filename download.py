#!/usr/bin/env python3

import os
import sys
import json
from transformers import AutoModelForTokenClassification, AutoTokenizer
from gliner import GLiNER

model_dir = "./models/model"
model_name = os.getenv('MODEL_NAME')
if model_name is None or model_name == "":
    print("Fatal: MODEL_NAME is required")
    sys.exit(1)

# download model
if "gliner" in model_name.lower():
    print("Downloading GLiNER model {}".format(model_name))
    model = GLiNER.from_pretrained(model_name, cache_dir=model_dir)
    # save model config
    with open(f"{model_dir}/model_config.json", "w") as f:
        json.dump(model.config.to_dict(), f)
else:
    print("Downloading model {} from huggingface model hub".format(model_name))

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

# save model name
with open(f"{model_dir}/model_name", "w") as f:
    f.write(model_name)

print("Success")
