import json
from transformers import AutoConfig

class Meta:
    def __init__(self, model_path: str, model_name: str):
        if model_name.startswith("urchade"):
            with open(f"{model_path}/model_config.json", "r") as f:
                self.config = json.load(f)
        else:
            config = AutoConfig.from_pretrained(model_path)
            self.config = config.to_dict()

    def get(self):
        return {
            'model': self.config
        }
