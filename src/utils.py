import yaml
import json


def load_config(config_path):
    if config_path.endswith(".yaml"):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    elif config_path.endswith(".json"):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
