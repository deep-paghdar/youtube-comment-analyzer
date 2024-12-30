import yaml

class Config:
    _config = None
    
    @classmethod
    def get_config(cls, config_path="models/config.yaml"):
        if cls._config is None:
            with open(config_path, "r") as file:
                cls._config = yaml.safe_load(file)
        return cls._config