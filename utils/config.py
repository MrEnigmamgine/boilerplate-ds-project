import yaml         # %pip install pyyaml


def load_config(path='config.yaml'):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config

CFG = load_config()

def get_config(dot_notation:str, config:dict = CFG) ->dict:
    """Gets a specific key from the config object."""
    for key in dot_notation.split('.'):
        config = config.get(key)
    if not config: 
        raise ValueError(f'The specified path "{dot_notation}" not found in config')
    return config
