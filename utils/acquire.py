import yaml
import pydataset
from contextlib import contextmanager
import os
import pandas as pd

from config import CFG, load_config, get_config


## Allows the temporary changing of python's directory context.  Useful for running scripts in a different context.
@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def read_pydataset(config=CFG):
    import pydataset
    dataset = get_config('acquire.pydataset')
    try:
        return pydataset.data(dataset)
    except:
        return None

def write_cache(data: pd.DataFrame, config=CFG):

    cache_cfg = get_config('acquire.cache', config)
    path = cache_cfg.get('path')
    folder, name = os.path.split(path)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    data.to_csv(path, index=False)

def read_cache(config=CFG):
    cache_cfg = get_config('acquire.cache', CFG)
    path = cache_cfg.get('path')

    if os.path.exists(path):
        data = pd.read_csv(path)
        return data
    else:
        return None

def get_script_data(path: str) -> object:
    """Executes a python script in the specified path.
    
    Args:
        path: Path to a python script.  The script must contain a DATA variable to be imported.
    
    Returns:
        DATA: A data object in a suitable format.  Usually a Pandas DataFrame.

    Raises:
        Exception: If DATA variable is not found in target script.
    """
    # Imports are within the function since they are only needed if this function is called
    import importlib
    import sys
    
    
    # Allow target script to run in its own relative paths
    directory, name = os.path.split(path)
    with cwd(directory): 
        # Use importlib to import and run the script
        spec = importlib.util.spec_from_file_location(name, path)
        custom_script = importlib.util.module_from_spec(spec)
        sys.modules[name] = custom_script
        spec.loader.exec_module(custom_script)
        
        try:
            return custom_script.DATA
        except:
            raise Exception(f'Variable: DATA does not exist in script located at "{path}". Assign the data you with to load to the DATA variable. eg. `DATA = pydataset.data(\'iris\')')

def read_script(config=CFG):
    path = get_config('acquire.script')
    if path:
        if os.path.exists(path):
            data = get_script_data(path)
            return data
        else:
            raise Exception(f'The path specified in "acquire.script": "{path}" does not exist.')
    return None

def acquire(path='config.yaml'):
    
    # Grab a fresh copy of config to avoid having to restart kernal.
    config = load_config(path)
    data = None
    
    cache_config = get_config('acquire.cache')
    if cache_config:
        cache_refresh = get_config('acquire.cache.refresh', config)
        if not cache_refresh:
            data = read_cache(config)

    if data is None:
        data = read_script(config)

    if data is None:
        data = read_pydataset(config)
    
    if data is None:
        raise Exception('Data not found. Is configuration file correct?')

    if cache_config:
        cache_path = get_config('path', cache_config)
        if cache_path:
            if (not os.path.exists(cache_path)) or cache_refresh:
                write_cache(data, config)
    
    return data