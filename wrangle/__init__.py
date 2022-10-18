import yaml
import pydataset
from contextlib import contextmanager
import os


## Allows the temporary changing of python's directory context.  Useful for running scripts in a different context.
@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


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


with open('config.yaml') as f:
    CFG = yaml.safe_load(f)

if 'acquire' in CFG:
    acquire = CFG['acquire']
    if 'script' in acquire:
        data = get_script_data(acquire['script'])
    elif 'pydataset' in acquire:
        data = pydataset.data(acquire['pydataset'])