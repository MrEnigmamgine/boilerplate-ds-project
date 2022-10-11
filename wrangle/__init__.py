import yaml
import pydataset

with open('config.yaml') as f:
    CFG = yaml.safe_load(f)

if CFG.get('data').get('pydataset'):
    DATA = pydataset.data(CFG.get('data').get('pydataset'))