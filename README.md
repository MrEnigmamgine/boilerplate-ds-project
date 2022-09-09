# Boilerplate Data Science Project

Inspired by my time as a student in CodeUp's Data Science course, this project aims to pave the way for other aspiring data scientists by providing generalized tools and templates meant to get a data exploration or machine learning modeling on the ground quickly and easily.  By providing a set of pre-built helper libraries and example notebooks, I hope to guide a beginning data scientist into making awesome projects as well as making my own projects easier to start.

## Project Scope
The boilerplate provided here is meant to be able to handle small, simple datasets.  There are several example datasets available in the pydataset module that will be the starting point for determining what this project can handle.

# How to get started
You'll need an environment in which to run the code and a python environment using version 3.7 or later.  

## Download a copy of this repository
Probably the easiest method to make a copy is to use git to clone the files to your machine:
```
git clone git@github.com:MrEnigmamgine/boilerplate-ds-project.git
```

## Dependencies

This project makes use of several technologies that will need to be installed
* [![python-shield](https://img.shields.io/badge/Python-3-blue?&logo=python&logoColor=white)
    ](https://www.python.org/)
* [![jupyter-shield](https://img.shields.io/badge/Jupyter-notebook-orange?logo=jupyter&logoColor=white)
    ](https://jupyter.org/)
* [![numpy-shield](https://img.shields.io/badge/Numpy-grey?&logo=numpy)
    ](https://numpy.org/)
* [![pandas-shield](https://img.shields.io/badge/Pandas-grey?&logo=pandas)
    ](https://pandas.pydata.org/)
* [![matplotlib-shield](https://img.shields.io/badge/Matplotlib-grey.svg?)
    ](https://matplotlib.org)
* [![seaborn-shield](https://img.shields.io/badge/Seaborn-grey?&logoColor=white)
    ](https://seaborn.pydata.org/)
* [![scipy-shield](https://img.shields.io/badge/SciPy-grey?&logo=scipy&logoColor=white)
    ](https://scipy.org/)
* [![sklearn-shield](https://img.shields.io/badge/_-grey?logo=scikitlearn&logoColor=white&label=scikit-learn)
    ](https://scikit-learn.org/stable/)
* [![nltk-shield](https://img.shields.io/badge/NLTK-grey?&logo=&logoColor=white)
    ](https://textblob.readthedocs.io/en/dev/)
* [![pydataset-shield](https://img.shields.io/badge/Pydataset-grey?&logo=&logoColor=white)
    ](https://pydataset.readthedocs.io/en/latest/)

<!-- * [![xgboost-shield](https://img.shields.io/badge/XGBoost-grey?&logo=&logoColor=white)
    ](https://xgboost.readthedocs.io/en/stable/)
* [![textblob-shield](https://img.shields.io/badge/TextBlob-grey?&logo=&logoColor=white)
    ](https://textblob.readthedocs.io/en/dev/) -->


Dependencies can be installed quickly with just a few lines of code.
```
%pip install notebook
%pip install numpy
%pip install pandas
%pip install matplotlib
%pip install seaborn
%pip install scipy
%pip install sklearn
%pip install nltk
%pip install pydataset
```

Additionally, our implementation of NLTK relies on some data that is not included in the base package. The following script will ensure the data is installed in your environmnet:
```
import nltk

# Ensuring required data is installed.
try:
    nltk.data.find('corpora/wordnet.zip')
except:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4.zip')
except:
    nltk.download('omw-1.4')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
```

