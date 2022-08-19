# Natural Language_Processing - Topic Modeling API
The purpose of this project is to build an API to detect topics for given articles. We start experimentally developing the model using the Latent Dirichlet Allocation approach based on Bayesian statistics. The training data contains over 1.000 different english articles covering various topics. Then we build a dataprocessing and modeling pipeline which is the basis for the api script. Here a short overview of a the relevant notebooks and scripts:

- "nlp.ipynb": 
    - short nlp warm up like tokenizing text and 
    - building bag of words model
- "topic_modeling.ipynb": 
    - explore the data i.e. articles
    - data preperation: transform text for modeling using sklearn´s count vectorizer class and lemmatization capabilities
    - model training: train the Latent-Dirichlet-Allocation model
    - evaluate the model
- "topic_modeling_pipe.ipynb": 
    - build a sklearn data preprocessing and modeling pipeline
- "scoring.py":
    - define the predict function to find topics best describing new text using the trained sklearn pipeline
- "flask_api.py":
    - POST api: input text and get topics