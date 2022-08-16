from flask import Flask, request
from pathlib import Path
import numpy as np
import pickle
# import my custom methods
from utils.mytokenizer import LemmaTokenizer
from utils.helper import get_topic_terms

# set path variables
MODELPATH = Path(Path.cwd() / 'model')

# start app
app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Let´s do some topic modeling!</h1>"

# use this simple function to test the api e.g. with VSCode´s thunder-client
@app.route('/add', methods=['POST'])
def add_POST():
    data = request.get_json()
    a = data['a']
    b = data['b']
    return str(int(a) + int(b))

# now lets get real and do topic prediction for a given article
@app.route('/predict', methods=['POST'])
def predict_POST():
    # import trained model
    with open(MODELPATH / 'pipe_model', 'rb') as f:
        pipe = pickle.load(f)
    # get text with key 'article' and value 'text'
    article = request.get_json()['article']
    # get predicted topic
    topic_num = np.argmax(pipe.transform([article]))
    top_terms = get_topic_terms(pipe, topic_num, top_n=10)
    return str(top_terms)

app.run(host="0.0.0.0", port=int("5000"), debug=True)  