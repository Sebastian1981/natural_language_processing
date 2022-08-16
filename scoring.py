import argparse
from pathlib import Path
import pandas as pd
import pickle

# import my custom methods
from utils.mytokenizer import LemmaTokenizer
from utils.helper import get_article_topic, get_topic_terms

# argument parser
parser = argparse.ArgumentParser(description="Provide article number and number of words per topic",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-an", "--article_num", default=0, type=int, help="provide article number")
parser.add_argument("-tn", "--top_n", default=5, type=int, help="provide number of words per topic")
args = vars(parser.parse_args())
print('script arguments: ', args)

# set path variables
DATAPATH = Path(Path.cwd() / 'data')
MODELPATH = Path(Path.cwd() / 'model')

# import the pipeline model
with open(MODELPATH / 'pipe_model', 'rb') as f:
    pipe = pickle.load(f)

# import txt data as dataframe
X_test = pd.read_csv(DATAPATH / 'npr_test.csv')

def predict():
    # get article topic for single article
    article_num = args["article_num"]
    top_n = args["top_n"]
    topic_num = get_article_topic(pipe, X_test.Article, article_num)
    top_terms = get_topic_terms(pipe, topic_num, top_n)

    # check results
    print('\n \n')
    print('dataset contains {} articles.'.format(len(X_test)))
    print(X_test.head())
    print('\n')
    print('The main topic for article #{} is topic #{}'.format(article_num, topic_num))
    print('The top {} terms for topic #{} are: \n {}'.format(top_n, topic_num, top_terms))
    print('\n')


if __name__ == "__main__":
    predict()