import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def get_article_topic(pipeline_obj:Pipeline, article:pd.DataFrame, article_num:int):
    """Generate article topics for given article and given i.e. trained pipeline
    """
    topic_num = np.argmax(pipeline_obj.transform([article[article_num]]))
    return topic_num 

def get_topic_terms(pipe_obj:Pipeline, topic_num=0, top_n=3):
    """ get the index position of the top 3 terms in a topic.
    input the fitted laten dirichtlet object.
    input the fitted count-vectorizer object.
    input the topic number.
    input the top-n words belonging to each topic.
    output the top_n words for topic_num. """
    return [pipe_obj['cv'].get_feature_names_out()[ind] for ind in pipe_obj['lda'].components_[topic_num].argsort()[-top_n:]]