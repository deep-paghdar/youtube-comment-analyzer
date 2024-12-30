from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

config = Config.get_config()

def train_model(model_name, X_train, y_train):
    pipline = create_pipeline(model_name)
    pipline.fit(X_train, y_train)
    return pipline
    



def create_pipeline(model_name):
    tfidf_params = config['models']['tfidf']['params']
    tfidf_params['ngram_range'] = tuple(tfidf_params['ngram_range'])
    print(tfidf_params['ngram_range'])
    model_params = config['models'][model_name]['params']

    if model_name == "xgboost":
        model = XGBClassifier(**model_params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**model_params)
    elif model_name == "lightgbm":
        model = LGBMClassifier(**model_params)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        (model_name, model)
    ])

    return pipeline

