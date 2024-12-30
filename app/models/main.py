import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_text
from train_model import train_model
from test_model import test_model
from registry import log_model_to_mlflow
import numpy as np

# df = pd.concat([pd.read_csv("/home/deeppaghdar/Documents/kaggle/youtube comment analyzer/archive/Reddit_Data.csv"), 
#                 pd.read_csv("/home/deeppaghdar/Documents/kaggle/youtube comment analyzer/archive/Twitter_Data.csv")], axis=0)

df = pd.read_csv("/home/deeppaghdar/Documents/kaggle/youtube comment analyzer/archive/Twitter_reddit_preprocessed_Data.csv", sep="\t")


df.dropna(inplace=True)

# df['preprocess_text'] = df['clean_text'].apply(preprocess_text)
# df.to_csv("/home/deeppaghdar/Documents/kaggle/youtube comment analyzer/archive/Twitter_reddit_preprocessed_Data.csv", sep="\t", index=False)
# print("preprocess complete.")

df['category'] = df["category"].astype(int)
df['category'] = np.where((df['category']==-1), 2, df['category'])

# df = df.sample(500)
print(df['category'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(df['preprocess_text'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

#Xgboost train, test and mlflow logging
xgb_classifier = train_model("xgboost", X_train, y_train)
accuracy = test_model(xgb_classifier, X_test, y_test)
log_model_to_mlflow("xgboost", xgb_classifier, accuracy)
print("Train, Test and logged xgboost complete.")


#Random Forest train, test and mlflow logging
rf_classifier = train_model("random_forest", X_train, y_train)
accuracy = test_model(rf_classifier, X_test, y_test)
log_model_to_mlflow("random_forest", rf_classifier, accuracy)
print("Train, Test and logged randomforest complete.")

#lightgbm train, test and mlflow logging
lgbm_classifier = train_model("lightgbm", X_train, y_train)
accuracy = test_model(lgbm_classifier, X_test, y_test)
log_model_to_mlflow("lightgbm", lgbm_classifier, accuracy)
print("Train, Test and logged lgbm classifier complete.")