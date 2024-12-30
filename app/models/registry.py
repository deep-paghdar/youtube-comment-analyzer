import mlflow
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from config import Config

config = Config.get_config()

def log_model_to_mlflow(model_name, model, accuracy):
    mlflow.set_experiment("model_versioning_experiment")
    model_params = config['models'][model_name]['params']
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(model_params)
        
        # Log metrics (Accuracy in this case)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric(f"{model_name} accuracy", accuracy)
        
        # Log the model itself
        # if isinstance(model.named_steps[model_name], XGBClassifier):
        #     print("i am in xgboost")
        #     mlflow.xgboost.log_model(model, model_name)
        # elif isinstance(model.named_steps[model_name], RandomForestClassifier):
        #     mlflow.sklearn.log_model(model, model_name)
        # elif isinstance(model.named_steps[model_name], LGBMClassifier):
        #     mlflow.lightgbm.log_model(model, model_name)

        model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id)
        mlflow.register_model(model_uri, model_name)
        
        print(f"Logged {model_name} model with accuracy: {accuracy}")