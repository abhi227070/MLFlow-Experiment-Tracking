import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dagshub
dagshub.init(repo_owner='abhi227070', repo_name='MLFlow-Experiment-Tracking', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/abhi227070/MLFlow-Experiment-Tracking.mlflow')

wine = load_wine()

x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

n_estimator = 30
max_depth = 2

mlflow.set_experiment("New Experiment")

with mlflow.start_run():
    
    rf = RandomForestClassifier(n_estimators= n_estimator, max_depth= max_depth)
    rf.fit(x_train, y_train)
    
    y_predict = rf.predict(x_test) 
    acc = accuracy_score(y_predict, y_test)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimator", n_estimator)
    
    mlflow.log_artifact(__file__)
    
    mlflow.set_tags({'Author': 'Abhijeet', 'Project': 'ML Flow Project'})
    
    mlflow.sklearn.log_model(rf, "Random Forest")
    
    print(acc)
    