import mlflow
import mlflow.data
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

mlflow.set_tracking_uri('http://localhost:5000')

wine = load_breast_cancer()

x = pd.DataFrame(data = wine.data, columns= wine.feature_names)
y = pd.DataFrame(data= wine.target, columns = ['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

rf = RandomForestClassifier()
n_estimator = [10,20,50,100]
max_depth = [None, 2, 4, 6]

param_grid = {
    "n_estimators": n_estimator,
    "max_depth": max_depth
}

grid = GridSearchCV(estimator= rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

mlflow.set_experiment("Hyperparameter Tuning")
with mlflow.start_run() as parent:
    
    grid.fit(x_train, y_train)
    
    for i in range(len(grid.cv_results_['params'])):
        
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid.cv_results_['mean_test_score'][i])
    
    best_params = grid.best_params_
    best_score = grid.best_score_
    best_estimator = grid.best_estimator_
    
    mlflow.log_metric("accuracy", best_score)
    mlflow.log_params(best_params)
    
    train_df = x_train.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training_data")
    
    test_df = x_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing_data")
    
    mlflow.set_tags({'Author': 'Abhijeet', 'Project': 'ML Flow Project'})
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(best_estimator, "Best Model")
    
    
    print(best_params)
    print(best_score)
    