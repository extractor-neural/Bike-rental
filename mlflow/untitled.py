import pandas as pd
import mlflow
import os
import numpy as np
import mlflow.sklearn
from sklearn import ensemble
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import ElasticNet
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

def main():
    
    make_experiment(
        experiment_name="red-wine",
        alphas=np.linspace(0.0001, 0.5, 10),
        l1_ratios=np.linspace(0.0001, 0.5, 10),
        n_splits=5,
        verbose=1,
    )



def load_data():

    import pandas as pd

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")


    return x, y

def load_table(file_name='../data-warehouse/bike_clean_for_training.csv'):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    data = pd.read_csv(file_name, sep=",")
    data.drop(['date','holiday','year','instant','casual','last_modified','registered', 'humidity'],axis=1,inplace=True)
    x = data.copy()
    y = x.pop('count')
    return x, y

def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=123456,
    )
    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2

def report(estimator, mse, mae, r2):

    print(estimator, ":", sep="")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

def make_pipeline(estimator):

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    pipeline = Pipeline(
        steps=[
            ("minMaxScaler", MinMaxScaler()),
            ("estimator", estimator),
        ],
    )

    return pipeline

def make_experiment(experiment_name, alphas, l1_ratios, n_splits=5, verbose=1):

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)

    param_grid = {
        "alpha": alphas,
        "l1_ratio": l1_ratios,
    }

    estimator = GridSearchCV(
        estimator=ElasticNet(
            random_state=12345,
        ),
        param_grid=param_grid,
        cv=n_splits,
        refit=True,
        verbose=0,
        return_train_score=False,
    )
    mlflow.set_experiment(experiment_name)
    set_tracking_uri()

    with mlflow.start_run() as run:

        run = mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))

        estimator.fit(x_train, y_train)

        #
        # Reporta el mejor modelo encontrado en la corrida
        #
        estimator = estimator.best_estimator_
        mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
        
        if verbose > 0:
            report(estimator, mse, mae, r2)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
       #
        # Registro del modelo como version 1
        #
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            registered_model_name=f"sklearn-{experiment_name}-model"
        )
        
    display_config()

        
def set_tracking_uri():
    mlflow.set_tracking_uri('sqlite:///mlruns.db')


def display_config():
    print("Current model registry uri: {}".format(mlflow.get_registry_uri()))
    print("      Current tracking uri: {}".format(mlflow.get_tracking_uri()))



if __name__ == '__main__':
    main()