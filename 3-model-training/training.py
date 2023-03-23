from sklearn.model_selection import train_test_split, GridSearchCV

import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns 
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import os
import warnings
import matplotlib
matplotlib.use('TkAgg', force=True)
warnings.filterwarnings('ignore')

#Importing Machine Learning Model

from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
#statistical Tools
from sklearn import metrics

#To tranform data
from sklearn import preprocessing

#To store data
accuracy = {}
rmse = {}
explained_variance = {}
max_error = {}
MAE = {}


sns.set_style(
    "white",
    {
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)
warnings.simplefilter("ignore")



# A dictionary of menu options
menu_options = {
    1: "Entrenar ",
    2: "Resultados de todos los modelos probados",
    3: "Exit"
    }


def main():
    os.system('clear')
    while(True):

        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        #Check what choice was entered and act accordingly
        if option == 1:
            build_model()
        elif option == 2:
            model_comparison()
        elif option == 3:
            print('Thanks message before exiting')
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 4.')

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

def load_and_drop():
    bike_training = load_training_table()
    bike_training.drop(['date','holiday','year','instant','casual','last_modified','registered', 'humidity'],axis=1,inplace=True)
    return bike_training

def build_model():

    data = load_and_drop()
    gbr = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, 
                                             max_depth=5, min_samples_split=8) # Gradient Boosting Model
    model_path = make_model(data, gbr, "Gradient Boost")

def model_comparison():
     
    with ScreenDimensions() as sd:
        width= sd.width
        height = sd.height

    # Load and analyze data
    data = load_and_drop()

    lr = LinearRegression() #Linear Regression Model
    train_model(data, lr, "Linear Regression")
    
    EN = ElasticNet() #Linear Regression Model
    train_model(data, EN, "ElasticNet")


    knn = KNeighborsRegressor(n_neighbors=10, n_jobs=4, leaf_size=50) # K Nearest Neighbors Regressor model
    train_model(data,knn, "K Nearest Neighbors")
    
    gbr = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, 
                                            max_depth=5, min_samples_split=8) # Gradient Boosting Model
    train_model(data, gbr, "Gradient Boost")


    mlp = MLPRegressor(hidden_layer_sizes=(200,2), learning_rate='adaptive', max_iter=400) #Multi-Layer Percepton Regression model
    train_model(data, mlp, "Multi-layer Perceptron")


def statistics(model,model_name,x_test,y_test):

    print('\n',model_name) # Printing model name
    pred = model.predict(x_test) # predicting our data
    
    acc = metrics.r2_score(y_test, pred)*100 #Checking R2_Score
    accuracy[model_name] = acc # Saving R2_Score to dict.
    print('R2_Score',acc)

    met = np.sqrt(metrics.mean_squared_error(y_test, pred)) #Calculating RMSE
    print('RMSE : ', met) 
    rmse[model_name] = met #Saving RMSE

    var = (metrics.explained_variance_score(y_test, pred)) #Calculating explained_variance_score
    print('Explained_Variance : ', var)
    explained_variance[model_name] = var #Saving explained_variance_score

    error = (metrics.max_error(y_test, pred)) #Calculating Max_Error
    print('Max_Error : ', error)
    max_error[model_name] = error #Saving Max_Error
    
    err = metrics.mean_absolute_error(y_test, pred) #Calculating mean_absolute_error
    print("Mean Absolute Error", err, '\n')
    MAE[model_name] = err #Saving mean_absolute_error
    # file name, I'm using *.pickle as a file extension

def make_model(data, model, model_name):
    x = data.copy()
    y = x.pop('count')
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8, random_state=134)
                     
    model.fit(x_train,y_train) # fitting the defined model

    statistics(model,model_name,x_test,y_test)

    file_name = [model_name,'pickle']
    file_name = '.'.join(file_name)
    file_path = os.path.join('../models',file_name)

    if not os.path.exists("../models"):
        os.makedirs("../models")
    with open(file_path, "wb") as file:
        pickle.dump(model, file)

    return file_path


def train_model(data, model, model_name):
    x = data.copy()
    y = x.pop('count')
    x = pd.get_dummies(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=134)
    model.fit(x_train, y_train)
    statistics(model,model_name,x_test,y_test)
    





class ScreenDimensions:
    def __enter__(self):
        # Initialize the object and get the screen dimensions
        self.root = tk.Tk()  # Create a Tk object
        self.width = self.root.winfo_screenwidth()  # Get screen width in pixels
        self.height = self.root.winfo_screenheight()  # Get screen height in pixels
        return self  # Return the ScreenDimensions object to the caller
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up the object when the with statement is exited
        self.root.destroy()  # Destroy the Tk object to free up resources
        
def load_training_table(file_name='../data-warehouse/bike_clean_for_training.csv'):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    data = pd.read_csv(file_name, sep=",")
    return data

if __name__ == "__main__":
    main()

