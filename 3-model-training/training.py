from sklearn.model_selection import train_test_split, GridSearchCV
import os
import numpy as np
import pandas as pd
import pickle
import os
import warnings
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



# un diccionario que despliega un menu de opciones
menu_options = {
    1: "Entrenar ",
    2: "Resultados de todos los modelos probados",
    3: "Exit"
    }

#limpiar teminal antes de imprimir
def main():
    os.system('clear')
    while(True):
        print_menu()
        option = ''
        try:
            option = int(input('ingrese su eleccion: '))
        except:
            print('Entrada erronea, por favor ingrese un numero ...')
        #comprobar si la opcion ingresada existe
        if option == 1:
            build_model()
        elif option == 2:
            model_comparison()
        elif option == 3:
            print('Ejecucion terminada')
            exit()
        else:
            print('Opcion invalida, ingrese un numero entre 1 y 4.')

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

# se eliminan categorias que no oportan informacion relevante
def load_and_drop():
    training_data = load_training_table()
    training_data.drop(['date','holiday','year','instant','casual','last_modified','registered', 'humidity'],axis=1,inplace=True)
    print('\nloading...')
    return training_data

def build_model():
    data = load_and_drop()
    # Gradient Boosting Model
    gbr = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, 
                                             max_depth=5, min_samples_split=8) 
    model_path = make_model(data, gbr, "Gradient Boost")

def model_comparison():
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

    print('\n',model_name)
    pred = model.predict(x_test) # resultados
    
    acc = metrics.r2_score(y_test, pred)*100 #Calcular R2_Score
    accuracy[model_name] = acc 
    print('R2_Score',acc)

    met = np.sqrt(metrics.mean_squared_error(y_test, pred)) #Calcular RMSE
    print('RMSE : ', met) 
    rmse[model_name] = met 

    var = (metrics.explained_variance_score(y_test, pred)) #Calcular explained_variance_score
    print('Explained_Variance : ', var)
    explained_variance[model_name] = var 

    error = (metrics.max_error(y_test, pred)) #Calcular Max_Error
    print('Max_Error : ', error)
    max_error[model_name] = error 
    
    err = metrics.mean_absolute_error(y_test, pred) #Calcular mean_absolute_error
    print("Mean Absolute Error", err, '\n')
    MAE[model_name] = err 


def make_model(data, model, model_name):
    x = data.copy()
    y = x.pop('count')
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8, random_state=134)
                     
    model.fit(x_train,y_train) 
    # Se imprimen las metricas de cada modelo
    statistics(model,model_name,x_test,y_test)
    # Se almacena el modelo corresponciente bajo la extencion pickle
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
    
def load_training_table(file_name='../data-warehouse/bike_clean_for_training.csv'):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    data = pd.read_csv(file_name, sep=",")
    return data

if __name__ == "__main__":
    main()

