import pandas as pd
import pickle
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import tkinter as tk

def main():
    test = load_test()
    x_test = test.copy()
    y_test = x_test.pop('count')
    #Cargar modelo 
    model_path = '../models/Gradient Boost.pickle'
    loaded_model = pickle.load(open(model_path, "rb"))
    
    y_predicted = loaded_model.predict(x_test)

    acc = metrics.r2_score(y_test, y_predicted)*100 #Checking R2_Score
    print('\nR2_Score',acc)

    save_results(y_test,y_predicted)

#Guardar resultados en formato .csv
def save_results(y_test,y_predicted):
    data = pd.DataFrame()
    data['Real_values'] = y_test.tolist()
    data['Predicted_values'] = y_predicted.tolist()
    archive_path='../data-warehouse/model_results.csv'
    data = data.round(decimals = 2)
    data[data<0] = 0
    data.to_csv(archive_path, index=False)  
    print('Sucessful')

# Mostrar comparativa valores reales vs prediccion mediante un grafico 
def plot_predictions(y_test,y_predicted):
    plt.plot(y_test[:500], label='True Values')
    plt.plot(y_predicted[:500], label = 'Predicted Values')
    plt.ylabel('Values')
    plt.xlabel('Numero de comparaciones')
    plt.title('Comparison of True vs. Predicted Values')
    plt.legend()
    plt.show()
        

def load_test(filename = "../raw-data/requests_table.csv"):
    data = pd.read_csv(filename, sep=",", usecols=['instant'])
    line = data.shape[0]
    # Se cargan las ultimas filas agregadas
    weeks_added = int((line-8645)/2)
    
    # La instruccion 'line-50-weeks_add' apunta a las ultimas 50 lineas + lineas agregas a medida que la 
    # simulacion avanza
    test_raw = pd.read_csv(filename,sep=',', skiprows=range(1, line-50-weeks_added))
    test_clean = load_and_drop(test_raw)
    
    return test_clean.round(decimals = 3)

def load_and_drop(test_raw):
    
    columns_real_names = eval(read_categories('../categories_names.txt')[0])
    test_clean = rename_columns_data(test_raw,columns_real_names)
    test_clean.drop(['date','holiday','year','instant','casual','last_modified','registered', 'humidity'],axis=1,inplace=True)
    return test_clean
    

def rename_columns_data(data, columns):
    data.rename(columns = columns , inplace=True)
    return data

def read_categories(file_name, data_split_character = '\n'):
    my_file = open(file_name, "r")
    data = my_file.read()
    data_into_list = data.split(data_split_character)
    my_file.close()
    return data_into_list 

if __name__ == "__main__":
    main()