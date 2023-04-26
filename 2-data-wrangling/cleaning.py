import warnings
import numpy as np
import pandas as pd
import os
warnings.filterwarnings('ignore')



def main():
    load_and_fix_data()

def load_and_fix_data(limit=8645):

    #8645 corresponde al numero de entradas en un año calendario
    bike_rental_raw = load_table()[:limit]
    
    #se carga el contenido del archivo 'categories_names' para renombrar categorias existentes
    columns_real_names = eval(read_categories('../categories_names.txt')[0])
    bike_rental_clean = clean_data(bike_rental_raw)
    bike_rental_clean = rename_columns_data(bike_rental_clean,columns_real_names)
    
    #se guarda una copia de los datos limpios
    overwrite_rdbms_requests_table(bike_rental_clean, file_name = 'bike_clean_for_training.csv',
                                  folder_path = '../data-warehouse')

    #se crea una version de solo lectura
    codes = read_categories('categories_per_column.txt')
    for s in codes:
        mapping_num_to_categories(bike_rental_clean,s)

    #se almacena la version de entrenamiento de la base de datos
    overwrite_rdbms_requests_table(bike_rental_clean, file_name = 'bike_clean_for_reading.csv',
                                  folder_path = '../data-warehouse')
    print('\nRutina exitosa, revisar la carpeta data-warehouse 2 archivos creados! ')
    



def load_table(file_name='../raw-data/requests_table.csv'):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    data = pd.read_csv(file_name, sep=",")
    return data


def clean_data(data):
    nan_count= data.isna().sum().sum()
    data.dropna(inplace=True)
    print('\nSe encontraron y se corrigieron {} valores nulos'.format(nan_count))
    return data

def rename_columns_data(data, columns):
    data.rename(columns = columns , inplace=True)
    return data

def overwrite_rdbms_requests_table(data, folder_path = "hola_mundo", file_name = 'request_table.csv'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    data.to_csv(file_path, sep=",", index=False)

def read_categories(file_name, data_split_character = '\n'):
    # abrir el archivo en modo de lectura'r' y leer
    my_file = open(file_name, "r")
    data = my_file.read()
    #se dividen los valores usando el caracter '\n' como separador
    data_into_list = data.split(data_split_character)
    my_file.close()
    
    return data_into_list 
#reemplaza los valores numericos de las colomnas categoricas cadenas
def mapping_num_to_categories(data, codes):
    col_name,  col_codes = codes.split('.')
    data[col_name] = data[col_name].map(eval(col_codes))



if __name__ == "__main__":
    main()