import os
from datetime import datetime

import pandas as pd

# -----------------------------------------------------------------------------
# parametros de la simulación
day_open ='2011-01-01'
day_close ='2012-12-31'


# las siguientes funciones corresponden a versiones
# con similar proposito encontradas en el proyecto 'Pqrs simulation'
def process_next_weeks(n_weeks=1):
    for _ in range(n_weeks):
        process_next_week()


def process_next_week():
    rdbms_requests_table = load_rdbms_requests_table()
    historical_requests_table = load_historial_requests_table()
    last_procesed_date = rdbms_requests_table.dteday.tail(1).values[0]
    batch = historical_requests_table[
        historical_requests_table.dteday > last_procesed_date
    ]
    batch = select_next_week(batch)
    batch = assign_last_modified_field(batch)
    rdbms_requests_table = pd.concat([rdbms_requests_table, batch])
    overwrite_rdbms_requests_table(rdbms_requests_table)
    print(rdbms_requests_table.loc[batch.index, :])


def select_next_week(batch_data):
    batch_data = batch_data.copy()
    current_date = batch_data.dteday.head(1).values[0]
    next_date = pd.to_datetime(current_date) + pd.Timedelta(days=7)
    next_day = next_date.strftime("%A").lower()
    next_date = next_date.strftime("%Y-%m-%d")
    batch_data = batch_data[batch_data.dteday < next_date]
    return batch_data


def restart(restart_date="2011-12-31"):
    historial_requests_table = load_historial_requests_table()
    requests_table = select_initial_request_table(
        historial_requests_table, restart_date
    )
    overwrite_rdbms_requests_table(requests_table)
    print(requests_table)



def overwrite_rdbms_requests_table(data):
    module_path = os.path.dirname(__file__)
    folder_path = os.path.join(module_path, "../raw-data")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = os.path.join(folder_path, "requests_table.csv")
    data.to_csv(filename, sep=",", index=False)



def load_rdbms_requests_table():
    module_path = os.path.dirname(__file__)
    filename = os.path.join(module_path, "../raw-data/requests_table.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    data = pd.read_csv(filename, sep=",")
    return data

def select_initial_request_table(historical_request_table, restart_date):
    historical_request_table = historical_request_table.copy()
    rdbms_request_table = historical_request_table[
        historical_request_table.dteday <= restart_date
    ]
    rdbms_request_table = assign_last_modified_field(rdbms_request_table)
    return rdbms_request_table


def assign_last_modified_field(table):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    table = table.assign(last_modified=now)
    return table


def load_historial_requests_table():
    module_path = os.path.dirname(__file__)
    filename = os.path.join(module_path, "bike-rental.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    data = pd.read_csv(filename, sep=",")
    return data
