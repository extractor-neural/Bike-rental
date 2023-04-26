import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.pyplot.switch_backend("Agg")


def run():
    _make_plot_status()


def _make_plot_status():
    data = load_test()
    plot_predictions(data['Real_values'],data['Predicted_values'])


def load_test(filename = "../data-warehouse/model_results.csv"):
    data= pd.read_csv(filename,sep=',')
    return data

def plot_predictions(y_test,y_predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:500], label='Real')
    plt.plot(y_predicted[:500], label = 'Prediccion')
    plt.ylabel('#Bicicletas rentadas por hora')
    plt.xlabel('Numero de muestras')
    plt.legend()
    plt.savefig("static/report1.png")
    plt.savefig("static/report2.png",dpi=250)

if __name__ == "__main__":
    run()
