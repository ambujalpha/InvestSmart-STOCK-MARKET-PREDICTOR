from django.shortcuts import render
from .models import Stock
from main.models import Main
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json


def stock(request):

    site = Main.objects.get(pk=2)
    allstocks = Stock.objects.all()
    return render(request, 'front/stock.html', {'site': site, 'allstocks': allstocks})


def stock_detail(request, word):

    dataset_train = pd.read_csv('media/Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    json_file = open('media/model_lstm_google.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("media/model_lstm_google.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')

    dataset_test = pd.read_csv('media/Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    # at each day of financial day, we need the prices of prev 60 days
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = loaded_model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    df1 = pd.DataFrame(real_stock_price)
    df2 = pd.DataFrame(predicted_stock_price)
    df3 = pd.DataFrame(range(0, 20, 1))
    df4 = pd.concat([df3, df1, df2], axis=1)
    df = df4.values.tolist()


    site = Main.objects.get(pk=2)
    showstock = Stock.objects.filter(name=word)
    return render(request, 'front/stock_detail.html', {'site': site, 'showstock': showstock, 'df':df})
