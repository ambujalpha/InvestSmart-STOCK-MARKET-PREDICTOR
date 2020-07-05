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

    print(word)
    real_stock_price = 0
    predicted_stock_price = 0

    TIME_STEP = 7
    DAYS = 20

    if(word=="Tesla"):
        TIME_STEP=5


    if(word=="Google"):

        dataset_train = pd.read_csv('media/Google/Google_Stock_Price_Train.csv')
        training_set = dataset_train.iloc[:, 1:2].values
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)

        json_file = open('media/Google/model_lstm_google.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("media/Google/model_lstm_google.h5")
        print("Loaded model from disk")
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')

        dataset_test = pd.read_csv('media/Google/Google_Stock_Price_Test.csv')
        real_stock_price = dataset_test.iloc[:, 1:2].values
        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
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

    else:

        str = 'media/' + word + '/' + word + '.csv'

        dataset = pd.DataFrame(pd.read_csv(str))

        dataset.drop(["Date", "High", "Low", "Close", "Volume", "Adj Close"], axis=1, inplace=True)
        dataset = dataset.values

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_scaled = scaler.fit_transform(dataset)

        def dataset_split(dataset):
            train = dataset[0: len(dataset) - DAYS]
            val = dataset[len(dataset) - DAYS - TIME_STEP: len(dataset)]
            return train, val

        train, val = dataset_split(dataset_scaled)

        val_x, val_y = [], []
        for i in range(TIME_STEP, val.shape[0]):
            val_x.append(val[i - TIME_STEP: i, 0])
            val_y.append(val[i, 0])
        val_x, val_y = np.array(val_x), np.array(val_y)

        val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

        str2 = 'media/' + word + '/' + word + '_model.json'
        str3 = 'media/' + word + '/' + word + '_model.h5'
        json_file = open(str2, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(str3)
        print("Loaded model from disk")
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')

        real_prices = val[TIME_STEP:]
        real_prices = scaler.inverse_transform(real_prices)
        real_stock_price=real_prices

        predicted_prices = loaded_model.predict(val_x)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        predicted_stock_price=predicted_prices

    df1 = pd.DataFrame(real_stock_price)
    df2 = pd.DataFrame(predicted_stock_price)
    df3 = pd.DataFrame(range(0, 20, 1))
    df4 = pd.concat([df3, df1, df2], axis=1)
    df = df4.values.tolist()

    ratio = -1
    it = 0
    dt = 0
    n = len(predicted_stock_price)
    for i in range(n):
        for j in range(i + 1, n):
            if ((predicted_stock_price[j] / predicted_stock_price[i]) > ratio):
                ratio = predicted_stock_price[j] / predicted_stock_price[i]
                it = i
                dt = j
    ratio=int(ratio*10000)
    site = Main.objects.get(pk=2)
    showstock = Stock.objects.filter(name=word)
    return render(request, 'front/stock_detail.html', {'site': site, 'showstock': showstock, 'df':df, 'ratio':ratio, 'it':it, 'dt':dt})
