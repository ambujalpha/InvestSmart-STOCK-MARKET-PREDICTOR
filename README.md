# STOCK-MARKET-PREDICTOR

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/about.jpg)

# (I) Introduction : 

Business and finance sector is today the leader of the world's economy, stock market trading is a
major practice in the finance sector. Financial exchange predictions are always trickier when it
comes to stock market predictions. It is basically a technique where one tries to predict the future
value of current stocks of a company to avoid the loss or perhaps gain profit. This project will
demonstrate a machine learning approach to predict the same using various quantities mentioned
later in this report. Python is the programming language used for better reach and understanding. We
propose a Machine Learning Algorithm which will be trained from different datasets of some
companies available from the past to make near effective predictions.
Stock market prediction is a technique to determine the upcoming worth of a corporation’s stock or
other financial instrument traded on an exchange. A noticeable consequential gain is the sole
purpose of stock market prediction, and, of course, to avoid significant losses. Some individuals may
disagree with the authenticity of results that these predictions considering the efficient market
hypothesis that these predictions cannot be made on the presently available data, thus concluding it
as inherently unpredictable. But there are numerous tools and technologies that help to gain future
trends’ information, thus resulting in effective profits

# (II) Dataset Used : 

Data Set of Google, AMD and Tesla was used.
Tesla - 2010-2017(predicted for a 20 day period)
AMD - 2009-2018(predicted for a 20 day period)
Google - 2012-2016(predicted for a 20 day period)

# (III) Libraries used :

## Numpy :

Fundamental package for scientific computing in Python3, helping us in creating and managing n-dimensional tensors. A vector can be regarded as a 1-D tensor, matrix as 2-D, and so on. 

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/tensor.jpg)
## Matplotlib :

A Python3 plotting library used for data visualization.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/visualize.webp)
## Tensorflow-Keras :
Is an open source deep learning framework for dataflow and differentiable programming. It’s created and maintained by Google.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/tf%20loves%20keras.png)
## Pandas :
Used for data manipulation and analysis. In particular, it offers data structures and operations for
manipulating numerical tables and time series.

## Sci-kit Learn :
Scikit-learn is a free software machine learning library for the Python
programming language. It features various classification, regression and clustering
algorithms including support vector machines and many more.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/sklearn.png)

# (IV) Website Overview : 

Stocks are important to a business because they can help the corporation quickly gain a lot of capital,
raise the prestige of the company with the public since people can now invest in the company, and
allow the initial investors to sell off shares and earn money on their investments. We provide an
efficient solution for easy investment in the stock market so that a layman can also benefit without
having prior knowledge of technicalities that a stock market carries. The Website is named as
InvestSmart and it helps people invest smartlyFirstly, the number of various examples belonging to each class were identified and plotted.

The website of Technological Stock Market Prediction has on its front page options to go to
services, know about the service, read testimonials, go to the home page, contact or stocks.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot(159).png)

From the front page, one navigates to the about us page which explains how InvestSmart works in a
simplified manner.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(160).png)

Next, the user is directed to various services offered by InvestSmart which are
    I. Business Consulting
    II. Stock Market Prediction
    III. Market Analysis
    IV. User Monitoring
    V. Financial Investment
    VI. Financial Management
 
![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(161).png)

As a prototype, the Stock Market Prediction page is ready as of now. With datasets from companies
like Amazon, Google & Data Global, the website accurately predicts the stock of these companies
using the LSTM method.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(166).png)

On clicking upon a company, In this case an american semiconductor manufacturing company
AMD, the user is directed to a page displaying Company information, current stock values and
predictions using the LSTM model. The progress includes embedding graphs to the website with
better accuracy. The website also predicts the best time to invest and divest in a short interval of
days.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(168).png)

The below two pictures show the real stock price and predicted stock price for google along with the
best invest & divest periods. It also gives you an example showing how much money you will get if
you invest a certain amount in a certain stock.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(167).png)

Tesla, Inc. is an American electric vehicle and clean energy company based in Palo Alto, California.
The company specializes in electric vehicle manufacturing, battery energy storage from home to grid
scale and, through its acquisition of SolarCity, solar panel and solar roof tile manufacturing.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(169).png)

Testimonials page

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(162).png)

This page comprises a form which users need to fill if they want assistance or they have any
complaints. It collects users’ data such as Name, contact details, reason of query, etc.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(164).png)

Teams Page

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(163).png)

footer section

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/Screenshot%20(165).png)

# (V) Results : 

## Learning based website for Stock Investment : 

Users can make their IDs on the website and
go through our services section to learn more about Business Consulting, Stock Market
Prediction, Market Analysis, User Monitoring, Financial Investment and Financial
Management. Thus providing a trustworthy platform to the users to invest their money to
achieve their financial goal

## Predicting stock values using Machine learning : 

The project is implemented LSTM using
Keras API of google’s Tensorflow to predict values of stock while training the algorithm on
past data. We concentrated on predicting the trend observed in the value of a stock for the
next 20 days from the day of prediction. For now we are covering 3 companies which are
AMD Inc., Google LLC and Tesla Inc. For this different python libraries were used as stated
above in the report

## User friendly output :

The most challenging aspect of the project was to generate an efficient
output system that is one, user friendly and easily understandable. For this purpose, after our
prediction is completed for a certain company, the web page will also display the best
possible scenarios that when a user and Invest and Divest to gain maximum profit. This
allows the users to clarify what these graphs mean and what challenges are expected to face
to avoid losses.

# (VI) Conclusions :

We were able to produce results which are consistent with the methodology proposed. Mainy the
focus was to integrate the machine learning algorithms with our website for the completion of this
project. This provides a powerful tool in the hands of a layman to observe predicted stocks and
invest accordingly. A user friendly terminal which shows when to invest and when to divest gives a
sense of trust to these predicted values. It is important to note that the results produced are likely to
be the trend in the next 20 days. That explains that in between this time period there might be days
which are not in resonance with the result but the end result is surely accurate to predict the trend.




