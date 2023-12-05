import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from pytictoc import TicToc
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
import itertools

def getDelaydata(file_path):
    
    df = pd.read_parquet(file_path)

    dfdelays = (df['timestamps.server.receive.wall'] - df['timestamps.client.send.wall']) / 1e6

    # output_csv_path = 'wall_latency.csv' 
    # df['wall_latency'].to_csv(output_csv_path, index=False)

    #plot the 'wall_latency' column
    plt.figure(figsize=(10, 6))
    plt.plot(dfdelays)
    plt.title('Wall Latency Over Time')
    plt.xlabel('Index')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    plt.show()

    return dfdelays

def sliceData(data, start_index, length):
    """
    Slices the data starting from 'start_index' for 'length' number of rows.

    :param data: Pandas DataFrame containing the data.
    :param start_index: The starting index for the slice.
    :param length: The number of rows to include in the slice.
    :return: Sliced DataFrame.
    """
    end_index = min(start_index + length, len(data))
    
    # Returning the sliced data
    return data.iloc[start_index:end_index]

def draw_ts(timeSeries):
    """
    Draws a time series plot.
    
    :param timeSeries: Pandas Series containing the time series data.
    """
    plt.figure(figsize=(12, 6), facecolor='white')
    timeSeries.plot(color='blue')
    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.show()

# Function to draw moving averages, variance
"""
They are 3 types of patterns that are usually observed in time series data:-

1) Trend: It describes the movement of data values either higher or lower at regular intervals of time over a long period. If the movement of data value is in the upper pattern, then it is known as an 
upward trend and if the movement of data value shows a lower pattern then it is known as a downward trend. If the data values show a constant movement, then is known as a horizontal trend.

2) Seasonality: It is a continuous upward and downward trend that repeat itself after a fixed interval of time.

3) Irregularity: It has no systematic pattern and occurs only for a short period of time and it does not repeat itself after a fixed interval of time. It can also be known as noise or residual.
"""
def draw_patterns(timeSeries, size):
    """
    Draws the original time series along with its rolling and weighted rolling mean.
    
    :param timeSeries: Pandas Series containing the time series data.
    :param size: Window size for calculating the moving averages.
    """
    plt.figure(figsize=(12, 6), facecolor='white')
    
    # Calculate rolling mean and exponential weighted mean
    rol_mean = timeSeries.rolling(window=size).mean()
    rol_weighted_mean = timeSeries.ewm(span=size).mean()
    rol_variance = timeSeries.rolling(window=size).std()

    # Plotting
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    rol_variance.plot(color='green', label='Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Trend Lines with Rolling & Weighted Means, Standard Deviation')
    plt.xlabel('time')
    plt.ylabel('Values')
    plt.show()

# Function for stationarity test
"""
Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.
We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above 
the threshold suggests we fail to reject the null hypothesis (non-stationary).
"""
def test_stationarity(ts):
    """
    Performs the Augmented Dickey-Fuller test on the time series data and returns the test results.
    
    :param ts: Pandas Series containing the time series data.
    :return: A Pandas Series containing the test statistic, p-value, and critical values.
    """
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# Function to draw ACF and PACF plots
def draw_acf_pacf(ts, lags):
    """
    Draws the Autocorrelation and Partial Autocorrelation plots for the time series data.
    
    :param ts: Pandas Series containing the time series data.
    :param lags: The number of lags to include in the plots.
    """
    plt.figure(figsize=(12, 6), facecolor='white')
    ax1 = plt.subplot(211)
    plot_acf(ts, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function')

    ax2 = plt.subplot(212)
    plot_pacf(ts, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()

def traintestSplit(data, percent):
    """
    Split the data into training and test sequence

    :param data: Time-series data to split
    :param percent: percentage of training data in the range (0,1)
    """
    trainlen = int(percent*len(data))
    train, test = data[0:trainlen], data[trainlen:len(data)]
    print("Length of data: ",len(data))
    print("Training sequence length: ",len(train))
    print("Test sequence length: ",len(test))
    return train,test

def findmodelOrder(data):
    model = auto_arima(data, 
                      m=1,               # frequency of series                      
                      seasonal=False,     # TRUE if seasonal series
                      d=None,             # let model determine 'd'
                      test='adf',         # use adftest to find optimal 'd'
                      start_p=0, start_q=0, # minimum p and q
                      max_p=12, max_q=12, # maximum p and q
                      D=None,             # let model determine 'D'
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    return model
    
def rollingForecast(data,test,predlen,order):
    train = [x for x in data]       
    testnp = np.array(test.values)
    predictions = []
    for t in range(predlen):
        model = ARIMA(endog=train,exog=None,order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()
        predictions.append(yhat)
        act = testnp[t]
        train.append(act)
        train.pop(0)
    return predictions

# def mean_absolute_percentage_error(y_true, y_pred): 
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) /y_true))*100

def forecastResult(train,test,order):
    predlen = len(test)
    predictions = rollingForecast(train,test,predlen,order)
    mape = mean_absolute_percentage_error(test.values, predictions)*100
    print(mape,'%')

    # Plotting the original data and the fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(test.values, color='blue', label='Original Test Data')
    # plt.plot(testrol_mean.values, color='green', label='Rolling Mean of Test Data')
    plt.plot(predictions, color='red', label='Rolling Forecasted Values')
    plt.title('ARIMA Model Fit')
    plt.xlabel('Observation')
    plt.ylabel('Wall Latency')
    plt.legend()
    plt.show()
    



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    t = TicToc()
    # Data preparation
    file_path = '02 latency measurement/session13-UL/10-42-3-2_55500_20231113_134426.parquet'  # Replace with your .parquet file path
    dfdelays = getDelaydata(file_path)
    dfwindata = pd.read_csv("02 latency measurement/session13-UL/10-42-3-2_55500_20231113_134426_moving_subsampled_window_15_rate_5.csv",index_col=False)
    dfwindata = dfwindata["0"]
    sliced_data = sliceData(dfdelays, start_index=50000, length=500)
    sliced_data2 = sliceData(dfwindata, start_index=50000, length=500)
    #draw_patterns(sliced_data,size=20)
    #draw_acf_pacf(sliced_data,lags=30)
    #test_stationarity(sliced_data)
    train,test = traintestSplit(sliced_data,0.75)
    trainrol_mean,testrol_mean = traintestSplit(sliced_data2,0.75)
    # trainrol_mean = train.rolling(window=15,min_periods=1).mean().dropna()
    # testrol_mean = test.rolling(window=15,min_periods=1).mean().dropna()

    # Find model order using auto arima function
    #model = findmodelOrder(train)
    #model = findmodelOrder(trainrol_mean)
    
    t.tic()
    # Forecast result for direct data
    #forecastResult(train,test,(5,1,0))
    
    # Forecast result for rolling mean of data
    forecastResult(trainrol_mean,testrol_mean,(3,1,0))
    t.toc()