import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statistics import mean

def Initial_Parameter_calculater(series,exogen):
    '''
    This function calculates initial parameters for the optimization of our ETS Model for our series.
    The Calculation is odne according to Forecasting by exponential smoothing (Hyndman et al. 2008) p.23-24.
    First, the initial seasonal parameters are calculated. This is done by computing a 7 lags moving average and then an
    additional 2 lag moving average on the resulting data. These results are used to detrend the series. Finally the
    average of the detrended values are used to obtain the initial seasonal parameters.
    Second, the initial Level and slope parameters are calculated. This is done by calculating a linear regression with
    a time dependent trend on first ten seasonally adjusted values. The model intercept becomes the initial level parameter.
    The initial slope is calculated by dividing the model trend through the mean of the first ten values of the series.
    The division is done as our model has a multiplicativ trend.
    The initial parameters for the exogen effects are calculated similar to the slope coefficient. We calculate a regression
    with each exogen variable as an explanatory variable. Then we divide the resulting coefficients by the mean of the series
    to obtain our initial parameters. Note that we use regress onto entire series as we have few observations for some events.
    Finally note that the smoothing parameters are set at 0.01 for beta and gamma and at 0.99 for omega. This assumes a
    consistent level, trend and seasonal effect, as small alpha, beta and gamma values mean weak adjustments of the
    level, slope and seasonal components to forecasting errors. A high omega value assumes a weak dampening of the trend
    as it is close to a value of 1 which would be a consistent trend.


    Parameters:

        series: the time series in a pandas Series format

        exogen: the exogen variables in a pandas DataFrame format with each column being a variable and the time as its index

    Return: an array of starting parameters for the model optimization
    '''

    #Initial seasonal Component

    #Computing Moving Average

    f = series[:371].rolling(window=7).mean()
    f = f.rolling(window=2).mean()

    #Detrending for multiplicative model
    #skip first 7 values of both series as they are used to start the moving average and only go till the 365 time point

    detrended_series = series[7:371]/f[7:]
    detrended_series.index = pd.to_datetime(detrended_series.index, format='%Y-%m-%d')

    #Check what weekday the first observation is and store it in order to get the
    #initial seasonal parameters in the right order.

    Daynumber = pd.to_datetime(series.index, format='%Y-%m-%d')[0].weekday()

    #grouping detrended series by the day of the week and computing the means

    weekyday_means = detrended_series.groupby(detrended_series.index.dayofweek).mean()

    #Define all inital seasonal values.
    #Note:The oldes value is the current seasonal.

    s_init = np.zeros(7)
    for i in range(0, 7):
        s_init[i] = weekyday_means[abs(Daynumber - i)]


    #Normalizing the seasonal indices so they add to m (m=7 as we have weekly seasonality).
    #done by dividing them all by there total sum and multiplying with m.

    total = sum(s_init)

    multiplier = 7 / total

    s_init  = s_init * multiplier

    #Initial Level and slope components

    #creating a dataframe containing the first 10 values seasonaly adjusted (values) and a time index (t)

    first_10 = pd.DataFrame()
    first_10['values'] = np.zeros(10)
    first_10['t'] = range(0,10)

    #computing the seasonal adjustment
    #first by creating a data frame with the first 10 seasonal adjustments

    s_intit_10 = np.concatenate((s_init,s_init[0:3]))
    s_intit_10 = pd.DataFrame(s_intit_10, columns=['inits'])

    #computing the seasonally adjusted values

    for i in range(0,10):
        first_10.at[i,'values'] = series[i] / s_intit_10.at[i,'inits']

    #Computing the Linear regression with the first 10 seasonally adjusted values

    reg = LinearRegression().fit(first_10['t'].values.reshape(-1,1),first_10['values'].values.reshape(-1,1))

    #Initial level component is equal to the intercept

    l_init = reg.intercept_[0]
    
    #Intial slope component is equal to the regression coefficient

    b_init = reg.coef_[0] / mean(series[0:10])

    #Initial values for the regressors

    reg2 = LinearRegression().fit(exogen,series)


    #defining values for starting parameters of the exogen variables
    #as we have a model with multiplicative effect i divide the coefficients by the mean over the time period

    exogen_initial_parameters = reg2.coef_[0:exogen.shape[1]] / mean(series)
   
    #Defining Starting Parameters array
    #The first values are the smoothing parameters: alpha, beta, gamma, omega

    Starting_Parameters = np.concatenate((0.1,
                           0.01,
                           0.01,
                           0.99,
                           l_init,
                           b_init,
                           s_init,
                          exogen_initial_parameters), axis=None)

    return Starting_Parameters
