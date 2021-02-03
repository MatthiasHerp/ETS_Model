import numpy as np
import pandas as pd
from scipy.optimize import minimize

def model_optimization(params, series, exogen, before, after, model):
    '''
    This function computes optimal parameters for the model by a Maximum Likelihood approach.
    Following Hyndman 2008 p.69 the Adjusted Least Squared (ALS) estimate is equal to the ML estimate
    in the case of homoskedastic errors. As we incorperate the heteroskedasticity of the series threw
    multiplicativ components this assumption is valid. The computation works by passing
    parameters as well as a series and exogen variables to the model. The model then computes an error and gives it
    back to the optimizer which minimizes the error by means of changing the parameter threw a
    Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS) under a set of bounds for the parameters. These bounds are based
    on Forecasting by exponential smoothing (Hyndman et al. 2008) p.25 and Chapters 5,10. In essence we restrict the smoothing
    parameters to be between 0 and 1. The level and slopes are not restricted. The weekly seasonality effects are restricted
    between 0 and infinity as they are multiplicative. Thus they can raise sales to infinity and lower them close to zero.
    Finally the exgen variables which are also multiplicative, are restricted between -1 and infinity. Like the daily
    seasonality they can raise sales to infinity and lower them close to zero. The lower bound difference is due to the
    additiv component of the exogen variables in the model. Note that as users can choose how many days before and after events
    they include in the model there is a loop which extends the bounds list to include these exogen variables.

    Parameters:

        params: initial parameters for the optimization

        series: the time series in a pandas Series format

        exog: the exogen variables in a pandas DataFrame format with each column being a variable and the time as its index

        before: Array [] of the length of the columns of exogen. Each number says how many days in the past are to be considered
               for the events. The numbers are to be arranged in the order of the columns of the events.

        after: Array [] of the length of the columns of exogen. Each number says how many days in the future are to be
               considered for the events. The numbers are to be arranged in the order of the columns of the events.

        model: the model for which the parameters are optimized.

    Return: Results containing information about the optimization such as the optimal parameters,
    the status of the optimization and the Sum of squared errors at the optimum.
    '''

    #Defining bounds

    bounds = [(0.000001,0.9999),(0.000001,0.9999),(0.000001,0.9999),(0.000001,0.9999),(-np.inf,np.inf),(-np.inf,np.inf),
              (0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),
              (0.000001,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf)]

    #adding one bound for each additional day before and after

    for i in range(0,sum(before)+sum(after)):
        bounds.append((-1,np.inf))

    #Optimization

    results = minimize(model, params, args=(series, exogen), method='L-BFGS-B', bounds = bounds)
    
    return results
