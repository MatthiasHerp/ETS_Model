import pandas as pd
import numpy as np

def model(params, series, exogen):

    """
    This function runs an ETS(M,Ad,M) model with exogen variables. This is an Error, Trend, Seasonality exponential smoothing
    model.The first M stands for multiplicative or relative errors, the Ad for an additive dampend trend and the last M for
    multiplicative seasonality. The model also contains additional exogen variables which are dummies for certain events.
    The actual computation of the fit model is done in the function ETS_M_Ad_M which further contains the functions
    calc_new_estimates, calc_error, save_estimates and seasonal_matrices. These are all explained in the following code.
    
    Parameters:

        params: model parameters

        series: the time series in a pandas Series format

        exog: the exogen variables in a pandas DataFrame format with each column being a variable and the time as its index
    
    Return: The function returns the sum of squared error of the fitted model. This allows the model to be inputed
    into an optimizer which minimizes the sum of squared residuals dependent on the input parameters (params).
    """
    #defining all model parameters from the params vector
    #Note that the seasonal and exogen variable parameters are vectors while the other parameters are scalars

    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    omega = params[3]
    level_initial = params[4]
    slope_initial = params[5]
    seasonal_initial = np.vstack(params[6:13])

    #added len(exogen) as now we have variable number of exogen variables due to days before and after

    reg = (params[13:13+len(exogen.columns)])

    #Built in exception that gives out the parameters and the error sum if an error in the model occurs

    try:
        results = ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen)

    except:
        print('alpha:', alpha, 'beta:', beta, 'gamma:', gamma, 'omega:', omega, level_initial, slope_initial,
              seasonal_initial, 'reg:', reg)
        print('error_sum:', error_sum)

    error_list = results['errors_list']

    error_list = [number ** 2 for number in error_list]

    error_sum = sum(error_list)

    return error_sum


def calc_new_estimates(level_past, slope_past, seasonal_past, alpha, beta, omega, gamma, e, weekly_transition_matrix, weekly_update_vector):

    """
    This function updates the state estimates of the ETS(M,Ad,M) model level_past, slope_past, seasonal_past by the innovations/errors
    of each period. It is a part of the loop of the fit calculator of the model. Note that it also moves up the dummies
    in the seasonality vector. The Inputs are all past states, the smoothing parameters and the weekly_transition_matrix and weekly_update_vector
    required to update the current dummy, put it at the bottom of the vector and move the dummies up one position each period.

    Parameters:

      Past state estimates:
      level_past = past level
      slope_past = past trend
      seasonal_past = past seasonal dummy vector

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      constant matrix and vector for the seasonality:
      weekly_transition_matrix = serves the purpose of pushing all dummies up one position
                                 while the current dummy goes to the bottom
      weekly_update_vector = is zero for all parameters except the current dummy in last position, which is updated by e.

    Return: The function returns the updated states:
      level = updated level
      slope = updated trend
      seasonal = updated seasonality vector
    """

    level = (level_past + omega * slope_past) * (1 + alpha * e)
    slope = omega * slope_past + beta * (level_past + omega * slope_past) * e
    seasonal = np.dot(weekly_transition_matrix,seasonal_past) + weekly_update_vector * gamma * e

    return level,slope,seasonal


def calc_error(level_past, slope_past, seasonal_past, omega, series, i, reg, exogen):

    """
    This function calculates the point forecast, the relativ and absolute forecasting error of the ETS(M,Ad,M) model.
    The Inputs are all past states and the trend dampening factor, the time point i, the time series to estimate y as well as  the
    exogen variables and their regressors.
    It is a part of the loop of the fit calculator of the model and thus time i dependent. The absolute errors are computed
    for the sum of squared errors. Note that the sum of squared errors could also be computed with the relativ errors.
    Here the exogen variables are included. In the computation of the forecast a term is added where the regressors are multiplied
    by their coefficients and the estimate purely based on the ETS Model. This allows us to have multiplicative effects of events
    without running into the issue of having an estimate of zero on days without events.

    Parameters:

      Past state estimates:
      level_past = past level
      slope_past = past trend
      seasonal_past = past seasonal dummy vector

      Time independent smoothing parameters:
      omega = trend dampening coefficient

      time dependent:
      series = time series
      i = current time point

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector for the exogen variables


    Return: The function returns the point forecast, the relativ and absolute error:
      estimate = point forecast
      e = relativ error
      e_absolute = absolute error
    """

    estimate = (level_past + omega * slope_past) * seasonal_past[0] + np.dot(reg,exogen.iloc[i]) * (level_past + omega * slope_past) * seasonal_past[0]
    e = (series[i] - estimate) /series[i]
    e_absolute = series[i] - estimate

    return estimate, e, e_absolute


def save_estimates(errors_list, point_forecast, level_list, slope_list, seasonal_list, e_absolute, estimate, level_past, slope_past, seasonal_past):

    """
    This function simply appends the state estimates, the point forecast and the absolute error of each period
    to previously defined lists in the ETS(M,Ad,M) model. The Inputs are all past states, the point forecast,
    the absolute error and their respective lists. It is a part of the loop of the fit calculator of the model.
    Sidenote: The Function has no difference to the model without exogen variables.

    Parameters:

      Past state estimates:
      level_past = past level
      slope_past = past trend
      seasonal_past = past seasonal dummy vector

      estimate = point forecast
      e_absolute = absolute error

      Lists accoridng to the above variables:
      errors_list
      point_forecast
      level_list
      slope_list
      seasonal_list

    Return: The function returns the updated Lists.
      errors_list
      point_forecast
      level_list
      slope_list
      seasonal_list
    """

    errors_list.append(e_absolute)
    point_forecast.append(estimate)
    level_list.append(level_past)
    slope_list.append(slope_past)
    seasonal_list.append(seasonal_past[0])

    return errors_list,point_forecast,level_list,slope_list,seasonal_list



def seasonal_matrices():

    '''
    This function simply defines the weekly transition matrix and weekly updating matrix needed in the computation of
    new weekly seasonality dummies. The function is part of the initialisation where it passes the matrices which are
    then used in the loop for the computation of new state estimates.
    Sidenote: The Function has no difference to the model without exogen variables.

    Parameters:
    Sidenote: It has no inputs although it can be made more general at which point it would include a scalar as input containing the
       length of the seasonality (here 7).

    Return: It returns the above weekly transition matrix and the weekly updating matrix:
       weekly_transition_matrix
       weekly_update_vector

    '''

    #defining weekly transition matrix:
    #1. defining first column of zeros (1 row to short)

    col_1 = np.vstack(np.zeros(6))

    #2. defining identity matrix 1 row and column to small

    col_2_6 = np.identity(6)

    #3. adding the 1 column and the identity matrix, now all states are updated to jump up one step in the state vector

    matrix_6 = np.hstack((col_1,col_2_6))

    #4. creating a final row in which the current state is put in last place and will be added by an update

    row_7 = np.concatenate((1,np.zeros(6)), axis = None)

    #5. adding the last row to the matrix to make it complete

    weekly_transition_matrix = np.vstack((matrix_6,row_7))

    #defining the weekly updating vector

    weekly_update_vector = np.vstack(np.concatenate((np.zeros(6),1), axis = None))

    return weekly_transition_matrix, weekly_update_vector


def ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen):

    '''
    This function computes the fit of the ETS(M,Ad,M) model with exogen variables for given initial and smoothing parameters.
    It is given these inputs by the model function and itself contains the functions calc_new_estimates,
    calc_error, save_estimates and seasonal_matrices. It first defines time t as the length of the series.
    Further it creates lists for parameters to return. Then it initialises by setting th initial states to be the past states.
    This allows the loop to start where one step ahead forecasts and errors are computed. This is followed by an update
    of the states with the new errors and then a redefinition of the states which in turn restarts the loop for the next period.

    Parameters:

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      Initial states computed above according to Hyndman 2008.
      level_initial = initial level
      slope_initial = initial trend
      seasonal_initial7 ... seasonal_initial = initial seasonal component where the number determines the lag of the dummy

      series = time series

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns lists of the fit errors, the point forecasts and the states:
        errors_list
        point_forecast
        level_list
        slope_list
        seasonal_list
    '''

    t = len(series)
    errors_list = list()
    point_forecast = list()
    level_list = list()
    slope_list = list()
    seasonal_list = list()

    #Initilaisation

    level_past = level_initial
    slope_past = slope_initial
    seasonal_past = seasonal_initial

    #defining the seasonal matrices for the calculation of new state estimates

    weekly_transition_matrix, weekly_update_vector = seasonal_matrices()

    for i in range(0,t):

        #compute one step ahead  forecast for timepoint i

        estimate, e, e_absolute = calc_error(level_past, slope_past, seasonal_past, omega, series, i, reg, exogen)

        #save estimation error for Likelihood computation as well as the states and forecasts (fit values)

        errors_list,point_forecast,level_list,slope_list,seasonal_list = save_estimates(errors_list,point_forecast,level_list,slope_list,seasonal_list,e_absolute,estimate,level_past,slope_past,seasonal_past)


        #Updating all state estimates with the information set up to time point i

        level,slope,seasonal = calc_new_estimates(level_past, slope_past, seasonal_past, alpha, beta, omega, gamma, e, weekly_transition_matrix, weekly_update_vector)

        #denote updated states from i as past states for time point i+1 in the next iteration of the loop

        level_past = level
        slope_past = slope
        seasonal_past = seasonal

    return  {'errors_list' : errors_list, 'point forecast' : point_forecast,'level_list' : level_list, 'slope_list' : slope_list, 'seasonal_list' : seasonal_list}


def fit_extracter(params, series, exogen):
    '''
    This function runs the optimal values threw the model to extract optimal (fitted) forecasts for the training data.
    In essence it is identical to the model function with the exception that it returns the full array of lists given by the
    ETS_M_Ad_M function.

    Parameters:

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      Initial states computed above according to Hyndman 2008.
      level_initial = initial level
      slope_initial = initial trend
      seasonal_initial7 ... seasonal_initial = initial seasonal component where the number determines the lag of the dummy

      regression:
      exog = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns the error, point forecast and states for every time point in separate lists:
      errors_list
      point forecast
      level_list
      slope_list
      seasonal_list
    '''

    #Note: the regression parameter has a variable length as due to the setting of before and after th enumber of exogen variables varies

    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    omega = params[3]
    level_initial = params[4]
    slope_initial = params[5]
    seasonal_initial = np.vstack(params[6:13])
    reg = (params[13:13+len(exogen.columns)])

    results = ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen)

    return results


def forecasting(params, exogen, h):
    '''
    This function runs the optimal values threw the model to extract optimal predictions for the evaluation data.
    In essence it is identical to the model function with the exception that it does not give the time series but the
    prediction horizon h. The computation of point forecast is done by passsing arguments to the ETS_M_Ad_M_forecast
    function below.

    Parameters:

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      last period T fit states computed above according to Hyndman 2008.
      level_initial = period T fit level
      slope_initial = period T fit trend
      seasonal_initial7 ... seasonal_initial_HM = period T fit seasonal component where the number determines the lag of the dummy

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns the point forecast and states for every time point in separate lists:
      point forecast
      level_list
      slope_list
      seasonal_list
    '''

    #Note: the regression parameter has a variable length as due to the setting of before and after th enumber of exogen variables varies

    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    omega = params[3]
    level_initial = params[4]
    slope_initial = params[5]
    seasonal_initial = np.vstack(params[6:13])
    reg = (params[13:13+len(exogen.columns)])

    results = ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,
          level_initial,slope_initial,seasonal_initial,reg,h,exogen)

    return results


def ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,
          level_initial,slope_initial,seasonal_initial,reg,h,exogen):
    '''
    The ETS_M_Ad_M_forecast forecast function computes the forecast h steps ahead. As errors e are unavailable in prediction,
    this simplifies the process compared to the ETS_M_Ad_M function. The ETS_M_Ad_M_forecast does not calculate new state estimates
    or errors but merely forecasts and updates the seasonal states.

    Parameters:

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      last period T fit states computed above according to Hyndman 2008.
      level_initial = period T fit level
      slope_initial = period T fit trend
      seasonal_initial7 ... seasonal_init = period T fit seasonal component where the number determines the lag of the dummy

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns the point forecast and states for every time point in separate lists:
      point forecast
      level_list
      slope_list
      seasonal_list
    '''

    #computing the number of time points as the length of the forecasting vector

    t = h
    point_forecast = list()
    level_list = list()
    slope_list = list()
    seasonal_list = list()

    #Initilaisation

    level_past = level_initial
    slope_past = slope_initial
    seasonal_past = seasonal_initial

    #defining the seasonal matrices for the calculation of new state estimates

    weekly_transition_matrix, weekly_update_vector = seasonal_matrices()

    for i in range(1,h+1):

        #compute one step ahead  forecast for timepoint t
        # Note: need -1 here exogen.iloc[i-1] as exogen contains the forecasting data and indexing starts at 0

        estimate = (level_past + omega * slope_past) * seasonal_past[0] + np.dot(reg,exogen.iloc[i-1]) * (level_past + omega * slope_past) * seasonal_past[0]

        point_forecast.append(estimate)
        level_list.append(level_past)
        slope_list.append(slope_past)
        seasonal_list.append(seasonal_past[0])

        #Updating
        #no changes in level (l) and slope (b) as they remain constant without new information
        #only changes in seasonality (s) as it cycles every 7 days, the effect of each individual seasonality is not updated

        seasonal_past = np.dot(weekly_transition_matrix,seasonal_past)

    return  {'point forecast' : point_forecast,
             'level_list' : level_list, 'slope_list' : slope_list, 'seasonal_list' : seasonal_list}




