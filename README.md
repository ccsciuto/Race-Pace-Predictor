# Race-Pace-Predictor
This script uses ARIMA time series forecasting model to predict estimated power for a given run on a certain date
Power is determined by takeing the AVG speed of a run and dividing it by the AVG heart rate, multiplying by the dewpoint+50, then multiplying by 10
Once an estimated Power is determined for a certain date, you can determine what your speed will be by diving power by 10, dividing by the predicted dew point of that day+50 then multiplying by expected avg HR. 
