import pandas as pd
import numpy as np
from flask import Flask
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from variables import *

np.random.seed(seed)
tf.random.set_seed(seed)

def get_dataframe(past_series_str, window_size, ma_periods, scaler):
    past_series = np.array(past_series_str.split(','))
    past_series_len = len(past_series)
    if (past_series_len != window_size + ma_periods):
        raise Exception(f'past_series_len != window_size + ma_periods. {past_series_len} != {window_size} + {ma_periods}')
    past_series = past_series.astype(np.float)
    df = pd.DataFrame({'OCAvg' : past_series})
    df['MA'] = df['OCAvg'].rolling(window=ma_periods).mean() # Simple Moving Average
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Returns'] = np.log(df['MA']/df['MA'].shift(1)) # Log Returns
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Scaled'] = scaler.transform(df[['Returns']].values)
    return df

app = Flask(__name__)

@app.route('/predict/<string:ticker>/<int:batch_size>/<int:window_size>/<int:ma_periods>/<float:abs_pips>/<int:pred_size>/<string:instance>/<string:series>', methods=['GET'])
def predict(ticker, batch_size, window_size, ma_periods, abs_pips, pred_size, instance, series):

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = get_dataframe(series, window_size, ma_periods, scaler)

    y_ma = float(df['MA'].iloc[-1])
    top_price = y_ma + abs_pips
    bottom_price = y_ma - abs_pips

    X = [df['Scaled'].values]
    y = []
    sum = 0
    for _ in range(pred_size):
        X = np.asarray(X)
        X = np.reshape(X, (1, window_size, 1))
        y_pred_scaled = model.predict(X)
        y_return = scaler.inverse_transform(y_pred_scaled)
        y_ma = y_ma * np.exp(y_return)

        if (y_ma >= top_price):
            sum += 1

        if (y_ma <= bottom_price):
            sum -= 1

        y.append(float(y_ma))
        # Remove first item in the list
        X = np.delete(X, 0)
        # Add the new prediction to the end
        X = np.append(X, y_pred_scaled)

    return str(sum)
    

if __name__ == '__main__':
    app.run()