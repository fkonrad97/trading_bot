import backtrader as bt
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import math
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
from variables import *

class Strat(bt.Strategy):
    params = dict( 
        stop_loss=0.02,
        trail=False,
    )

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS {0:8.5f}, NET {1:8.5f}, PORTFOLIO_VAL {2:8.5f}'.format(
            trade.pnl, trade.pnlcomm, cerebro.broker.getvalue()))
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self.log('ORDER ACCEPTED/SUBMITTED')
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                 self.log('BUY EXECUTED, Price: {0:8.5f}, Cost: {1:8.5f}, Comm: {2:8.5f}, ID: {3:8.1f}'.format(
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm,
                    order.ref))
                
            elif order.issell():
                self.log('SELL EXECUTED, {0:8.5f}, Cost: {1:8.5f}, Comm{2:8.5f}, ID: {3:8.1f}'.format(
                    order.executed.price, 
                    order.executed.value,
                    order.executed.comm,
                    order.ref))

        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.order = None
            return

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.datetime().strftime('%a %Y-%m-%d %H:%M:%S')
        print('%s, %s' % (dt, txt))

    def __init__(self):
        print("Start")
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.df = pd.DataFrame(columns=['Open','Close'])
        self.iter = 0
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.order = None
        self.stop_price = None

    def next(self):
        ''' STOP/LOSS'''
        if self.order:
            if self.order.isbuy():
                if self.dataclose[0] <= self.stop_price:
                    self.close()
                else:
                    self.stop_price = self.dataclose[0] * (1.0 - self.p.stop_loss)
                    return
            elif self.order.issell():
                if self.dataclose[0] >= self.stop_price:
                    self.close()
                else:
                    self.stop_price = self.dataclose[0] * (1.0 + self.p.stop_loss)
                    return

        if self.iter < window_size + ma_periods:
            self.df = self.df.append({'Open':self.dataopen[0], 'Close':self.dataclose[0]}, ignore_index=True)

        if(self.iter == window_size + ma_periods):
            self.df['OCAvg'] = self.df['Open'].add(self.df['Close']).div(2)
            self.df['MA'] = self.df['OCAvg'].rolling(window=ma_periods).mean()
            self.df['Returns'] = np.log(self.df['MA']/self.df['MA'].shift(1))

        if self.iter > window_size + ma_periods:
            self.df_ = self.df.copy()
            self.df_.dropna(how='any', inplace=True)
            self.df_.reset_index(drop=True, inplace=True)
            self.df_['Scaled'] = self.scaler.transform(self.df_[['Returns']].values)
            
            self.y_ma = float(self.df_['MA'].iloc[-1])
            self.top_price = self.y_ma + abs_pips
            self.bottom_price = self.y_ma - abs_pips

            self.X = [self.df_['Scaled'].values]
            self.y = []

            sum = 0
            for _ in range(pred_size):
                self.X = np.asarray(self.X)
                self.X = np.reshape(self.X, (1, window_size, 1))
                self.y_pred_scaled = self.model.predict(self.X)
                self.y_return = self.scaler.inverse_transform(self.y_pred_scaled)
                self.y_ma = self.y_ma * np.exp(self.y_return) # Reverse Log Returns
            
                if (self.y_ma >= self.top_price):
                    sum += 1
                elif (self.y_ma <= self.bottom_price):
                    sum -= 1

                self.y.append(float(self.y_ma))
                self.X = np.delete(self.X, 0)
                self.X = np.append(self.X, self.y_pred_scaled)

            print("Sum: ", sum)

            if self.order is None:
                if sum > 0:
                    self.log('BUY CREATE {0:8.5f}, VALUE: {1:8.5f}'.format(self.dataclose[0], cerebro.broker.getvalue() * 0.3))
                    self.stop_price = self.dataclose[0] * (1.0 - self.p.stop_loss)
                    self.order = self.buy()
                elif sum < 0:
                    self.log('SELL CREATE {0:8.5f}'.format(self.dataclose[0]))
                    self.stop_price = self.dataclose[0] * (1.0 + self.p.stop_loss)
                    self.order = self.sell()

            self.df = self.df.iloc[1:]
            self.df = self.df.append({'Open':self.dataopen[0], 
                            'Close':self.dataclose[0], 
                            'OCAvg': (self.dataopen[0] + self.dataclose[0])/2,
                            }, ignore_index=True)
            self.df['MA'] = self.df['OCAvg'].rolling(window=ma_periods).mean()
            self.df['Returns'] = np.log(self.df['MA']/self.df['MA'].shift(1))
        self.iter += 1

dataframe = pd.read_csv('data/btc19-20.csv', usecols=['Date','Open','Close'], 
    index_col=['Date'], parse_dates=['Date'])

print(dataframe.head())

cerebro = bt.Cerebro()
cerebro.addstrategy(Strat)
cerebro.broker.setcommission(commission=0.001)

data = bt.feeds.PandasData(dataname=dataframe)

cerebro.adddata(data)

cerebro.broker.setcash(1000.0)

print('Starting Portfolio Value: %.5f' % cerebro.broker.getvalue())

cerebro.run()

print('Final Portfolio Value: %.5f' % cerebro.broker.getvalue())

b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
cerebro.plot(b)