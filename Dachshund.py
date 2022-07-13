import numpy as np
import pandas as pd

nInst=100
currentPos = np.zeros(nInst)


def getMyPosition (prcSoFar):
    global currentPos

    # hardcoded variables
    max_size = 10000

    # dataframe
    df = pd.DataFrame(prcSoFar.T)
    df = df.reset_index()
    df = df.rename(columns={'index': 'time'})

    ########
    # pers #
    ########
    pers_timeframe = 50
    pers_size = 0.3
    data_ = pd.melt(df, id_vars=['time'], value_vars=list(range(100)))
    data_ = data_.rename(columns={'variable': 'stock', 'value': 'price'})
    # calc pers
    data_['prev_price'] = data_.groupby('stock')['price'].shift(1)
    data_ = data_.assign(price_move=data_['price'] - data_['prev_price'])
    data_['pers'] = data_.groupby('stock')['price_move'].rolling(pers_timeframe).apply(persistence, raw=True, args=(1,)).reset_index(level=0, drop=True)
    # calc signal1
    data_ = data_.assign(signal1=np.where(data_['pers']<=-pers_size,-1,0))
    data_ = data_.assign(signal1=np.where(data_['pers']>= pers_size,1,data_['signal1']))

    #######
    # rsi #
    #######
    rsi_timeframe = 20
    # short
    rsi_open_short = 20
    rsi_close_short = 50
    # long
    rsi_open_long = 80
    rsi_close_long = 50
    # calc RSI
    data_ = data_.assign(u=np.where(data_['price_move'] > 0, data_['price_move'], 0))
    data_ = data_.assign(d=np.where(data_['price_move'] < 0, abs(data_['price_move']), 0))
    data_['avg_u'] = data_.groupby('stock')['u'].rolling(rsi_timeframe).mean().reset_index(level=0, drop=True)
    data_['avg_d'] = data_.groupby('stock')['d'].rolling(rsi_timeframe).mean().reset_index(level=0, drop=True)
    data_['rsi'] = 100 - (100 / (1 + (data_['avg_u'] / data_['avg_d'])))
    data_rsi_ = pd.DataFrame()
    latest_time = data_['time'].max()
    for i in range(0, 100):
        data_sub_ = data_[(data_['stock']==i)&(data_['time']>=(latest_time-50))].copy()
        data_sub_ = data_sub_.reset_index(drop=True)
        data_sub_['signal2'] = 0
        for idx, row in data_sub_.iterrows():
            if (data_sub_.at[idx, 'rsi']<rsi_open_short) or ((idx != 0) and (data_sub_.at[idx-1, 'signal2'] == -1) and (data_sub_.at[idx, 'rsi']<rsi_close_short)):
                data_sub_.at[idx,'signal2'] = -1
            elif (data_sub_.at[idx, 'rsi']>rsi_open_long) or ((idx != 0) and (data_sub_.at[idx-1, 'signal2'] == 1) and (data_sub_.at[idx, 'rsi']>rsi_close_long)):
                data_sub_.at[idx, 'signal2'] = 1
            elif (idx != 0) and (data_sub_.at[idx-1, 'signal2'] == -1) and (data_sub_.at[idx, 'rsi']>rsi_close_short):
                data_sub_.at[idx, 'signal2'] = 0
            elif (idx != 0) and (data_sub_.at[idx-1, 'signal2'] == 1) and (data_sub_.at[idx, 'rsi']<rsi_close_long):
                data_sub_.at[idx, 'signal2'] = 0
        data_sub_ = data_sub_[['stock', 'time', 'signal2']]
        data_rsi_ = data_rsi_.append(data_sub_, sort=False)

    # merge pers with rsi signal, net the signals, cap at -1 and +1
    data_ = data_.merge(data_rsi_, how='left',on=['stock','time'])
    data_['signal2'] = data_['signal2'].fillna(0)
    data_['signal'] = data_['signal1'] + data_['signal2']
    data_['signal'] = data_['signal'].clip(lower=-1, upper=1)

    # calc shares
    data_['shares'] = round(data_['signal']*max_size/data_['price'])
    data_['shares'] = data_['shares'].astype(int)
    # output last day
    latest_time = data_['time'].max()
    currentPos = data_[data_['time']==latest_time]['shares'].to_list()

    return currentPos


def persistence(values, t):
    values = values[0::t]  # only look at every t element in the array
    T = len(values)
    num_up = len(values[np.where(values>0)])  # count number of up periods
    num_down = len(values[np.where(values<0)])  # count number of up periods
    pers = (num_up-num_down)/T
    return pers
