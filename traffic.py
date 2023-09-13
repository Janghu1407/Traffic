#####This notebook only contains the data-preprocessing and modelling, not the visualization part
########The solution is to take average of every 50 rows (without using time and file_id)-----
######and create sequences of 100 length to train lstm model----
########with 5 folds and use time in final catboost model to predict target
###NVIDIA Tesla P100-PCIE Driver Version: 515.48.07    
###CUDA Version: 11.7
###python version - 3.7
###pandas version - '1.3.5'
###numpy version - '1.21.6'
##tensorflow version - '2.9.1'
##sklearn version - '1.0.2'
##scipy version - '1.7.3'
##keras version - '2.9.0'
##catboost version - '0.25.1'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input,Activation, Flatten, TimeDistributed, Concatenate
from tensorflow.keras.layers import RepeatVector, Layer, BatchNormalization, Dropout,Bidirectional, GRU, Conv1D ,MaxPool1D
#from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
import random
import tensorflow as tf
tf.random.set_seed(2023)
np.random.seed(2023)
random.seed(2023)

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

import glob
path = "./Train_data/*.csv"
train = []
for fname in glob.glob(path):
    df = pd.read_csv(fname)
    file_id = fname.split('.')[0].split('/')[-1].split('n')[1]
    df['file_id'] = file_id
    train.append(df)

    
train = pd.concat(train)
train = train.reset_index(drop=True)

train['in_out'] = (1+train['portPktOut'])/(1+train['portPktIn'])
train['q_in'] = (1+train['qSize'])/(1+train['portPktIn'])

#####Code to rollup every bucket_size rows of data and create features based on Statistics
def transform_ts(ts, bucket_size):
    sample_size = ts.shape[0]
    # new_ts will be the container of the new data
    new_ts = []
    for i in range(0, sample_size, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts[i:i + bucket_size]
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std() # standard deviation
        std_top = mean + std # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentil_calc = np.percentile(ts_range, [0, 1,5, 10, 25, 50, 75,90, 95,99, 100]) 
        max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        peaks = len(find_peaks(ts_range)[0])
        sk = skew(ts_range)
        kurt = kurtosis(ts_range)
        #energy = np.sum(ts_range**2)/100
        #############fft based features
        #fft_ts = pd.Series(np.abs(np.fft.fft(ts_range))[1:26])
        #fft_mean = fft_ts.mean()
        #fft_std = fft_ts.std()
        #fft_top = fft_mean+fft_std
        #fft_bot = fft_mean-fft_std
        #percentil_calc_fft = np.percentile(fft_ts, [0, 1, 25, 50, 75, 99, 100]) 
        #max_range_fft = percentil_calc_fft[-1] - percentil_calc_fft[0] # this is the amplitude of the chunk
        #relative_percentile_fft = percentil_calc_fft - fft_mean
        #fft_peaks = len(find_peaks(fft_ts)[0])
        #fft_sk = skew(fft_ts)
        #fft_kurt = kurtosis(fft_ts)
        #####################indices based
        max_i = np.argmax(ts_range)
        min_i = np.argmin(ts_range)
        min_max_i = abs(max_i-min_i)
        #max_fft_i = np.argmax(fft_ts)
        #min_fft_i = np.argmin(fft_ts)
        #min_max_fft_i = np.abs(max_fft_i-min_fft_i)
        ###############
        #new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range, peaks, sk, kurt,max_i, min_i, min_max_i,
        #                                          fft_mean, fft_std, fft_top, fft_bot, max_range_fft, fft_peaks, fft_sk, fft_kurt,max_fft_i, min_fft_i, min_max_fft_i]),
        #                                            percentil_calc, relative_percentile,percentil_calc_fft, relative_percentile_fft]))
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range, peaks, sk, kurt,max_i, min_i, min_max_i]),
                                      percentil_calc, relative_percentile]))
    return np.asarray(new_ts)

##########Code to create new statistic cols for every column in train data
def stats(signal):
    cols = ['mean', 'std', 'std_top', 'std_bot', 'max_range', 'peaks','sk','kurt','max_i','min_i','min_max_i',
              'percetil_0', 'percentil_1','percentil_5', 'percentil_10','percentil_25', 'percentil_50', 'percentil_75', 'percentil_90',
            'percentil_95','percentil_99', 'percentil_100', 'rel_0', 'rel_1','rel_5','rel_10', 'rel_25', 'rel_50', 'rel_75','rel_90',
               'rel_95','rel_99', 'rel_100']
    
    new_cols = []
    for i in cols:
        new_cols.append(signal+'_'+i)
    return new_cols


############ Rolling up labels seperately -- 
######same as transform ts but for labels---
########using mode of every bucket_size rows-----
def sample_labels(inp_df, target, window_size):
    y_sampled = []
    #window = list(range(inp_df.shape[0] + window_size))[::window_size]
    sample_size = inp_df.shape[0]
    for x in range(0, sample_size, window_size):
        window_sample_y = target.iloc[x:x+window_size].tolist()
        window_sample_y = max(window_sample_y, key=window_sample_y.count)
        y_sampled.append(window_sample_y)
        
    return np.array(y_sampled)

########sampling time for 50(bucket_size) window--- taking first of every bucket
def sample_time(inp_df, target, window_size):
    y_sampled = []
    sample_size = inp_df.shape[0]
    for x in range(0, sample_size, window_size):
        window_sample_y = target.iloc[x]
        y_sampled.append(window_sample_y)
    return np.array(y_sampled)


##############Main functions for rolling up data (with use of sample_labels, transform_ts and stats function)
def roll_up(dataframe, window_size=50):
    new_df = pd.DataFrame()
    for i in ['portPktIn','portPktOut', 'qSize', 'in_out', 'q_in']:
        columns = stats(i)
        df = pd.DataFrame(transform_ts(dataframe[i], window_size),columns = columns)
        new_df = pd.concat([new_df, df], axis=1)
    if 'label' in dataframe.columns:
        new_df['label'] = sample_labels(dataframe, dataframe['label'], window_size)
    new_df['file_id'] = sample_labels(dataframe, dataframe['file_id'], window_size)
    return new_df

#########Functions to sample 100 length sequences from rolled up data
def sample(inp_df, target, window_size):
    x_sampled = []
    y_sampled = []
    window = list(range(inp_df.shape[0] + window_size))[::window_size]
    for x in range(len(window)-1):
        window_sample_x = inp_df.iloc[window[x]:window[x+1], :].values.tolist()
        x_sampled.append(window_sample_x)
        window_sample_y = target.iloc[window[x]:window[x+1]].tolist()
        y_sampled.append(window_sample_y)          
    return np.array(x_sampled), np.array(y_sampled)

########same function as above but for test when labels are not ther
def sample_oot(inp_df, window_size):
    x_sampled = []
    window = list(range(inp_df.shape[0]+window_size))[::window_size]
    for x in range(len(window)-1):
        window_sample_x = inp_df.iloc[window[x]:window[x+1], :].values.tolist()
        x_sampled.append(window_sample_x)
        return np.array(x_sampled)

#####LSTM based model
def model_seq(inp_shape):
    inp = Input(shape = (inp_shape[1], inp_shape[2]))
    l1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
    l4 = Bidirectional(GRU(128, return_sequences=True))(l1)
    l2 = Dense(256, activation='sigmoid')(l4)
    l5 = Dropout(0.2)(l2)
    l8 = Dense(128, activation='sigmoid')(l5)
    l6 = BatchNormalization()(l8)
    l3 = Dense(64, activation='sigmoid')(l6)
    l10 = TimeDistributed(Dense(12, activation='softmax'))(l3)
    model = Model(inp, l10)
    return model

###########Prediction function which converts predicted sequences into length of 100 again
def predicted_data(model, x_data, l):
    pred = model.predict(x_data)
    rows = x_data.shape[0]
    pred = pred.reshape(rows*l, 12)
    pred = pd.DataFrame(pred)
    pred.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    cols = pred.columns
    #pred['label'] = pred.apply(lambda x: cols[np.argmax([x[col] for col in cols])], axis = 1)
    return pred

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
from keras import backend as K
from keras.models import load_model

x_cols = ['portPktIn_mean', 'portPktIn_std', 'portPktIn_std_top',
    'portPktIn_std_bot', 'portPktIn_max_range', 'portPktIn_peaks',
    'portPktIn_sk', 'portPktIn_kurt', 'portPktIn_max_i',
    'portPktIn_min_i', 'portPktIn_min_max_i', 'portPktIn_percetil_0',
    'portPktIn_percentil_1', 'portPktIn_percentil_5',
    'portPktIn_percentil_10', 'portPktIn_percentil_25',
    'portPktIn_percentil_50', 'portPktIn_percentil_75',
    'portPktIn_percentil_90', 'portPktIn_percentil_95',
    'portPktIn_percentil_99', 'portPktIn_percentil_100',
    'portPktIn_rel_0', 'portPktIn_rel_1', 'portPktIn_rel_5',
    'portPktIn_rel_10', 'portPktIn_rel_25', 'portPktIn_rel_50',
    'portPktIn_rel_75', 'portPktIn_rel_90', 'portPktIn_rel_95',
    'portPktIn_rel_99', 'portPktIn_rel_100', 'portPktOut_mean',
    'portPktOut_std', 'portPktOut_std_top', 'portPktOut_std_bot',
    'portPktOut_max_range', 'portPktOut_peaks', 'portPktOut_sk',
    'portPktOut_kurt', 'portPktOut_max_i', 'portPktOut_min_i',
    'portPktOut_min_max_i', 'portPktOut_percetil_0',
    'portPktOut_percentil_1', 'portPktOut_percentil_5',
    'portPktOut_percentil_10', 'portPktOut_percentil_25',
    'portPktOut_percentil_50', 'portPktOut_percentil_75',
    'portPktOut_percentil_90', 'portPktOut_percentil_95',
    'portPktOut_percentil_99', 'portPktOut_percentil_100',
    'portPktOut_rel_0', 'portPktOut_rel_1', 'portPktOut_rel_5',
    'portPktOut_rel_10', 'portPktOut_rel_25', 'portPktOut_rel_50',
    'portPktOut_rel_75', 'portPktOut_rel_90', 'portPktOut_rel_95',
    'portPktOut_rel_99', 'portPktOut_rel_100', 'qSize_mean',
    'qSize_std', 'qSize_std_top', 'qSize_std_bot', 'qSize_max_range',
    'qSize_peaks', 'qSize_sk', 'qSize_kurt', 'qSize_max_i',
    'qSize_min_i', 'qSize_min_max_i', 'qSize_percetil_0',
    'qSize_percentil_1', 'qSize_percentil_5', 'qSize_percentil_10',
    'qSize_percentil_25', 'qSize_percentil_50', 'qSize_percentil_75',
    'qSize_percentil_90', 'qSize_percentil_95', 'qSize_percentil_99',
    'qSize_percentil_100', 'qSize_rel_0', 'qSize_rel_1', 'qSize_rel_5',
    'qSize_rel_10', 'qSize_rel_25', 'qSize_rel_50', 'qSize_rel_75',
    'qSize_rel_90', 'qSize_rel_95', 'qSize_rel_99', 'qSize_rel_100',
    'in_out_mean', 'in_out_std', 'in_out_std_top', 'in_out_std_bot',
    'in_out_max_range', 'in_out_peaks', 'in_out_sk', 'in_out_kurt',
    'in_out_max_i', 'in_out_min_i', 'in_out_min_max_i',
    'in_out_percetil_0', 'in_out_percentil_1', 'in_out_percentil_5',
    'in_out_percentil_10', 'in_out_percentil_25',
    'in_out_percentil_50', 'in_out_percentil_75',
    'in_out_percentil_90', 'in_out_percentil_95',
    'in_out_percentil_99', 'in_out_percentil_100', 'in_out_rel_0',
    'in_out_rel_1', 'in_out_rel_5', 'in_out_rel_10', 'in_out_rel_25',
    'in_out_rel_50', 'in_out_rel_75', 'in_out_rel_90', 'in_out_rel_95',
    'in_out_rel_99', 'in_out_rel_100', 'q_in_mean', 'q_in_std',
    'q_in_std_top', 'q_in_std_bot', 'q_in_max_range', 'q_in_peaks',
    'q_in_sk', 'q_in_kurt', 'q_in_max_i', 'q_in_min_i',
    'q_in_min_max_i', 'q_in_percetil_0', 'q_in_percentil_1',
    'q_in_percentil_5', 'q_in_percentil_10', 'q_in_percentil_25',
    'q_in_percentil_50', 'q_in_percentil_75', 'q_in_percentil_90',
    'q_in_percentil_95', 'q_in_percentil_99', 'q_in_percentil_100',
    'q_in_rel_0', 'q_in_rel_1', 'q_in_rel_5', 'q_in_rel_10',
    'q_in_rel_25', 'q_in_rel_50', 'q_in_rel_75', 'q_in_rel_90',
    'q_in_rel_95', 'q_in_rel_99', 'q_in_rel_100']


new_train = roll_up(train, 50)
rows = new_train.shape[0]
w=50
l = 100
rem = rows%l
if rem!=0:
    train_df = new_train.iloc[:-rem]
else:
    train_df = new_train

X_ = train_df[x_cols]
Y_ = train_df['label']
scaler.fit(X_)
X_scaled = pd.DataFrame(scaler.transform(X_), columns = X_.columns)
x_df, y_df = sample(X_scaled, Y_,l)

kf = KFold(n_splits = 5, shuffle = True,random_state = 2000)
iter=0
for train_index, val_index in kf.split(x_df):
    x_train = x_df[train_index]
    y_train = y_df[train_index]
    y_val = y_df[val_index]
    x_val = x_df[val_index]
    model = model_seq(x_train.shape)
    file_path = './kfold_{}_{}_final'.format(w, iter)
    mc =tf.keras.callbacks.ModelCheckpoint(monitor='val_sparse_categorical_accuracy', mode='max', verbose=1, patience=10,save_best_only=True,filepath=file_path)
    es=tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',patience=10)
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', 
              metrics = ['sparse_categorical_accuracy'])
    K.set_value(model.optimizer.learning_rate, 0.001)
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'])
    model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 50, batch_size=64, callbacks=[es,mc], verbose=1)
    iter = iter+1
    


def predictions(new_data):
    rows_data = new_data.shape[0]
    rem_data = rows_data%100
    if rem_data==0:
        data_df = new_data
    else:
        data_df = new_data.iloc[:-rem_data]
    X_data = data_df[x_cols]
    X_data_scaled = pd.DataFrame(scaler.transform(X_data), columns = X_data.columns)
    x_data_df = sample_oot(X_data_scaled,100)
    pred_data = []
    for iter in range(5):
        file_path = './kfold_{}_{}_final'.format(50, iter)
        model_l = load_model(file_path, compile=False)
        model_l.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
        pred_data.append(predicted_data(model_l, x_data_df, 100))   
    data_pred_df = pd.concat(pred_data)
    data_pred_df = data_pred_df.reset_index()
    data_predictions = data_pred_df.groupby('index').agg('mean')
    data_predictions.shape
    cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    data_predictions['Target'] = data_predictions.apply(lambda x: cols[np.argmax([x[col] for col in cols])], axis = 1)
    new_data['Target'] = data_predictions['Target'].astype('int')
    r_1  = rem_data+1
    new_data['Target'] = new_data['Target'].fillna(int(new_data.iloc[-r_1]['Target']))


test_id = random.sample(new_train['file_id'].unique().tolist(), 10)
new_test = new_train[new_train['file_id'].isin(test_id)]
new_train_1 = new_train[~new_train['file_id'].isin(test_id)]

val_id = random.sample(new_train_1['file_id'].unique().tolist(), 10)
new_val = new_train_1[new_train_1['file_id'].isin(val_id)]
new_train_1 = new_train_1[~new_train_1['file_id'].isin(val_id)]

new_train_1 = new_train_1.reset_index(drop=True)
new_val = new_val.reset_index(drop=True)
new_test = new_test.reset_index(drop=True)

predictions(new_train_1)
predictions(new_val)
predictions(new_test)

##to free up memory used
from numba import cuda 
device = cuda.get_current_device()
device.reset()

cb_cols = x_cols.append('time')
cb_cols.append('Target')

from catboost import Pool, CatBoostClassifier
model_cb = CatBoostClassifier(iterations=2000, 
                               learning_rate = 0.05,
                               depth= 6,
                               #eval_metric="AUC",
                               #loss_function="Logloss",
                               random_seed=2,
                               #od_wait=100,
                               od_type="Iter",
                               max_ctr_complexity=1,
                               boosting_type= "Plain",
                               one_hot_max_size=2,
                               bagging_temperature=1,
                               random_strength=1,
                               task_type='GPU'
                               )


model_cb = model_cb.fit(new_train_1[cb_cols], new_train_1['label'], verbose=True, 
                         eval_set=[(new_val[cb_cols], new_val['label'])],early_stopping_rounds=100)

val_pred = model.predict(new_val[cb_cols])
train_pred = model.predict(new_train_1[cb_cols])
test_pred = model.predict(new_test[cb_cols])

val_out = [val_pred[i][0] for i in range(len(val_pred))]
train_out = [train_pred[i][0] for i in range(len(train_pred))]
test_out = [test_pred[i][0] for i in range(len(test_pred))]

import glob
path = "./Test_data/*.csv"
oot = []
for fname in glob.glob(path):
    df = pd.read_csv(fname)
    file_id = fname.split('.')[0].split('/')[-1].split('n')[1]
    df['file_id'] = file_id
    oot.append(df)

    
oot = pd.concat(oot)
oot = oot.reset_index(drop=True)
oot['in_out'] = (1+oot['portPktOut'])/(1+oot['portPktIn'])
oot['q_in'] = (1+oot['qSize'])/(1+oot['portPktIn'])

new_oot = roll_up(oot, 50)
predictions(new_oot)
oot_pred = model.predict(new_oot[x_cols])
oot_out = [oot_pred[i][0] for i in range(len(oot_pred))]
new_oot['label'] = oot_out
oot['Target'] = [int(x) for x in new_oot['label'] for i in range(50)]

def id(row):
    return 'test'+str(row.file_id).split('.')[0]+'_'+str(row.time).split('.')[0]

oot['ID'] = oot.apply(id, axis=1)
oot[['ID', 'Target']].to_csv('./kf_5.csv', index=False)
