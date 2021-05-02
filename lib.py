import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Input, Flatten, Concatenate, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from matplotlib import pyplot as plt
from scipy.stats import zscore
from pandas import DataFrame

def get_emb(feat_name, feat_size):
    embed_size = (feat_size + 1) // 2
    embed_size = 50 if (embed_size > 50) else embed_size
    inp = Input((1,), name=f'input_{feat_name}')
    out = Flatten(name=f'flatten_{feat_name}')(Embedding(feat_size+1, embed_size, name=f'embed_{feat_name}', mask_zero=False)(inp))
    return inp, out

def get_input_layers(all_features, feat_to_embed):
    inputs = []
    outs = []
    
    for f in all_features:
        if (f in feat_to_embed.keys()):
            emb = get_emb(f, feat_to_embed[f])
            inputs.append(emb[0])
            outs.append(emb[1])
        else:
            inp = Input(shape=(1,), name=f'input_{f}')
            inputs.append(inp)
            outs.append(inp)
    
    concat_out = Concatenate()(outs)
    return inputs, concat_out

def get_model(all_features, 
              feat_to_embed, 
              outputs=None, 
              num_layers=3, 
              hidden_units=500, 
              activation='relu', 
              lr=0.001, 
              batch_normalization=False, 
              dropout=0,
              input_dropout=0,
              l2_lambda=0,
              kernel_initializer='glorot_uniform',
              output_loss_weights = None
             ):
    inputs, concat = get_input_layers(all_features, feat_to_embed)
    
    out = concat
    if (input_dropout > 0):
        out = Dropout(input_dropout)(out)
    regularizer = l2(l2_lambda) if (l2_lambda > 0) else None
    for i in range(num_layers):
        out = Dense(hidden_units, kernel_initializer=kernel_initializer,  kernel_regularizer=regularizer)(out)
        if (batch_normalization):
            out = BatchNormalization()(out)
        if (dropout > 0):
            out = Dropout(dropout)(out)
        out = Activation(activation)(out)
    
    loss_weights = output_loss_weights if (output_loss_weights) else [1 for o in outputs]
    if (outputs):
        out_list = []
        for o in outputs:
            out_list.append(Dense(1, name=o, activation="sigmoid")(out))
        out = out_list
    else:
        out = Dense(1)(out, activation="sigmoid")
    
    model = Model(inputs, out)
    model.compile(optimizers.Adam(lr=lr), loss='mse', metrics=['mse', rmspe], loss_weights = loss_weights)
    return model, inputs, concat

def rmspe(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_true - y_pred) / y_true)))

def get_rmspe_score(y_true, y_pred):
    return np.sqrt(np.square((y_pred - y_true) / y_true).mean())

def light_gbm_rmspe_metric(y_true, y_pred):
    rmspe = np.sqrt(np.square((y_pred - y_true) / y_true).mean())
    return ("RMSPE", rmspe, False)

def plot_losses(history):    
    plt.figure(figsize=(20, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def join_df(left, right, left_on, right_on=None):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", "_y"))

def get_filtered_data(X, filter_outliers = False):
    result = X[X["Sales"] > 0]
    if (filter_outliers):
        z_scores = np.abs(zscore(result[['Sales', 'Customers']]))
        result = result[(z_scores < 3).all(axis=1)]
    return result

def get_missing_columns(df):
    return list(df.columns[df.describe(include = 'all').loc['count']<len(df)])

def generate_submit_file(test, pred_sales, fileName = 'rossman_submission.csv', correct_negatives = False, correct_closed_stores = False):
    submission = DataFrame()
    submission["Id"] = test["Id"]
    submission["Sales"] = pred_sales
    if (correct_negatives):
        submission.loc[submission['Sales'] < 0, ['Sales']] = 0
    if (correct_closed_stores):
        submission.loc[test['Open'] == 0, ['Sales']] = 0
    submission.to_csv(fileName, index=False)
    return submission