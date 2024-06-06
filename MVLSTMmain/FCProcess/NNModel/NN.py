import os
import pickle

#import tensorflow.python.keras

from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM, SimpleRNN, GRU
#from tensorflow.python.keras.optimizers import RMSprop,Adam,Nadam
from tensorflow.keras.optimizers import RMSprop,Adam,Nadam
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
#from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization


import pydotplus as pydot
from tensorflow.python.keras.utils.vis_utils import plot_model

def VLSTM_Affine(train,label,hyperparam,tensorboard_path,model_Input_Output):
    try:
        #LSTMモデル構築
        #ここはKeras documentationに詳しくあるので、そちらを参照
        model = Sequential()

        #model = make_RNNmodel2(model,hyperparam,train,model_Input_Output)  #RNNモデル作成
        #model = make_GRUmodel2(model,hyperparam,train,model_Input_Output)  #GRUモデル作成
        model = make_LSTMmodel3(model,hyperparam,train,model_Input_Output) #LSTMモデル作成[シンプルだし精度高いからこれで良さそう]

        #最適化問題を解くときに使う関数
        if   hyperparam["optimizer"] == "RMSprop": optimizer=RMSprop()
        elif hyperparam["optimizer"] == "Adam"   : optimizer=Adam()
        elif hyperparam["optimizer"] == "Nadam"  : optimizer=Nadam()
        else                                     : optimizer=None
        assert optimizer is not None,{"Hyper parameter's optimizer is not set"}


        #モデルをコンパイル
        model.compile(loss="mean_squared_error",optimizer=optimizer)
        #model.compile(loss="mean_absolute_error",optimizer=optimizer)
        model.summary()

        plot_model(model,to_file='VLSTMmodel.png',show_shapes=True,show_layer_names=False)

        #誤差(損失)が前回のエポック時よりも大きくなった場合、学習を早く止める関数
        #patienceは、最低でも指定した値のエポックまでは学習する。その後は指定しなかった場合と同じ
        #modeのmax,minは監視する値の増加か減少かで訓練を終了させる。autoは増減の傾向を自動で推定して訓練を終了する。
        early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=2,verbose=1)

        #最良のモデルを上書きで保存していく
        model_path = tensorboard_path + "/bestmodel.hdf5"
        model_checkpoint = ModelCheckpoint(filepath=model_path,
                                           monitor ='val_loss',
                                           save_best_only=True)

        #前のモデルより精度が良い場合、そのモデルに別個に名前を付けて保存
        #ファイル名が多様で、後でmodelの読み込みができていないからその部分を書いてから利用すること[20221216]
        #model_checkpoint = ModelCheckpoint(filepath=tensorboard_path + "/model-{epoch:02d}-{val_loss:.2f}.h5",
        #                                   monitor ='val_loss',
        #                                   save_best_only=True)


        """
        #Tensorboardによる可視化のためにログを吐き出す
        tensorboard_path = tensorboard_path + "/TensorBoard.logs/"
        tensor_board = TensorBoard(log_dir=tensorboard_path,
                                   histogram_freq=5,
                                   batch_size=(None,train.shape[1],train.shape[2]),
                                   write_graph=True,
                                   write_grads=True,
                                   write_images=True)
        """
        #実際に学習をする。epochsはエポックで、学習1回の単位のこと。historyには学習した時の損失値データ等が入っている。
        #batch_sizeは、一度に処理するデータ量を決められる。小さくするほど、1回のepochで大きく学習する
        history = model.fit(train,
                            label,
                            batch_size       = hyperparam["batch_size"],
                            shuffle          = True,
                            epochs           = hyperparam["epoch"],
                            validation_split = 0.2,
                            callbacks        = [early_stopping, model_checkpoint]
        )

        #最良のモデルを読み込む
        model = load_model(filepath=model_path)

        return model,history
    except KeyError as e:
        raise("{} keyword is missing from the hyperparameter".format(e.args))


def make_LSTMmodel1(model,hyperparam,train,model_Input_Output):
    '''
    2022年度卒研で使ったモデル[鉢嶺]
    '''
    model.add(LSTM(hyperparam["layerH_unit"],
                batch_input_shape=(None,train.shape[1],train.shape[2],),
                return_sequences=True))
    model.add(LSTM(hyperparam["layerH_unit"],return_sequences=True))
    model.add(LSTM(hyperparam["layerH_unit"]))
    model.add(Dropout(rate=hyperparam["dropout_rate"]))
    model.add(Dense(hyperparam["layerH_unit"]))
    model.add(Activation("sigmoid"))
    model.add(Dropout(rate=hyperparam["dropout_rate"]))
    model.add(Dense(hyperparam["layerH_unit"]))
    model.add(Activation("sigmoid"))
    model.add(Dropout(rate=hyperparam["dropout_rate"]))#ここ
    model.add(BatchNormalization())
    model.add(Dense(model_Input_Output))#
    model.add(Activation("sigmoid"))

    return model

def make_RNNmodel1(model,hyperparam,train,model_Input_Output):
    '''
    RNNモデルを作成する
    元あったやつ
    '''
    model.add(SimpleRNN(hyperparam["layerH_unit"], activation='linear', batch_input_shape=(None, train.shape[1], train.shape[2])))
    model.add(Dense(hyperparam["layerH_unit"], activation='linear'))
    model.add(Dense(model_Input_Output, activation='linear'))

    return model

def make_RNNmodel2(model,hyperparam,train,model_Input_Output):
    '''
    RNNモデルを作成する
    '''
    model.add(SimpleRNN(hyperparam["layerH_unit"], batch_input_shape=(None, train.shape[1], train.shape[2])))
    #model.add(Dense(hyperparam["layerH_unit"], activation='linear'))
    model.add(Dense(model_Input_Output, activation='linear'))

    return model

def make_GRUmodel1(model,hyperparam,train,model_Input_Output):
    '''
    GRUモデルを作成する.
    元あったやつ
    '''
    model.add(GRU(hyperparam["layerH_unit"], activation='linear', batch_input_shape=(None, train.shape[1], train.shape[2])))
    model.add(Dense(hyperparam["layerH_unit"], activation='linear'))
    model.add(Dense(model_Input_Output, activation='linear'))

    return model

def make_GRUmodel2(model,hyperparam,train,model_Input_Output):
    '''
    GRUモデルを作成する
    '''
    model.add(GRU(hyperparam["layerH_unit"], batch_input_shape=(None, train.shape[1], train.shape[2])))
    #model.add(Dense(hyperparam["layerH_unit"], activation='linear'))
    model.add(Dense(model_Input_Output, activation='linear'))

    return model


def make_LSTMmodel2(model,hyperparam,train,model_Input_Output):

    model.add(LSTM(hyperparam["layerH_unit"],
                batch_input_shape=(None,train.shape[1],train.shape[2],),
                return_sequences=True))
    model.add(LSTM(hyperparam["layerH_unit"],return_sequences=True))
    model.add(LSTM(hyperparam["layerH_unit"]))
    model.add(Dropout(rate=hyperparam["dropout_rate"]))
    model.add(Dense(hyperparam["layerH_unit"]))
    model.add(Activation("tanh"))
    model.add(Dropout(rate=hyperparam["dropout_rate"]))
    model.add(Dense(hyperparam["layerH_unit"]))
    model.add(Activation("relu"))
    model.add(Dropout(rate=hyperparam["dropout_rate"]))#ここ
    model.add(BatchNormalization())
    model.add(Dense(model_Input_Output))
    model.add(Activation("linear"))

    return model

def make_LSTMmodel3(model,hyperparam,train,model_Input_Output):
    '''
    LSTMモデルを作成する
    やけに精度が良いモデル。シンプルisベストかもしれない。
    '''
    model.add(LSTM(hyperparam["layerH_unit"], batch_input_shape=(None, train.shape[1], train.shape[2])))
    #model.add(Dense(hyperparam["layerH_unit"], activation='relu'))
    model.add(Dense(model_Input_Output, activation='linear'))

    return model