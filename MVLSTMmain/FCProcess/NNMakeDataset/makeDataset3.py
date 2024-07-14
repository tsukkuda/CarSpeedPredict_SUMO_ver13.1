import sys
import os
import pickle
import numpy as np
import random
import pandas as pd
from joblib import Parallel, delayed
from ReadData import tqdm_joblib as tj

#ニューラルネットワーク(NN)に入力するデータセットを作る
#ここでは現在のデータ～n期前のデータを入力とし、1期先の値を出力としている(問題設定より)
#引数はrawdata(2列のseriesデータ),maxlen(予測するデータ長),rows(予測に使うデータの行数)
#戻り値はdataset(5階行列(テンソル)、(シーケンス長,訓練データ長,channel長))と
#targetset(1階行列、訓練データの最後の値の次の値)となる。
def ConvLSTM2Ddataset(rawdata,maxlen,rows):
    def preData(rawdata,maxlen,rows):
        #DataFrame型からlist型に変換
        def timesCreate(rawdata,rows):
            timesdata = [np.array(rawdata[j:j+rows]) for j in range(len(rawdata)-rows)]
            return timesdata

        data = [np.array(timesCreate(rawdata,rows)[i:i+maxlen]) for i in range(len(rawdata)-(maxlen+rows))]
        #numpy.arrayで配列を作る。reshapeメソッド(関数ではない)で目的の配列に整形している
        dataset = np.array(data).reshape((len(data), maxlen, rows, len(rawdata.columns), 1))

        return dataset

    convTraindata = [preData(rawdata[num],maxlen,rows) for num in range(0,len(rawdata))]
    #train:訓練データ,label:ラベル
    traintemp  = [convTraindata[num][0:convTraindata[num].shape[0]-1,:,:,:,:] for num in range(0,len(convTraindata))]
    targettemp = [convTraindata[num][1:convTraindata[num].shape[0],-1,:,:,:] for num in range(0,len(convTraindata))]

    trainset  = np.concatenate(traintemp,axis=0)
    targetset = np.concatenate(targettemp,axis=0)

    return trainset,targetset

def VLSTMdataset(rawdata,maxlen):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(0,len(rawdata)):
        traintemp  = [rawdata[i][j:j+maxlen] for j in range(0,len(rawdata[i])-maxlen)]
        targettemp = [rawdata[i][j+maxlen,0] for j in range(0,len(rawdata[i])-maxlen)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(len(rawdata[i])-maxlen,maxlen,rawdata[i].shape[1]))
        targetset.append(np.array(targettemp).reshape(len(rawdata[i])-maxlen, 1))

    #リストごとに分けられている3階ndarrayを第一次元を軸に連結
    #trainset = np.concatenate(trainset,axis=0)
    #targetset = np.concatenate(targetset,axis=0)
    return trainset,targetset

def OutputManylocVLSTMdataset(rawdata,maxlen):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(0,len(rawdata)):
        traintemp  = [rawdata[i][j:j+maxlen] for j in range(0,len(rawdata[i])-maxlen)]
        targettemp = [rawdata[i][j+maxlen]   for j in range(0,len(rawdata[i])-maxlen)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(len(rawdata[i])-maxlen,maxlen,rawdata[i].shape[1]))
        targetset.append(np.array(targettemp).reshape(len(rawdata[i])-maxlen,rawdata[i].shape[1]))

    return trainset,targetset

def LSTMdataset(rawdata,maxlen):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(0,len(rawdata)):
        traintemp  = [rawdata[i][j:j+maxlen,0] for j in range(0,len(rawdata[i])-maxlen)]
        targettemp = [rawdata[i][j+maxlen,0] for j in range(0,len(rawdata[i])-maxlen)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(len(rawdata[i])-maxlen,maxlen,1))
        targetset.append(np.array(targettemp).reshape(len(rawdata[i])-maxlen, 1))

    #リストごとに分けられている3階ndarrayを第一次元を軸に連結
    #trainset  = np.concatenate(trainset,axis=0)
    #targetset = np.concatenate(targetset,axis=0)
    return trainset,targetset

def DownstLSTMdataset(rawdata,maxlen):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(0,len(rawdata)):
        traintemp  = [rawdata[i][j:j+maxlen,1] for j in range(0,len(rawdata[i])-maxlen)]
        targettemp = [rawdata[i][j+maxlen,0] for j in range(0,len(rawdata[i])-maxlen)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(len(rawdata[i])-maxlen,maxlen,1))
        targetset.append(np.array(targettemp).reshape(len(rawdata[i])-maxlen, 1))

    #リストごとに分けられている3階ndarrayを第一次元を軸に連結
    #trainset  = np.concatenate(trainset,axis=0)
    #targetset = np.concatenate(targetset,axis=0)
    return trainset,targetset

def DownstMLSTMdataset(rawdata,maxlen,predict_steps):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen
        #テストデータを作成する
        traintemp  = [rawdata[i][j:j+maxlen,1] for j in range(sample_size)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(sample_size,maxlen,1))

        #教師データを作成する
        monotarget = []
        for j in range(sample_size):
            #範囲を超えた部分は0で埋める
            targettemp = rawdata[i][j+maxlen:j+maxlen+predict_steps,0]
            if targettemp.shape[0] < predict_steps:
                zero_temp = np.zeros(predict_steps-targettemp.shape[0])
                targettemp = np.append(targettemp,zero_temp)

            monotarget.append(targettemp)

        targetset.append(np.array(monotarget).reshape(sample_size,predict_steps))

    return trainset,targetset

def MVLSTMdataset(rawdata,maxlen,predict_steps):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen
        #テストデータを作成する
        traintemp  = [rawdata[i][j:j+maxlen] for j in range(sample_size)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(sample_size,maxlen,rawdata[i].shape[1]))

        #教師データを作成する
        monotarget = []
        for j in range(sample_size):
            #範囲を超えた部分は0で埋める
            targettemp = rawdata[i][j+maxlen:j+maxlen+predict_steps,0]
            if targettemp.shape[0] < predict_steps:
                zero_temp = np.zeros(predict_steps-targettemp.shape[0])
                targettemp = np.append(targettemp,zero_temp)

            monotarget.append(targettemp)

        targetset.append(np.array(monotarget).reshape(sample_size,predict_steps))

    return trainset,targetset

def VarStepLSTMdataset(rawdata,maxlen,stepnum):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen-stepnum
        traintemp  = [rawdata[i][j:j+maxlen,0] for j in range(sample_size)]
        targettemp = [rawdata[i][j+maxlen+stepnum,0] for j in range(sample_size)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(sample_size,maxlen,1))
        targetset.append(np.array(targettemp).reshape(sample_size,1))

    return trainset,targetset

def VarStepDownstLSTMdataset(rawdata,maxlen,stepnum):
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen-stepnum
        traintemp  = [rawdata[i][j:j+maxlen,1] for j in range(sample_size)]
        targettemp = [rawdata[i][j+maxlen+stepnum,0] for j in range(sample_size)]
        #numpy.arrayで配列を作る。reshapeメソッドでlen(data),maxlen,1の3次元配列に整形している
        trainset.append(np.array(traintemp).reshape(sample_size,maxlen,1))
        targetset.append(np.array(targettemp).reshape(sample_size,1))

    return trainset,targetset

def VarStepVLSTMdataset(rawdata,maxlen,stepnum):#(生データ,学習データステップ数,1ステップの大きさ)
    '''
    欠損値を含む入力データ・教師データを除外するように学習データを成型
    '''
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen-stepnum
        traintemp,targettemp = [],[]
        new_sample_size = 0 #欠損無データを数える用
        for j in range(sample_size):
            #入力データである自車速度と平均速度の15ステップ配列rawdata[i][j:j+maxlen]のどちらの縦列内にもNoneがない
            #かつ
            #教師データの自車速度rawdata[i][j+maxlen+stepnum,0]がNoneではない
             if all(~np.isnan(rawdata[i][j:j+maxlen]).any(axis=0)) and all(~np.isnan(rawdata[i][j+maxlen+stepnum-1])):
                traintemp.append(rawdata[i][j:j+maxlen]) #windowsize分のデータ長
                targettemp.append(rawdata[i][j+maxlen+stepnum-1]) #windowsizeデータのstep個分先のデータ
                new_sample_size = new_sample_size + 1 #欠損データを棄却したので、sample_sizeは元より小さくなっている。

        #numpy.arrayで配列を作る。
        trainset.append(np.array(traintemp).reshape(new_sample_size,maxlen,rawdata[i].shape[1]))#(学習データ数,入力ステップ数,入力次元数)
        targetset.append(np.array(targettemp).reshape(new_sample_size,rawdata[i].shape[1]))#(学習データ数,出力次元数)

    return trainset,targetset


def VarStepVLSTMdataset4(rawdata,maxlen,stepnum,whole_data,train_date):#(生データ,学習データステップ数,1ステップの大きさ,所望学習データ数,所望学習日数)
    '''
    2入力2出力用学習データ形成プログラム
    欠損値を含む入力データ・教師データを除外するように学習データを成型
    '''
    #1日当たりの必要学習データ数
    need_data_num = whole_data//train_date

    #車のデータの日付が同じか判断する変数。
    judge_date = -1

    #DataFrame型からlist型に変換
    trainset,targetset = [],[]#学習データ数を確定してから1日分をこの配列に結合
    trainset_sub,targetset_sub = [],[]#学習データを間引く前の1日分の学習データ
    total_sample_size = 0
    sub_sample_size = 0
    trainset_dict = {}#日付がキーの辞書。同じ日付のものを配列にまとめる。入力データ
    targetset_dict = {}#日付がキーの辞書。同じ日付のものを配列にまとめる。教師データ
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen-stepnum
        traintemp,targettemp = [],[]
        new_sample_size = 0 #欠損無データを数える用
        for j in range(sample_size):
            #入力データである自車速度と平均速度の15ステップ配列rawdata[i][j:j+maxlen]のどちらの縦列内にもNoneがない
            #かつ
            #教師データの自車速度rawdata[i][j+maxlen+stepnum,0]がNoneではない
             if all(~np.isnan(rawdata[i][j:j+maxlen]).any(axis=0)) and all(~np.isnan(rawdata[i][j+maxlen+stepnum-1])): #(学習データ判別 and 教師データ判別)
                traintemp.append(rawdata[i][j:j+maxlen,0:2]) #windowsize分のデータ長。第三列は日付なので除外(,0:2)。
                targettemp.append(rawdata[i][j+maxlen+stepnum-1,0:2]) #windowsizeデータのstep個分先のデータ。第三列は日付なので除外(,0:2)
                new_sample_size = new_sample_size + 1 #欠損データを棄却したので、sample_sizeは元より小さくなっている。

        #CHANGED エラーが出るため変更
        #[x] rawdata[i][0][2]で"index 0 is out of bounds for axis 0 with size 0"がでる. i=7とか
        #? この処理で大丈夫なのか？要検証
        if rawdata[i].size!=0:
            car_date = rawdata[i][0][2]#車データ取得の日付

        if car_date not in trainset_dict: #日付が初見だった場合
            trainset_dict[car_date] = []#キーがその日付である空の配列を作成
            targetset_dict[car_date] = []#キーがその日付である空の配列を作成
        #学習データと教師データをappend。配列番号は必ず揃える。
        trainset_dict[car_date].append(np.array(traintemp).reshape(new_sample_size,maxlen,rawdata[i].shape[1] - 1))#(学習データ数,入力ステップ数,入力次元数)
        targetset_dict[car_date].append(np.array(targettemp).reshape(new_sample_size,rawdata[i].shape[1] - 1))#(学習データ数,出力次元数)


    #この時点で辞書の中に、全ての車両データからの欠損の無いデータ学習データと教師データが抽出され、日付ごとに格納されている。
    #各日の学習データを、指定された学習データ数に無作為に間引く。
    for date in trainset_dict.keys():#日付でループを回す。
        if len(np.concatenate(trainset_dict[date],axis=0)) >= need_data_num:#1日の学習データが足りていた場合
            #学習データを1日分結合
            cut_trainset = np.concatenate(trainset_dict[date],axis=0)
            cut_targetset = np.concatenate(targetset_dict[date],axis=0)
            print("Success : Sufficient data   : ", len(cut_trainset))
            #間引くために一旦リスト型に変換。後でnumpy型に戻す
            cut_trainset = cut_trainset.tolist()
            cut_targetset = cut_targetset.tolist()

            #seed値は日ごとに変えてもいいが、入力と教師データでは揃えないと、教師ラベルがズレる。
            random.seed(0)
            day_trainset = random.sample(cut_trainset,need_data_num)
            random.seed(0)
            day_targetset = random.sample(cut_targetset,need_data_num)

            #間引いたのでnumpy型に戻す
            day_trainset = np.array(day_trainset)
            day_targetset = np.array(day_targetset)

            #間引かれた学習データを確定して結合
            trainset.extend(day_trainset)
            targetset.extend(day_targetset)
            total_sample_size = total_sample_size + len(day_trainset)

        else:
            total_sample_size = total_sample_size + len(np.concatenate(trainset_dict[date],axis=0))
            print("Error   : Insufficient data : ", len(np.concatenate(trainset_dict[date],axis=0)))
    return trainset,targetset,total_sample_size

#[x]
def VarStepVLSTMdataset5(rawdata,maxlen,stepnum,whole_data,train_date):#(生データ,学習データステップ数,教師データのステップ数,所望学習データ数,所望学習日数)
    '''
    1入力1出力用学習データ形成プログラム
    欠損値を含む入力データ・教師データを除外するように学習データを成型
    '''
    #1日当たりの必要学習データ数
    #my //は商知らなかったわけじゃないんだ度忘れしてただけです～～
    need_data_num = whole_data//train_date

    #車のデータの日付が同じか判断する変数。
    judge_date = -1

    #DataFrame型からlist型に変換
    trainset,targetset = [],[]#学習データ数を確定してから1日分をこの配列に結合
    trainset_sub,targetset_sub = [],[]#学習データを間引く前の1日分の学習データ
    total_sample_size = 0
    sub_sample_size = 0
    trainset_dict = {}#日付がキーの辞書。同じ日付のものを配列にまとめる。入力データ
    
    #bookmark stepnumで変えれるのでは
    targetset_dict = {}#日付がキーの辞書。同じ日付のものを配列にまとめる。教師データ
    
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen-stepnum
        traintemp,targettemp = [],[]
        new_sample_size = 0 #欠損無データを数える用
        for j in range(sample_size):
            #入力データである自車速度と平均速度の15ステップ配列rawdata[i][j:j+maxlen]のどちらの縦列内にもNoneがない
            #かつ
            #教師データの自車速度rawdata[i][j+maxlen+stepnum,0]がNoneではない
             if all(~np.isnan(rawdata[i][j:j+maxlen][:,:1]).any(axis=0)) and ~np.isnan(rawdata[i][j+maxlen+stepnum-1][0]): #(学習データ(自車速度の列のみ)判別 and 教師データ(自車速度の列・配列0番)判別)
                traintemp.append(rawdata[i][j:j+maxlen,0:1]) #windowsize分のデータ長。第2列前方平均速度なので除外。第3列は日付なので除外(,0:1)。
                targettemp.append(rawdata[i][j+maxlen+stepnum-1,0:1]) #windowsizeデータのstep個分先のデータ。第2列前方平均速度なので除外。第3列は日付なので除外(,0:1)
                new_sample_size = new_sample_size + 1 #欠損データを棄却したので、sample_sizeは元より小さくなっている。
        
        #CHANGED エラーが出るため変更
        #[x] rawdata[i][0][2]で"index 0 is out of bounds for axis 0 with size 0"がでる. i=7とか
        #? この処理で大丈夫なのか？要検証
        if rawdata[i].size!=0:
            car_date = rawdata[i][0][2]#車データ取得の日付

        if car_date not in trainset_dict: #日付が初見だった場合
            trainset_dict[car_date] = []#キーがその日付である空の配列を作成
            targetset_dict[car_date] = []#キーがその日付である空の配列を作成
        #学習データと教師データをappend。配列番号は必ず揃える。
        trainset_dict[car_date].append(np.array(traintemp).reshape(new_sample_size,maxlen,rawdata[i].shape[1] - 2))#(学習データ数,入力ステップ数,入力次元数)
        targetset_dict[car_date].append(np.array(targettemp).reshape(new_sample_size,rawdata[i].shape[1] - 2))#(学習データ数,出力次元数)


    #この時点で辞書の中に、全ての車両データからの欠損の無いデータ学習データと教師データが抽出され、日付ごとに格納されている。
    #各日の学習データを、指定された学習データ数に無作為に間引く。
    for date in trainset_dict.keys():#日付でループを回す。
        if len(np.concatenate(trainset_dict[date],axis=0)) >= need_data_num:#1日の学習データが足りていた場合
            #学習データを1日分結合
            cut_trainset = np.concatenate(trainset_dict[date],axis=0)
            cut_targetset = np.concatenate(targetset_dict[date],axis=0)
            print("Success : Sufficient data   : ", len(cut_trainset))
            #間引くために一旦リスト型に変換。後でnumpy型に戻す
            cut_trainset = cut_trainset.tolist()
            cut_targetset = cut_targetset.tolist()

            #seed値は日ごとに変えてもいいが、入力と教師データでは揃えないと、教師ラベルがズレる。
            random.seed(0)
            day_trainset = random.sample(cut_trainset,need_data_num)
            random.seed(0)
            day_targetset = random.sample(cut_targetset,need_data_num)

            #間引いたのでnumpy型に戻す
            day_trainset = np.array(day_trainset)
            day_targetset = np.array(day_targetset)

            #間引かれた学習データを確定して結合
            trainset.extend(day_trainset)
            targetset.extend(day_targetset)
            total_sample_size = total_sample_size + len(day_trainset)

        else:
            total_sample_size = total_sample_size + len(np.concatenate(trainset_dict[date],axis=0))
            print("Error   : Insufficient data : ", len(np.concatenate(trainset_dict[date],axis=0)))

    #学習データ、教師データ、データセット数を返す
    return trainset,targetset,total_sample_size

#[x]
def slice_df(df: pd.DataFrame, maxlen: int, val_step: int, R_range: str) -> list:
    """pandas.DataFrameを1行ずつズラシながら行数maxlenずつにスライスしてリストに入れて返す"""
    #bookmark ここ大事そうval_stepを変えればできるのでは
    for i in range(val_step): #予測先各ステップの正解データの列を追加する。
        num = i + 1
        #カラム情報を1行上にずらしたデータフレームを作成する
        df_shift = df.shift(-num)
        #正解データの列を追加する
        df['self_valLabel'+str(num)] = df_shift['car_speed'] #予測対象車両の速度列
        df['ahead_valLabel'+str(num)] = df_shift['avr_speed_R'+R_range] #予測対象車両の速度列


    n = df.shape[0]
    list_indices = [(i, i+maxlen) for i in range(0, n, 1)]
    df_indices = [(i, i+maxlen-1) for i in range(0, n, 1)]
    sliced_dfs = []
    for i in range(len(df_indices)):
        begin_i, end_i = df_indices[i][0], df_indices[i][1]
        begin_l, end_l = list_indices[i][0], list_indices[i][1]
        df_i = df.loc[begin_i:end_i, :]
        sliced_dfs += [df_i]

    return sliced_dfs

#def VarStepVLSTMdataset7(rawdata,maxlen,MFwindow,val_step=1):#(生データ,入力ステップ数,平滑化ステップ数,検証するステップ数)
#    '''
#    1入力1出力用検証データ形成プログラム
#    欠損値を含む入力データは個別に補完。予測先ステップまでの各ステップを正解データとして検証データを成型
#    '''
#    sliced_rawdata_list=[]
#    valInput_list=[]
#    valLabel_list=[]

#    for i in range(len(rawdata)):
#        #車1台分の全入力データdf
#        dfs = slice_df(rawdata[i], maxlen,val_step)
#        for k in range(len(dfs)):
#            #入力データは2次元用意。入力直前でカットする。
#            valInput = dfs[k][['car_speed','avr_speed_R50']]
#            #indexを0から番号振り直し
#            valInput = valInput.reset_index(drop=True)

#            #正解ラベルはステップ分、2次元分用意。現在2ステップ予測のみ対応
#            valLabel = dfs[k][['self_valLabel1','ahead_valLabel1','self_valLabel2','ahead_valLabel2']] 
#            #indexを0から番号振り直し
#            valLabel = valLabel.reset_index(drop=True)

#            if len(valInput)==5:
#                #欠損値を線形補完
#                valInput = valInput.interpolate(limit_direction='both')
#                #全欠損の場合は'avr_speed_R50'に'car_speed'を代入
#                if valInput.isnull().values.sum() != 0: #欠損が1つでもある場合(補完後なので全欠損が対象となる)
#                    valInput['avr_speed_R50'] = valInput['car_speed']

#                #入力データ一つをnumpy化して結合
#                valInput_list.append(valInput.values)
#                #正解データ一つをnumpy化して結合
#                valLabel_list.append(valLabel.loc[[4], :].values)

#    total_sample_size = len(valInput_list)
#    if len(valLabel_list) != total_sample_size:
#        print("検証データサイズエラー：入力データと正解ラベルの数が不一致")
#        sys.exit(0)

#    #入力データ、正解ラベルを返す
#    return valInput_list, valLabel_list, total_sample_size

#[x]
#CHANGED
def VarStepVLSTMdataset8(rawdata,maxlen,MFwindow,R_range,val_step=3):#(生データ,入力ステップ数,平滑化ステップ数,検証するステップ数,Rの半径)
    '''
    検証データ形成プログラム
    欠損値を含む入力データは個別に補完。予測先ステップまでの各ステップを正解データとして検証データを成型
    2次元の入力データ群と、各出力・各ステップの正解ラベル、全入力データ数が返される
    '''
    sliced_rawdata_list=[]
    valInput_list=[]
    valLabel_list=[]

    print()
    print("Now making valdation data...")
    with tj.tqdm_joblib(len(rawdata)):
        results = Parallel(n_jobs=-1)(delayed(do_process)(rawdata[i], maxlen, val_step, R_range) for i in range(len(rawdata)))

    #resultsを入力とラベル形式に整える
    valInput_list_append = valInput_list.append 
    valLabel_list_append = valLabel_list.append 
    for i in range(len(results)):
        valInput_list_append
        for x in results[i][0]:
            valInput_list_append(x)
        for x in results[i][1]:
            valLabel_list_append(x)

    total_sample_size = len(valInput_list)
    if len(valLabel_list) != total_sample_size:
        print("検証データサイズエラー：入力データと正解ラベルの数が不一致")
        sys.exit(0)

    print()
    print("Total validation data : ", total_sample_size)

    #入力データ、正解ラベル、検証データ数を返す
    return valInput_list, valLabel_list, total_sample_size

#[x]
def do_process(rawdata, maxlen, val_step, R_range):
    do = make_valData(rawdata, maxlen,val_step,R_range)
    valInput_list, valLabel_list = do.process_making()
    return valInput_list, valLabel_list

#[x]
class make_valData:
    #* valDataの加工
    
    def __init__(self,rawdata, maxlen, val_step,R_range):
        self.rawdata = rawdata
        self.maxlen = maxlen
        self.val_step = val_step
        self.R_range = str(R_range)

    #[x]
    def process_making(self):
        valInput_list=[]
        valLabel_list=[]
        #車1台分の全入力データdf
        dfs = slice_df(self.rawdata, self.maxlen, self.val_step, self.R_range)
        for k in range(len(dfs)):
            #入力データは2次元用意。入力直前でカットする。
            valInput = dfs[k][['car_speed','avr_speed_R'+self.R_range]]
            #indexを0から番号振り直し
            valInput = valInput.reset_index(drop=True)

            #正解ラベルはステップ分、2次元分用意。現在2ステップ予測のみ対応
            valLabel = dfs[k][['self_valLabel1','ahead_valLabel1','self_valLabel2','ahead_valLabel2']] 
            #indexを0から番号振り直し
            valLabel = valLabel.reset_index(drop=True)

            #?この5はなんだ？もしかしてwindow_sizeかもしれない
            #CHANGED ↑ということで一応15にしとく
            if len(valInput)==self.maxlen:
                #欠損値を線形補完
                valInput = valInput.interpolate(limit_direction='both')
                #全欠損の場合は'avr_speed_Rxx'に'car_speed'を代入
                if valInput.isnull().values.sum() != 0: #欠損が1つでもある場合(補完後なので全欠損が対象となる)
                    valInput['avr_speed_R'+self.R_range] = valInput['car_speed']

                #入力データ一つをnumpy化して結合
                valInput_list.append(valInput.values)
                #正解データ一つをnumpy化して結合
                valLabel_list.append(valLabel.loc[[self.maxlen-1], :].values)

        return valInput_list, valLabel_list



def VarStepVLSTMdataset_for_Origin(rawdata,maxlen,stepnum):#(生データ,学習データステップ数,1ステップの大きさ)
    '''
    欠損値を含んだまま学習データを成型
    '''
    #DataFrame型からlist型に変換
    trainset,targetset = [],[]
    for i in range(len(rawdata)):
        sample_size = len(rawdata[i])-maxlen-stepnum
        traintemp  = [rawdata[i][j:j+maxlen] for j in range(sample_size)]
        #targettemp = [rawdata[i][j+maxlen+stepnum,0] for j in range(sample_size)]
        targettemp = [rawdata[i][j+maxlen+stepnum-1] for j in range(sample_size)]

        #numpy.arrayで配列を作る。
        trainset.append(np.array(traintemp).reshape(sample_size,maxlen,rawdata[i].shape[1]))
        targetset.append(np.array(targettemp).reshape(sample_size,rawdata[i].shape[1]))

    return trainset,targetset