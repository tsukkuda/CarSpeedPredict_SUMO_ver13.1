#必要なもの(ライブラリ)をここで読み込む
import copy
import datetime
import pickle
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
import pathlib
# このファイルがあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append(str(current_dir) + '/./')

from preprocess.Preprocessing import DataPreprocessing,Normarizing,DataPreprocessing2,Normarizing2,DataPreprocessing3
from evaluate.PredEval import PredEval
from preprocess.Process_train import train_func 
import NNMakeDataset.makeDataset3 as mkdataset
import NNkeras.kerasInit as kInit
import NNModel.VLSTM
import NNModel.NN


model_Input_Output = 0 #1入力1出力学習と2入力2出力学習を切り替える変数。次元数をここで指定する。入力と出力の次元数は同じ。0なら1次元と2次元両方。
switch=False #* 1入力1出力学習の学習データを自車速度(True)にするか、前方平均速度(False)にするか

#[x]
def ProcessMVLSTM(original_data, original_valdata_list, starttime, hyper_parameter, pred_step, stepnum, R_num):
    #このプログラムの開始時刻取得
    dt_now = datetime.datetime.now()

    #福丸 original_dataのうちどの特徴量を利用するか
    # data_list = []
    # data_list = [0,1]
    # data_list = [0,2]
    #data_list = [0,3]
    
    #original_dataは[car_speed,avr_speedR50,avr_speedR100,avr_speedR150,avr_speedR200,avr_speed250,avr_speed300,date]
    data_list = [0,R_num,7] #自車速度と15s先の位置の平均速度と学習データの記録日
    # data_list = [0,1,2,3]
    # data_list = [0,2,4,6]
    # data_list = [0,3,6]
    # data_list = [0,1,2]
    # data_list = [0,1,3]
    # data_list = [0,2,3]
    # data_list = [0,0]

    R_range = R_num*50 #Rの半径の大きさ

    with open('log.txt', 'a') as f:
        f.write(str(dt_now))
        f.write('\n')
        for d in data_list:
            f.write(str(d) + ", ")
        f.write('\n')
    
#===ここから学習・検証に使用する特徴量列のみを抜粋=======================================================================
    original_data_for_processing = copy.deepcopy(original_data) #加工用に生データをコピーする
    original_valdata_list_for_processing = copy.deepcopy(original_valdata_list)
    if len(data_list) > 0:
        #学習・交差検証用データに対する処理
        #my ここで該当するRum_numの列を抽出している
        for i in range(len(original_data_for_processing)):
            original_data_for_processing[i] = original_data_for_processing[i].iloc[:,data_list]

        #検証用データに対する処理
        #リストの中身を一つずつ処理。original_valdata_list_for_processingそのものが改変される。
        for original_valdata in original_valdata_list_for_processing:
            for i in range(len(original_valdata)):
                original_valdata[i] = original_valdata[i].iloc[:,data_list]
    
#===ここから訓練データを正規化した後、欠損補完して平滑化し、再欠損させる=======================================================================

    #全データを使用して学習モデルを形成

    #オリジナル訓練データを正規化する。
    normalized_original_data,max_origin_data = Normarizing2(original_data_for_processing) #オリジナル訓練データ正規化
    #オリジナルデータ訓練を平滑化する。欠損を全て補完してから平滑化を施し、同箇所を再欠損させる。
    preprocessed_original_data = DataPreprocessing3(normalized_original_data,MFwindow=hyper_parameter["median"])#オリジナル訓練データを平滑化
    
    #? preprocessed_original_dataはどういう形なのか car_speed, avr_speed_R50, date(?)
#===ここから検証データを正規化する=======================================================================

    valInset_list=[]        #検証用の入力データ用リスト
    valLabset_list=[]       #検証用の正解データ用リスト
    valSampleSize_list=[]   #検証用データの入力変数セットの個数のリスト
    #リストの中身を一つずつ処理。
    for original_valdata in original_valdata_list_for_processing:
        #オリジナル検証データを正規化する。
        normalized_original_valdata,max_origin_valdata = Normarizing2(original_valdata) #オリジナル検証データ正規化

        #検証データをLSTMの入力形式に変換する。入力データ個別に欠損補完が施される。
        #*　ここでもう正解ラベルとかついてる
        #CHANGED val_Step=6つまり5s*6=30s後の結果入れてる
        valInset,valLabset,valSampleSize = mkdataset.VarStepVLSTMdataset8(normalized_original_valdata,hyper_parameter["window_len"],hyper_parameter["median"],R_range,val_step=6)

        #リストにまとめる
        valInset_list.append(valInset)
        valLabset_list.append(valLabset)
        valSampleSize_list.append(valSampleSize)
    
#===ここから学習・検証データを指定個数に選定=======================================================================

    #訓練データを任意の個数に選定する

    #使用する訓練データの個数を指定。65536セット。
    #2の16乗
    #whole_data = 65536
    #whole_data = 150
    #whole_data = 15360 #256個60日
    whole_data = 100 #100個60日 #CHANGED 100個6日(seed10種)
    #whole_data = 300
    #whole_data = 4000
    #学習したい日数
    #train_date = 3 #whole_data/train_dateが1日あたりに取得したい訓練データの個数。
    train_date = 1 #CHANGED 1回のシミュレーションから取得したい訓練データの数。日数*seed種数
    #train_date = 40

    #オリジナルデータをLSTMの訓練データの形に変形する。Window_sizeステップずつに切り分けて教師データとセットにする。
    #ここで欠損のない訓練データの個数を数える
    #日付ごとの訓練データの個数をカウントして返してもらう
    #日付順に整列される
    
    #* 1入力1出力(前方平均速度)
    if not switch:
        #剪定済み訓練データと訓練データ総数を返される
        trainset,trainLabset,sample_size = mkdataset.VarStepVLSTMdataset6(preprocessed_original_data,hyper_parameter["window_len"],stepnum,whole_data,train_date) #1入力1出力        
        #学習と交差検証の実施
        #! 訓練データは6step後のものを与えている。
        train_func(trainset,trainLabset,whole_data,sample_size,
                   valInset_list,valLabset_list,valSampleSize_list,
                   hyper_parameter,1,starttime,dt_now,1,R_range,switch,)
        

    if model_Input_Output == 0: #1入力1出力と2入力2出力
        #1入力1出力
        #剪定済み訓練データと訓練データ総数を返される
        trainset,trainLabset,sample_size = mkdataset.VarStepVLSTMdataset5(preprocessed_original_data,hyper_parameter["window_len"],stepnum,whole_data,train_date) #1入力1出力
        #学習と交差検証の実施
        #! 訓練データは6step後のものを与えている。
        train_func(trainset,trainLabset,whole_data,sample_size,
                   valInset_list,valLabset_list,valSampleSize_list,
                   hyper_parameter,1,starttime,dt_now,1,R_range)
        #2入力2出力
        #剪定済み訓練データと訓練データ総数を返される
        trainset,trainLabset,sample_size = mkdataset.VarStepVLSTMdataset4(preprocessed_original_data,hyper_parameter["window_len"],stepnum,whole_data,train_date) #2入力2出力
        #学習と交差検証の実施
        #! 訓練データは6step後のものを与えている。
        train_func(trainset,trainLabset,whole_data,sample_size,
                   valInset_list,valLabset_list,valSampleSize_list,
                   hyper_parameter,1,starttime,dt_now,2,R_range)
    elif model_Input_Output == 1: #1入力1出力
        #剪定済み訓練データと訓練データ総数を返される
        trainset,trainLabset,sample_size = mkdataset.VarStepVLSTMdataset5(preprocessed_original_data,hyper_parameter["window_len"],stepnum,whole_data,train_date) #1入力1出力        
        #学習と交差検証の実施
        #! 訓練データは6step後のものを与えている。
        train_func(trainset,trainLabset,whole_data,sample_size,
                   valInset_list,valLabset_list,valSampleSize_list,
                   hyper_parameter,1,starttime,dt_now,model_Input_Output,R_range)
    elif model_Input_Output == 2: #2入力2出力
        #剪定済み訓練データと訓練データ総数を返される
        trainset,trainLabset,sample_size = mkdataset.VarStepVLSTMdataset4(preprocessed_original_data,hyper_parameter["window_len"],stepnum,whole_data,train_date) #2入力2出力
        #学習と交差検証の実施
        #! 訓練データは6step後のものを与えている。
        train_func(trainset,trainLabset,whole_data,sample_size,
                   valInset,valLabset,valSampleSize,
                   hyper_parameter,1,starttime,dt_now,model_Input_Output,R_range)