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
import NNMakeDataset.makeDataset3 as mkdataset
import NNkeras.kerasInit as kInit
import NNModel.VLSTM
import NNModel.NN

def ProcessMVLSTM(original_data,starttime,hyper_parameter,pred_step,stepnum):
    #このプログラムの開始時刻取得
    dt_now = datetime.datetime.now()

    #福丸 original_dataのうちどの特徴量を利用するか
    # data_list = []
    # data_list = [0,1]
    # data_list = [0,2]
    data_list = [0,3]
    # data_list = [0,1,2,3]
    # data_list = [0,2,4,6]
    # data_list = [0,3,6]
    # data_list = [0,1,2]
    # data_list = [0,1,3]
    # data_list = [0,2,3]
    # data_list = [0,0]

    with open('log.txt', 'a') as f:
        f.write(str(dt_now))
        f.write('\n')
        for d in data_list:
            f.write(str(d) + ", ")
        f.write('\n')
    
#===ここから学習・検証データ成型=======================================================================
    if len(data_list) > 0:
        for i in range(len(original_data)):
            original_data[i] = original_data[i].iloc[:,data_list]

    #何個分訓練元データとして使うかを指定
    #TODO hold-out法 精度を上げたいときはここを増やして試したい
    #train_range = 80
    #train_range = 50

    #NOTE 福丸: k-分割法のテストデータの個数
    #test_range = 50
    test_range = len(original_data)//5

    #TODO 20191110 trainとtestが分離されている状況を改善するか検討
    #元データを訓練元データとテスト元データに分ける
    #train_rangeで指定した個数分の訓練元データと1個のテスト元データができる(Hold-out法)
    trainorigin,testorigin = [],[]

    #先頭からtrain_range分は訓練元データとする
    #残りはテストデータとする
    #例：car1のデータからcar10までをtrainorigin[0]とし、car11のデータをtestorigin[0]とする
    #for i in range(len(original_data)-train_range):
    #    trainorigin.append(original_data[i:i+train_range])
    #    testorigin.append(original_data[i+train_range:i+train_range+1])

    #TODO(福丸): k-分割法のテスト
    #厳密には分割になっていない。割合や分割数を変えたい場合は調整の必要あり
    for i in range(len(original_data) // test_range):
        trainorigin.append(original_data[:i*test_range] + original_data[(i+1)*test_range:])
        testorigin.append(original_data[i*test_range:(i+1)*test_range])

    #評価指標計算結果を保存する箱を用意
    #test_eval = PredEval(sample_num=len(testorigin),
    #                     hyper_param=hyper_parameter)
    count = 0
    for testdata in testorigin:
        count += len(testdata)
    test_eval = PredEval(sample_num=count,
                         hyper_param=hyper_parameter)

    for i in range(len(testorigin)):
        #TODO 20191110 最大値の保持をクラスで行うかどうか検討
        #データの前処理(平滑化&正規化)を行って最大値を保持する
        #平滑化する場合はDataPreprocessing3、平滑化しない場合はDataPreprocessing2
        preprocessed_trainorigin = DataPreprocessing3(trainorigin[i],MFwindow=hyper_parameter["median"])#訓練データを平滑化
        preprocessed_testorigin  = DataPreprocessing3(testorigin[i] ,MFwindow=hyper_parameter["median"])#検証データを平滑化
        normarized_trainorigin,max_trainorigin = Normarizing2(preprocessed_trainorigin) #訓練データ正規化
        normarized_testorigin ,max_testorigin  = Normarizing2(preprocessed_testorigin)  #検証データ正規化

        #学習用に訓練元データをLSTMの訓練データに変形する。Window_sizeステップずつに切り分けて教師データとセットにする。
        trainset,trainLabset = mkdataset.VarStepVLSTMdataset(normarized_trainorigin,hyper_parameter["window_len"],stepnum)
        #学習データの検証用にNoneを含んだ入力・教師データのセットを作成する。学習には使用しない。
        trainsetOrigin,trainLabsetOrign = mkdataset.VarStepVLSTMdataset_for_Origin(normarized_trainorigin,hyper_parameter["window_len"],stepnum)
        #TODO 一時的にデータが足りない場合に対応
        #if trainset
        #テスト元データをLSTMのテストデータに変形する
        testset ,_           = mkdataset.VarStepVLSTMdataset(normarized_testorigin ,hyper_parameter["window_len"],stepnum)
        #未知データの検証用にNoneを含んだ入力・教師データのセットを作成する。学習には使用しない。
        testsetOrigin ,_           = mkdataset.VarStepVLSTMdataset_for_Origin(normarized_testorigin ,hyper_parameter["window_len"],pred_step)#任意のステップ数で予測させる

        #リストごとに分けられている3階ndarrayを第一次元を軸に連結
        trainIn  = np.concatenate(trainset,axis=0)
        trainLab = np.concatenate(trainLabset,axis=0)

#=======ここから学習===================================================================
        #kerasGPU設定初期化(ここでGPUメモリ占有の設定を行う)
        kInit.kerasInit()

        #訓練データによる予測(fitting)
        #訓練データを入れて予測した場合、モデルが学習できているかをテストすることになる
        train_eval = PredEval(sample_num=len(trainorigin[0]),
                              hyper_param=hyper_parameter)

        #結果を出力するフォルダを設定
        learntype = 'train'+str(i)  #何番目の学習か
        #TODO 20191122 モデルの名前を書き換えずにモデルを変えてしまうと一貫性が崩れてしまうため要改善
        # modelname = 'RNN_Affine'  #モデルの名前は何か
        modelname = 'MVLSTM_Affine'  #モデルの名前は何か
        # modelname = 'GRU_Affine'  #モデルの名前は何か
        train_eval.SetFolderpath(modelname=modelname,
                                 hyper_param=hyper_parameter,
                                 starttime=starttime,
                                 dt_now=dt_now,
                                 learntype=learntype,
                                 pred_num=stepnum)

        #訓練データを使ってモデルを学習させる
        TensorBoard_path = train_eval.GetFolderpath()
        model,history = NNModel.NN.VLSTM_Affine(train=trainIn,
                                                   label=trainLab,
                                                   hyperparam=hyper_parameter,
                                                   tensorboard_path=TensorBoard_path)

        #学習のlossをグラフで出力
        train_eval.ExportLossGraph(history=history)

        #渋滞の基準設定
        jamspeed = 60
        jamspeed = jamspeed / 1.61 #mileをkm/hに変換
#=======ここから学習データで予測===================================================================
        print()
        print("now predicting with TRAIN DATA and saving the result...")
        for ti in tqdm(range(len(trainsetOrigin))):
            if trainsetOrigin[ti].size == 0:
                continue
            #学習データで予測
            #使うのはNoneを残した予測検証用のセット。Noneを含む入力はmodel.predictが自動で避けてくれる。マジで神。
            pred_speed = model.predict(trainsetOrigin[ti])
            pred_speed = pred_speed * max_trainorigin*3.66 #正規化を元に戻す
            normarized_trainorigin[ti] = normarized_trainorigin[ti] * max_trainorigin*3.66 #実測値の正規化を元に戻す

            #指標評価&記録
            train_eval.RecordMetrics(pred_speed=pred_speed,
                                     pred_index=0,
                                     origindata=pd.DataFrame(normarized_trainorigin[ti]),
                                     jamspeed=jamspeed,
                                     pred_range=stepnum,
                                     pred_num=stepnum
                                     )

        #評価指標出力
        train_eval.ExportMetrics()

#=======ここから検証データで予測===================================================================
        #検証データ予測(predict)
        #検証データを入力して、1ステップ先を予測する
        test_eval = PredEval(sample_num=len(testorigin[0]),
                              hyper_param=hyper_parameter)

        #結果を出力するフォルダを設定
        testtype = 'test'+str(i)  #何番目のテストか
        test_eval.SetFolderpath(modelname=modelname,
                                hyper_param=hyper_parameter,
                                starttime=starttime,
                                dt_now=dt_now,
                                learntype=testtype,
                                pred_num=stepnum)

        print()
        print("now predicting with VALIDATION DATA and saving the result...")
        for ti in tqdm(range(len(testsetOrigin))):
            if testsetOrigin[ti].size == 0:
                continue
            #学習に含めていないテストデータで予測。
            #使うのはNoneを残した予測検証用のセット。Noneを含む入力はmodel.predictが自動で避けてくれる。マジで神。
            looptestset_for_pred = np.copy(testsetOrigin[ti]) #ループ処理で使う用にコピーを用意

            for loop_step in range(pred_step):
                pred_speed = model.predict(looptestset_for_pred) #1ステップ先を予測

                #直近に予測値を追加した新しいwindowsizeの時系列配列を作成
                pred_speed = np.reshape(pred_speed,(len(pred_speed),1,2)) #3次元に結合するための前処理
                looptestset_for_pred = np.append(looptestset_for_pred,pred_speed,axis=1) #時系列の最後尾に予測値を追加
                looptestset_for_pred = np.delete(looptestset_for_pred, 0, axis=1) #1行目を削除

            #配列の形状を元に戻す
            pred_speed = np.reshape(pred_speed,(len(pred_speed),2))

            pred_speed = pred_speed * max_testorigin*3.66 #正規化を元に戻す
            normarized_testorigin[ti] = normarized_testorigin[ti] * max_testorigin*3.66 #実測値の正規化を元に戻す

            #指標評価&記録
            test_eval.RecordMetrics(pred_speed=pred_speed,
                                    pred_index=0,
                                    origindata=pd.DataFrame(normarized_testorigin[ti]),
                                    jamspeed=jamspeed,
                                    pred_range=stepnum,
                                    pred_num=pred_step
                                    )

        #評価指標出力
        test_eval.ExportMetrics()
