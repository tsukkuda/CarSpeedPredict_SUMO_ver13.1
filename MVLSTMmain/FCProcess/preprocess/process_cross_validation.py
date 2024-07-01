from evaluate.PredEval import PredEval
import NNModel.NN
import NNkeras.kerasInit as kInit
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import csv
import itertools
import random

def cross_val(trainIn,trainLab,valIn_list,valLab_list,hyper_parameter,starttime,dt_now,stepnum,model_Input_Output,R_range):#(訓練データの入力,訓練データの教師データ)

    whole_data_origin = len(trainIn) #元々の訓練データの数

    #交差検証分割数
    #なるべく学習させる日数（sim回数）の約数を指定。
    #そうしないと、交差検証の検証データに訓練データと似たデータが含まれてしまい、精度が高く出てしまう。(1台の車の速度データcsvが3つに分けられて読み込まれているいるため)
    n_splits=3 #学習させる日数（sim回数）の約数を指定。

    #訓練データと教師データを分割する。割り切れない場合は[0]に余りが足される。
    trainIn_splited_list = list(np.array_split(trainIn, n_splits))#
    trainLab_splited_list = list(np.array_split(trainLab, n_splits))

    valid_scores = [] #交差検証の結果を代入する空のリスト
    valid_scores2 = [] #交差検証の結果を代入する空のリスト
    for i in range(n_splits):

        TrainIn_list = copy.deepcopy(trainIn_splited_list)
        del TrainIn_list[i]#検証データを除く
        TrainIn = TrainIn_list[0]#訓練データ統合の前準備
        for j in range(len(TrainIn_list)-1):
            TrainIn = np.concatenate([TrainIn,TrainIn_list[j+1]])#一つずつ統合

        TrainLab_list = copy.deepcopy(trainLab_splited_list)
        del TrainLab_list[i]#検証データを除く
        TrainLab = TrainLab_list[0]#訓練データ統合の前準備
        for j in range(len(TrainLab_list)-1):
            TrainLab = np.concatenate([TrainLab,TrainLab_list[j+1]])#一つずつ統合

        #検証に用いる配列を指定
        ValIn = trainIn_splited_list[i]
        ValLab = trainLab_splited_list[i]

        whole_data = len(TrainIn) #訓練データの数
        Cross_Val = cross_validation(hyper_parameter,starttime,dt_now,stepnum,model_Input_Output,whole_data,whole_data_origin,R_range,val_num=i+1)
        Cross_Val.set_FolderPath()#出力先フォルダをセット
        Cross_Val.TrainModel(TrainIn,TrainLab)#交差検証の実行
        Cross_Val.ExportLossGraph()#学習のlossをグラフで出力
        #学習モデルの予測精度を交差検証データで検証
        trainval_RMSE,trainval_MAE,trainval_SDAE = Cross_Val.velify(ValIn,ValLab,"cross_val")
        #学習モデルの予測精度を学習データで検証
        train_RMSE,train_MAE,train_SDAE = Cross_Val.velify(TrainIn,TrainLab,"train_val")

        #学習モデルの予測精度を検証データで検証
        val_RMSE_list=[]
        val_MAE_list=[]
        val_SDAE_list=[]
        val_RMSE2_list=[]
        val_MAE2_list=[]
        val_SDAE2_list=[]
        for index in range(len(valIn_list)):#用意した割合の種類分でループ
            rate = str(index + 1)
            val_RMSE,val_MAE,val_SDAE,val_RMSE2,val_MAE2,val_SDAE2 = Cross_Val.velify2(valIn_list[index],valLab_list[index],"rate"+rate+"_val",model_Input_Output)
            val_RMSE_list.append(val_RMSE)
            val_MAE_list.append(val_MAE)
            val_SDAE_list.append(val_SDAE)
            val_RMSE2_list.append(val_RMSE2)
            val_MAE2_list.append(val_MAE2)
            val_SDAE2_list.append(val_SDAE2)

        #評価指標をまとめる
        #1stepscore
        valid_score = [i+1,trainval_RMSE,trainval_MAE,trainval_SDAE,train_RMSE,train_MAE,train_SDAE,val_RMSE_list,val_MAE_list,val_SDAE_list]
        valid_scores.append(valid_score)
        #2stepscore
        valid_score2 = [i+1,val_RMSE2_list,val_MAE2_list,val_SDAE2_list]
        valid_scores2.append(valid_score2)
        
    Cross_Val.make_valid_scores_csv(valid_scores) #交差検証の結果をまとめたcsvを書き出す。
    Cross_Val.make_valid_scores2_csv(valid_scores2) #交差検証の結果をまとめたcsvを書き出す。

class cross_validation:
    
    def __init__(self,hyper_parameter,starttime,dt_now,stepnum,model_Input_Output,whole_data,whole_data_origin,R_range,val_num=0):
        self.hyper_parameter = hyper_parameter
        self.starttime = starttime
        self.dt_now = dt_now
        self.stepnum = stepnum
        self.model_Input_Output = model_Input_Output
        self.whole_data = whole_data
        self.whole_data_origin = whole_data_origin
        self.val_num = val_num
        self.folder_path =  '.'
        self.score_folder_path =  '.'
        self.model = None
        self.histry = None
        self.learntype = None
        self.R_range = R_range

    def TrainModel(self,trainIn,trainLab):
        '''
        予測モデルの学習
        '''
        
        #kerasGPU設定初期化(ここでGPUメモリ占有の設定を行う)
        kInit.kerasInit()
        self.model,self.history = NNModel.NN.VLSTM_Affine(train=trainIn,
                                            label=trainLab,
                                            hyperparam=self.hyper_parameter,
                                            tensorboard_path=self.folder_path,
                                            model_Input_Output = self.model_Input_Output)

    #? 110から120にしたほうがいいかもしれない おそらく最高速度
    def velify(self,validIn,validLab,val_mode):
        '''
        予測精度の検証
        '''
        print("Now predicting for validation...")
        pred = self.model.predict(validIn)#予測値を全て計算
        
        if len(pred[0])==1:#次元が1の場合
            loss = pred - validLab #予測誤差を計算
            square_loss = loss**2 #予測誤差の2乗を計算
            MSE = np.mean(square_loss) #予測誤差の2乗平均を計算
            RMSE = np.sqrt(MSE)*110 #RMSEを計算
            MAE = np.mean(np.sqrt(square_loss))*110 #MAEを計算
            SDAE = np.sqrt(np.mean((MAE - loss)**2)) #Standard Deviation Absolute Error 絶対誤差の標準偏差を計算
            # 散布図を描画
            print("Now drawing scatters...")
            if len(validLab)>20000:#データ数が20000を超える場合
                random.seed(1)#間引く用の乱数の種
                cut_validLab = np.array(random.sample(validLab.tolist(),20000))
                random.seed(1)#間引く用の乱数の種
                cut_pred     = np.array(random.sample(pred.tolist(),20000))
            else:
                cut_validLab = copy.deepcopy(validLab)
                cut_pred = copy.deepcopy(pred)
            plt.scatter(cut_validLab*110, cut_pred*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE,2))+' MAE:'+str(round(MAE,2))+' SDAE:'+str(round(SDAE,2))) #グラフタイトルを指定
            plt.savefig(self.folder_path + val_mode + "_self.png")#グラフを保存
            plt.close()

        elif len(pred[0])==2:#次元が2の場合
            loss = pred - validLab #予測誤差を計算
            square_loss = loss**2 #予測誤差の2乗を計算
            MSE = np.mean(square_loss,axis=0)#列ごとに平均 予測誤差の2乗平均を計算
            RMSE = np.sqrt(MSE)*110 #RMSEを計算
            MAE = np.mean(np.sqrt(square_loss),axis=0)*110 #MAEを計算
            SDAE = np.sqrt(np.mean((MAE - loss)**2,axis=0)) #Standard Deviation Absolute Error 絶対誤差の標準偏差を計算
            # 散布図を描画
            print("Now drawing scatters...")
            if len(validLab)>20000:#データ数が20000を超える場合
                random.seed(1)#間引く用の乱数の種
                cut_validLab = np.array(random.sample(validLab.tolist(),20000))
                random.seed(1)#間引く用の乱数の種
                cut_pred     = np.array(random.sample(pred.tolist(),20000))
            else:
                cut_validLab = copy.deepcopy(validLab)
                cut_pred = copy.deepcopy(pred)
            plt.scatter(cut_validLab[:,0]*110, cut_pred[:,0]*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE[0],2))+' MAE:'+str(round(MAE[0],2))+' SDAE:'+str(round(SDAE[0],2))) #グラフタイトルを指定
            plt.savefig(self.folder_path + val_mode + "_self.png")#グラフを保存
            plt.close()
            # 散布図を描画
            print("Now drawing scatters...")
            plt.scatter(cut_validLab[:,1]*110, cut_pred[:,1]*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE[1],2))+' MAE:'+str(round(MAE[1],2))+' SDAE:'+str(round(SDAE[1],2))) #グラフタイトルを指定
            plt.savefig(self.folder_path + val_mode + "_ahead.png")#グラフを保存
            plt.close()
            
        else:
            print("Errorr of Scatter")
        pass
        return RMSE,MAE,SDAE #評価指標を返す

    def velify2(self,validIn,validLab,val_mode,model_Input_Output):
        '''
        検証データでの予測精度の検証
        '''
        #出力フォルダの指定と作成
        result_path1 = self.folder_path + "1step_val/"
        if not os.path.exists(result_path1):
            os.makedirs(result_path1)
        result_path2 = self.folder_path + "2step_val/"
        if not os.path.exists(result_path2):
            os.makedirs(result_path2)
        
        if model_Input_Output==1:#次元が1の場合

            print("Now predicting for validation...")
            valIn = np.delete(validIn, 1, 2)#前方平均の列を削除
            pred = self.model.predict(valIn)#1ステップ予測値を全て計算

            for_merge = np.reshape(pred,(len(pred),1,1)) #結合に向けて次元変更
            new_valIn = np.concatenate([valIn,for_merge],1) #新しい時系列を後ろに一つ結合
            valIn2 = np.delete(new_valIn,0,1) #一番頭の古い時系列を一つ削除

            print("Now predicting for validation...")
            pred2 = self.model.predict(valIn2)#2ステップ予測値を全て計算

            valLab1 = np.delete(validLab, 1, 2)#1列削除
            valLab1 = np.delete(valLab1, 1, 2)#1列削除
            valLab1 = np.delete(valLab1, 1, 2)#1列削除
            valLab1 = valLab1.reshape(len(valLab1),1)#1stepの正解ラベル
            valLab2 = np.delete(validLab, 0, 2)#1列削除
            valLab2 = np.delete(valLab2, 0, 2)#1列削除
            valLab2 = np.delete(valLab2, 1, 2)#1列削除
            valLab2 = valLab2.reshape(len(valLab2),1)#2stepの正解ラベル

            #===ここから1step予測に対する評価処理===
            loss = pred - valLab1 #予測誤差を計算
            square_loss = loss**2 #予測誤差の2乗を計算
            MSE = np.nanmean(square_loss) #予測誤差の2乗平均を計算
            RMSE = np.sqrt(MSE)*110 #RMSEを計算
            MAE = np.nanmean(np.sqrt(square_loss))*110 #MAEを計算
            SDAE = np.sqrt(np.nanmean((MAE - loss)**2)) #Standard Deviation Absolute Error 絶対誤差の標準偏差を計算
            # 散布図を描画
            print("Now drawing scatters...")
            if len(valLab1)>20000:#データ数が20000を超える場合
                random.seed(1)#間引く用の乱数の種
                cut_valLab1 = np.array(random.sample(valLab1.tolist(),20000))
                random.seed(1)#間引く用の乱数の種
                cut_pred     = np.array(random.sample(pred.tolist(),20000))
            else:
                cut_valLab1 = copy.deepcopy(valLab1) #? valLab1のタイプミスか？
                cut_pred = copy.deepcopy(pred)
            plt.scatter(cut_valLab1*110, cut_pred*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE,2))+' MAE:'+str(round(MAE,2))+' SDAE:'+str(round(SDAE,2))) #グラフタイトルを指定
            plt.savefig(result_path1 + val_mode + "_self.png")#グラフを保存
            plt.close()
            #===ここまで1step予測に対する評価処理===

            #===ここから2step予測に対する評価処理===
            loss2 = pred2 - valLab2 #予測誤差を計算
            square_loss2 = loss2**2 #予測誤差の2乗を計算
            MSE2 = np.nanmean(square_loss2) #予測誤差の2乗平均を計算
            RMSE2 = np.sqrt(MSE2)*110 #RMSEを計算
            MAE2 = np.nanmean(np.sqrt(square_loss2))*110 #MAEを計算
            SDAE2 = np.sqrt(np.nanmean((MAE2 - loss2)**2)) #Standard Deviation Absolute Error 絶対誤差の標準偏差を計算
            # 散布図を描画
            print("Now drawing scatters...")
            if len(valLab2)>20000:#データ数が20000を超える場合
                random.seed(1)#間引く用の乱数の種
                cut_valLab2 = np.array(random.sample(valLab2.tolist(),20000))
                random.seed(1)#間引く用の乱数の種
                cut_pred2     = np.array(random.sample(pred2.tolist(),20000))
            else:
                cut_valLab2 = copy.deepcopy(valLab2)
                cut_pred2 = copy.deepcopy(pred2)
            plt.scatter(cut_valLab2*110, cut_pred2*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE2,2))+' MAE:'+str(round(MAE2,2))+' SDAE:'+str(round(SDAE2,2))) #グラフタイトルを指定
            plt.savefig(result_path2 + val_mode + "_self.png")#グラフを保存
            plt.close()
            #===ここまで2step予測に対する評価処理===


        elif model_Input_Output==2:#次元が2の場合
            print("Now predicting for validation...")
            valIn = validIn
            pred = self.model.predict(valIn)#予測値を全て計算

            for_merge = np.reshape(pred,(len(pred),1,2)) #結合に向けて次元変更
            new_valIn = np.concatenate([valIn,for_merge],1) #新しい時系列を後ろに一つ結合
            valIn2 = np.delete(new_valIn,0,1) #一番頭の古い時系列を一つ削除

            print("Now predicting for validation...")
            pred2 = self.model.predict(valIn2)#2ステップ予測値を全て計算

            valLab1 = np.delete(validLab, 2, 2)#1列削除
            valLab1 = np.delete(valLab1, 2, 2)#1列削除
            valLab1 = valLab1.reshape(len(valLab1),2)#1stepの正解ラベル
            valLab2 = np.delete(validLab, 0, 2)#1列削除
            valLab2 = np.delete(valLab2, 0, 2)#1列削除
            valLab2 = valLab2.reshape(len(valLab2),2)#2stepの正解ラベル

            #===ここから1step予測に対する評価処理===
            loss = pred - valLab1 #予測誤差を計算
            square_loss = loss**2 #予測誤差の2乗を計算
            MSE = np.nanmean(square_loss,axis=0)#列ごとに平均 予測誤差の2乗平均を計算
            RMSE = np.sqrt(MSE)*110 #RMSEを計算
            MAE = np.nanmean(np.sqrt(square_loss),axis=0)*110 #MAEを計算
            SDAE = np.sqrt(np.nanmean((MAE - loss)**2,axis=0)) #Standard Deviation Absolute Error 絶対誤差の標準偏差を計算
            # 散布図を描画
            print("Now drawing scatters...")
            if len(valLab1)>20000:#データ数が20000を超える場合
                random.seed(1)#間引く用の乱数の種
                cut_valLab1 = np.array(random.sample(valLab1.tolist(),20000))
                random.seed(1)#間引く用の乱数の種
                cut_pred     = np.array(random.sample(pred.tolist(),20000))
            else:
                cut_valLab1 = copy.deepcopy(valLab1)
                cut_pred = copy.deepcopy(pred)
            plt.scatter(cut_valLab1[:,0]*110, cut_pred[:,0]*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE[0],2))+' MAE:'+str(round(MAE[0],2))+' SDAE:'+str(round(SDAE[0],2))) #グラフタイトルを指定
            plt.savefig(result_path1 + val_mode + "_self.png")#グラフを保存
            plt.close()
            # 散布図を描画
            print("Now drawing scatters...")
            plt.scatter(cut_valLab1[:,1]*110, cut_pred[:,1]*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE[1],2))+' MAE:'+str(round(MAE[1],2))+' SDAE:'+str(round(SDAE[1],2))) #グラフタイトルを指定
            plt.savefig(result_path1 + val_mode + "_ahead.png")#グラフを保存
            plt.close()
            #===ここまで1step予測に対する評価処理===

            #===ここから2step予測に対する評価処理===
            loss2 = pred2 - valLab2 #予測誤差を計算
            square_loss2 = loss2**2 #予測誤差の2乗を計算
            MSE2 = np.nanmean(square_loss2,axis=0)#列ごとに平均 予測誤差の2乗平均を計算
            RMSE2 = np.sqrt(MSE2)*110 #RMSEを計算
            MAE2 = np.nanmean(np.sqrt(square_loss2),axis=0)*110 #MAEを計算
            SDAE2 = np.sqrt(np.nanmean((MAE2 - loss2)**2,axis=0)) #Standard Deviation Absolute Error 絶対誤差の標準偏差を計算
            # 散布図を描画
            print("Now drawing scatters...")
            if len(valLab2)>20000:#データ数が20000を超える場合
                random.seed(1)#間引く用の乱数の種
                cut_valLab2 = np.array(random.sample(valLab2.tolist(),20000))
                random.seed(1)#間引く用の乱数の種
                cut_pred2     = np.array(random.sample(pred2.tolist(),20000))
            else:
                cut_valLab2 = copy.deepcopy(valLab2)
                cut_pred2 = copy.deepcopy(pred2)
            plt.scatter(cut_valLab2[:,0]*110, cut_pred2[:,0]*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE2[0],2))+' MAE:'+str(round(MAE2[0],2))+' SDAE:'+str(round(SDAE2[0],2))) #グラフタイトルを指定
            plt.savefig(result_path2 + val_mode + "_self.png")#グラフを保存
            plt.close()
            # 散布図を描画
            print("Now drawing scatters...")
            plt.scatter(cut_valLab2[:,1]*110, cut_pred2[:,1]*110,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 110) #横軸の最小値最大値を指定
            plt.ylim(0, 110) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 110], [0, 110], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE2[1],2))+' MAE:'+str(round(MAE2[1],2))+' SDAE:'+str(round(SDAE2[1],2))) #グラフタイトルを指定
            plt.savefig(result_path2 + val_mode + "_ahead.png")#グラフを保存
            plt.close()
            #===ここまで2step予測に対する評価処理===

            
        else:
            print("Errorr of Scatter")
        pass
        return RMSE,MAE,SDAE, RMSE2,MAE2,SDAE2#評価指標を返す




    def set_FolderPath(self):
        '''
        結果を出力するフォルダを設定
        '''
        learntype_origin = str(self.model_Input_Output)+'D_'+'TrainData'+str(self.whole_data_origin)+'_R'+str(self.R_range)

        self.learntype = 'val'+str(self.val_num)+'_'+str(self.model_Input_Output)+'D_'+'TrainData'+str(self.whole_data)+'_R'+str(self.R_range)  #モデルに使用したデータ数
        modelname = 'MVLSTM_Affine'  #モデルの名前は何か

        try:
            param_fname = "k{}_WLen{}_H{}".format(self.hyper_parameter["median"],
                                                  self.hyper_parameter["window_len"],
                                                  self.hyper_parameter["layerH_unit"])

        except KeyError as e:
            raise("{} keyword is missing from the hyperparameter".format(e.args))
        dt_st_fname = self.starttime.strftime(('%Y%m%d-T%H'))
        dt_fname    = self.dt_now.strftime('_%m%d-T%H%M')

        learntype_origin = str(learntype_origin)
        origin_folder = learntype_origin + '_{}step'.format(self.stepnum) + '/'

        #グラフと評価指標結果を入れるためのフォルダを作る
        self.learntype = str(self.learntype)
        self.score_folder_path = os.getcwd() + '/result/pred_' + dt_st_fname + '/' + modelname + '_' + param_fname + dt_fname + '/' + origin_folder + 'cross_val/'
        self.folder_path = os.getcwd() + '/result/pred_' + dt_st_fname + '/' + modelname + '_' + param_fname + dt_fname + '/' + origin_folder + 'cross_val/' + self.learntype + '_{}step'.format(self.stepnum) + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        
    def ExportLossGraph(self):
        '''
        モデルの損失を出力
        '''

        loss_graphpath = self.folder_path + 'loss.png'
        #loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        #plt.savefig(loss_graphpath,tight_layout=True) #tight_layout=Trueは古い記述方法。bbox_inches='tight'が新しい記述方法。
        plt.title(self.learntype + '_loss')
        plt.savefig(loss_graphpath, bbox_inches='tight')
        plt.close()

    def make_valid_scores_csv(self,valid_scores):
        #用意した検証データの割合は何種類か。交差検証1回目のデータの形を見て判断
        val_num = len(valid_scores[0][7])
        
        if isinstance(valid_scores[0][1],float):
            train_header = ["times","trainvalRMSE","trainvalMAE","trainvalSDAE"
                      ,"trainRMSE","trainMAE","trainSDAE"]
            header = train_header
            for i in range(val_num):
                rate = str(i + 1)
                val_header = ["rate"+rate+"検証RMSE","rate"+rate+"検証MAE","rate"+rate+"検証SDAE"]
                header.extend(val_header)

        elif len(valid_scores[0][1])==2:
            train_header = ["times","trainvalRMSE_self","trainvalRMSE_ahead","trainvalMAE_self","trainvalMAE_ahead","trainvalSDAE_self","trainvalSDAE_ahead"
                      ,"trainRMSE_self","trainRMSE_ahead","trainMAE_self","trainMAE_ahead","trainSDAE_self","trainSDAE_ahead"]
            header = train_header
            for i in range(val_num):
                rate = str(i + 1)
                val_header = ["rate"+rate+"検証RMSE_self","rate"+rate+"検証RMSE_ahead",
                              "rate"+rate+"検証MAE_self","rate"+rate+"検証MAE_ahead",
                              "rate"+rate+"検証SDAE_self","rate"+rate+"検証SDAE_ahead"]            
                header.extend(val_header)

        valid_scores = flatten(valid_scores)
        f=open(self.score_folder_path+'cross_valid_scores_1step.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_scores)
        f.close()

    def make_valid_scores2_csv(self,valid_scores):
        #用意した検証データの割合は何種類か。交差検証1回目のデータの形を見て判断
        val_num = len(valid_scores[0][1])
        
        if isinstance(valid_scores[0][1][0],float):
            train_header = ["times"]
            header = train_header
            for i in range(val_num):
                rate = str(i + 1)
                val_header = ["rate"+rate+"検証RMSE","rate"+rate+"検証MAE","rate"+rate+"検証SDAE"]
                header.extend(val_header)

        elif len(valid_scores[0][1][0])==2:
            train_header = ["times"]
            header = train_header
            for i in range(val_num):
                rate = str(i + 1)
                val_header = ["rate"+rate+"検証RMSE_self","rate"+rate+"検証RMSE_ahead",
                              "rate"+rate+"検証MAE_self","rate"+rate+"検証MAE_ahead",
                              "rate"+rate+"検証SDAE_self","rate"+rate+"検証SDAE_ahead"]            
                header.extend(val_header)

        valid_scores = flatten2(valid_scores)
        f=open(self.score_folder_path+'cross_valid_scores_2step.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_scores)
        f.close()

def flatten(complex_list):
    '''
    csvに書き出すために配列の次元を整える
    '''
    #用意した検証データの割合は何種類か。交差検証1回目のデータの形を見て判断
    val_num = len(complex_list[0][7])

    flatten_list=[]

    if isinstance(complex_list[0][1],float):#データ型がfloatかどうか（要素が1つか二つかの判断）
        for k in range(len(complex_list)):
            one_row=[]
            rate_one_row=[[] for i in range(val_num)]#空の他次元配列作成

            for i in range(len(complex_list[k])):
                if i <=6:#6以下は要素が一つ
                    one_row.append(complex_list[k][i])
                else:#7以上は用意した検証データ分の要素が配列でまとまっている。7:RMSE,8:MAE,9:SDAE
                    for j in range(val_num):#用意した割合の種類分ループ
                        rate_one_row[j].append(complex_list[k][i][j])                  

            for L in range(val_num):#用意した割合の種類分ループ
                one_row.extend(rate_one_row[L])

            flatten_list.append(one_row)#1行確定

    elif len(complex_list[0][1])==2:
        for k in range(len(complex_list)):#交差検証の回数分ループ
            one_row=[]
            rate_one_row=[[] for i in range(val_num)]#空の他次元配列作成

            for i in range(len(complex_list[k])):
                if i == 0:#配列の一つ目は交差検証の番号
                    one_row.append(complex_list[k][i])
                elif i>=1 and i<=6:#配列の2つ目以降は2次元モデルの場合は要素が2つずつ
                    one_row.append(complex_list[k][i][0])
                    one_row.append(complex_list[k][i][1])
                else:
                    for j in range(val_num):#用意した割合の種類分ループ
                        rate_one_row[j].append(complex_list[k][i][j][0])
                        rate_one_row[j].append(complex_list[k][i][j][1])

            for L in range(val_num):#用意した割合の種類分ループ
                one_row.extend(rate_one_row[L])

            flatten_list.append(one_row)#1行確定

    return flatten_list

def flatten2(complex_list):
    '''
    csvに書き出すために配列の次元を整える
    '''
    #用意した検証データの割合は何種類か。交差検証1回目のデータの形を見て判断
    val_num = len(complex_list[0][1])

    flatten_list=[]

    if isinstance(complex_list[0][1][0],float):#データ型がfloatかどうか（要素が1つか二つかの判断）
        for k in range(len(complex_list)):
            one_row=[]
            rate_one_row=[[] for i in range(val_num)]#空の他次元配列作成

            for i in range(len(complex_list[k])):
                if i == 0:#0は要素が一つ
                    one_row.append(complex_list[k][i])
                else:#1以上は用意した検証データ分の要素が配列でまとまっている。1:RMSE,2:MAE,3:SDAE
                    for j in range(val_num):#用意した割合の種類分ループ
                        rate_one_row[j].append(complex_list[k][i][j])                  

            for L in range(val_num):#用意した割合の種類分ループ
                one_row.extend(rate_one_row[L])

            flatten_list.append(one_row)#1行確定

    elif len(complex_list[0][1][0])==2:
        for k in range(len(complex_list)):#交差検証の回数分ループ
            one_row=[]
            rate_one_row=[[] for i in range(val_num)]#空の他次元配列作成

            for i in range(len(complex_list[k])):
                if i == 0:#配列の一つ目は交差検証の番号
                    one_row.append(complex_list[k][i])
                else:
                    for j in range(val_num):#用意した割合の種類分ループ
                        rate_one_row[j].append(complex_list[k][i][j][0])
                        rate_one_row[j].append(complex_list[k][i][j][1])

            for L in range(val_num):#用意した割合の種類分ループ
                one_row.extend(rate_one_row[L])

            flatten_list.append(one_row)#1行確定

    return flatten_list