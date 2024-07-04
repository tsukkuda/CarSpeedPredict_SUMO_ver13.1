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

class final_validation:
    
    def __init__(self,hyper_parameter,starttime,dt_now,stepnum,model_Input_Output,whole_data,model,R_range):
        self.hyper_parameter = hyper_parameter
        self.starttime = starttime
        self.dt_now = dt_now
        self.stepnum = stepnum
        self.model_Input_Output = model_Input_Output
        self.whole_data = whole_data
        self.folder_path =  '.'
        self.model = model
        self.histry = None
        self.learntype = None
        self.R_range = R_range


    def Final_velify(self,validIn,validLab,val_mode,model_Input_Output):
        '''
        検証データでの予測精度の検証
        '''
        #出力フォルダの指定と作成
        result_path1 = self.folder_path + "1step_validation/"
        if not os.path.exists(result_path1):
            os.makedirs(result_path1)
        result_path2 = self.folder_path + "2step_validation/"
        if not os.path.exists(result_path2):
            os.makedirs(result_path2)
        
        if model_Input_Output==1:#次元が1の場合

            print("Now predicting for validation...")
            valIn = np.delete(validIn, 1, 2)#前方平均の列を削除
            pred = self.model.predict(valIn)#予測値を全て計算

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
            RMSE = np.sqrt(MSE)*120 #RMSEを計算
            MAE = np.nanmean(np.sqrt(square_loss))*120 #MAEを計算
            SDAE = np.sqrt(np.nanmean((MAE - loss)**2)) #Standard Deviation Absolute Error 絶対誤差の標準偏差を計算
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
            plt.scatter(cut_valLab1*120, cut_pred*120,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 120) #横軸の最小値最大値を指定
            plt.ylim(0, 120) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 120], [0, 120], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE,2))+' MAE:'+str(round(MAE,2))+' SDAE:'+str(round(SDAE,2))) #グラフタイトルを指定
            plt.savefig(result_path1 + val_mode + "_self.png")#グラフを保存
            plt.close()
            #===ここまで1step予測に対する評価処理===

            #===ここから2step予測に対する評価処理===
            loss2 = pred2 - valLab2 #予測誤差を計算
            square_loss2 = loss2**2 #予測誤差の2乗を計算
            MSE2 = np.nanmean(square_loss2) #予測誤差の2乗平均を計算
            RMSE2 = np.sqrt(MSE2)*120 #RMSEを計算
            MAE2 = np.nanmean(np.sqrt(square_loss2))*120 #MAEを計算
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
            plt.scatter(cut_valLab2*120, cut_pred2*120,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 120) #横軸の最小値最大値を指定
            plt.ylim(0, 120) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 120], [0, 120], color='r', lw=1) #対角線を赤で描画
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
            RMSE = np.sqrt(MSE)*120 #RMSEを計算
            MAE = np.nanmean(np.sqrt(square_loss),axis=0)*120 #MAEを計算
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
            plt.scatter(cut_valLab1[:,0]*120, cut_pred[:,0]*120,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 120) #横軸の最小値最大値を指定
            plt.ylim(0, 120) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 120], [0, 120], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE[0],2))+' MAE:'+str(round(MAE[0],2))+' SDAE:'+str(round(SDAE[0],2))) #グラフタイトルを指定
            plt.savefig(result_path1 + val_mode + "_self.png")#グラフを保存
            plt.close()
            # 散布図を描画
            print("Now drawing scatters...")
            plt.scatter(cut_valLab1[:,1]*120, cut_pred[:,1]*120,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 120) #横軸の最小値最大値を指定
            plt.ylim(0, 120) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 120], [0, 120], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE[1],2))+' MAE:'+str(round(MAE[1],2))+' SDAE:'+str(round(SDAE[1],2))) #グラフタイトルを指定
            plt.savefig(result_path1 + val_mode + "_ahead.png")#グラフを保存
            plt.close()
            #===ここまで1step予測に対する評価処理===

            #===ここから2step予測に対する評価処理===
            loss2 = pred2 - valLab2 #予測誤差を計算
            square_loss2 = loss2**2 #予測誤差の2乗を計算
            MSE2 = np.nanmean(square_loss2,axis=0)#列ごとに平均 予測誤差の2乗平均を計算
            RMSE2 = np.sqrt(MSE2)*120 #RMSEを計算
            MAE2 = np.nanmean(np.sqrt(square_loss2),axis=0)*120 #MAEを計算
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
            plt.scatter(cut_valLab2[:,0]*120, cut_pred2[:,0]*120,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 120) #横軸の最小値最大値を指定
            plt.ylim(0, 120) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 120], [0, 120], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE2[0],2))+' MAE:'+str(round(MAE2[0],2))+' SDAE:'+str(round(SDAE2[0],2))) #グラフタイトルを指定
            plt.savefig(result_path2 + val_mode + "_self.png")#グラフを保存
            plt.close()
            # 散布図を描画
            print("Now drawing scatters...")
            plt.scatter(cut_valLab2[:,1]*120, cut_pred2[:,1]*120,s=1) #プロットするデータとマーカーのサイズを指定
            plt.xlim(0, 120) #横軸の最小値最大値を指定
            plt.ylim(0, 120) #縦軸の最小値最大値を指定
            plt.xlabel("実測値", fontname="MS Gothic") # x 軸のラベルを設定する。
            plt.ylabel("予測値", fontname="MS Gothic") # y 軸のラベルを設定する。
            plt.plot([0, 120], [0, 120], color='r', lw=1) #対角線を赤で描画
            plt.title('RMSE:'+str(round(RMSE2[1],2))+' MAE:'+str(round(MAE2[1],2))+' SDAE:'+str(round(SDAE2[1],2))) #グラフタイトルを指定
            plt.savefig(result_path2 + val_mode + "_ahead.png")#グラフを保存
            plt.close()
            #===ここまで2step予測に対する評価処理===
            
        else:
            print("Errorr of Scatter")
        pass
        return RMSE,MAE,SDAE, RMSE2,MAE2,SDAE2 #評価指標を返す

    def set_FolderPath(self):
        '''
        結果を出力するフォルダを設定
        '''
        self.learntype = str(self.model_Input_Output)+'D_'+'TrainData'+str(self.whole_data)+'_R'+str(self.R_range)  #モデルに使用したデータ数
        modelname = 'MVLSTM_Affine'  #モデルの名前は何か

        try:
            param_fname = "k{}_WLen{}_H{}".format(self.hyper_parameter["median"],
                                                  self.hyper_parameter["window_len"],
                                                  self.hyper_parameter["layerH_unit"])

        except KeyError as e:
            raise("{} keyword is missing from the hyperparameter".format(e.args))
        dt_st_fname = self.starttime.strftime(('%Y%m%d-T%H'))
        dt_fname    = self.dt_now.strftime('_%m%d-T%H%M')

        #グラフと評価指標結果を入れるためのフォルダを作る
        self.learntype = str(self.learntype)
        self.folder_path = os.getcwd() + '/result/pred_' + dt_st_fname + '/' + modelname + '_' + param_fname + dt_fname + '/' + self.learntype + '_{}step'.format(self.stepnum) + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)


    def make_valid_scores_csv(self,valid_scores,step_num):
        #用意した検証データの割合は何種類か。交差検証1回目のデータの形を見て判断
        val_num = len(valid_scores[0])
        
        if isinstance(valid_scores[0][0],float):
            header = []
            for i in range(val_num):
                rate = str(i + 1)
                val_header = ["rate"+rate+"検証RMSE","rate"+rate+"検証MAE","rate"+rate+"検証SDAE"]
                header.extend(val_header)

        elif len(valid_scores[0][0])==2:
            header = []
            for i in range(val_num):
                rate = str(i + 1)
                val_header = ["rate"+rate+"検証RMSE_self","rate"+rate+"検証RMSE_ahead",
                              "rate"+rate+"検証MAE_self","rate"+rate+"検証MAE_ahead",
                              "rate"+rate+"検証SDAE_self","rate"+rate+"検証SDAE_ahead"]            
                header.extend(val_header)

        valid_scores = flatten(valid_scores)
        f=open(self.folder_path+'valid_scores_'+step_num+'step.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_scores)
        f.close()

def flatten(complex_list):
    '''
    csvに書き出すために配列の次元を整える
    '''
    #用意した検証データの割合は何種類か。交差検証1回目のデータの形を見て判断
    val_num = len(complex_list[0])

    flatten_list=[]

    if isinstance(complex_list[0][0],float):#データ型がfloatかどうか（要素が1つか二つかの判断）
        
        one_row=[]
        rate_one_row=[[] for i in range(val_num)]#空の他次元配列作成

        for i in range(len(complex_list)):
            #0:RMSE,1:MAE,2:SDAE
            for j in range(val_num):#用意した割合の種類分ループ
                rate_one_row[j].append(complex_list[i][j])                  

        for L in range(val_num):#用意した割合の種類分ループ
            one_row.extend(rate_one_row[L])

        flatten_list.append(one_row)#1行確定

    elif len(complex_list[0][0])==2:

        one_row=[]
        rate_one_row=[[] for i in range(val_num)]#空の他次元配列作成

        for i in range(len(complex_list)):           
            for j in range(val_num):#用意した割合の種類分ループ
                rate_one_row[j].append(complex_list[i][j][0])
                rate_one_row[j].append(complex_list[i][j][1])

        for L in range(val_num):#用意した割合の種類分ループ
            one_row.extend(rate_one_row[L])

        flatten_list.append(one_row)#1行確定

    return flatten_list