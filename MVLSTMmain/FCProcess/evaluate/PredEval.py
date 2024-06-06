import csv
import json
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from pandas.core.indexes.period import PeriodDelegateMixin
from sklearn.metrics import mean_absolute_error,mean_squared_error

import sys
import pathlib
# このファイルがあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append(str(current_dir) + '/./')

import Mytools.graphdrawer

# 予測結果を元に評価指標の記録・出力や、予測結果のグラフを出力するクラス
class PredEval():
    def __init__(self,sample_num,hyper_param):
        #TODO 20191104 評価指標計算結果用のメンバ変数が多すぎる。もっときれいな実装方法があるはず
        #結果を出力するフォルダのパスを保持する
        self.folder_path = '.'
        self.learntype = ''

        #ハイパーパラメータ保持
        self.hyper_param = hyper_param

        #出力用ファイルのindex作成
        self.output_index = np.empty(sample_num)
        for i in range(sample_num):
            self.output_index[i] = i

        #評価指標計算結果用の変数初期化
        self.jam_start           = np.empty(sample_num,dtype=int)
        self.fc_jam_start        = np.empty(sample_num,dtype=int)
        self.Entire_RMSE         = np.empty(sample_num)
        self.Entire_MAE          = np.empty(sample_num)
        self.jam11_naive_RMSE    = np.empty(sample_num)
        self.jam11_naive_MAE     = np.empty(sample_num)
        self.jam11_RMSE          = np.empty(sample_num)
        self.jam11_MAE           = np.empty(sample_num)
        #TODO 全体のnaive_MAE,naive_RMSEを追加
        self.Entire_naive_RMSE   = np.empty(sample_num)
        self.Entire_naive_MAE    = np.empty(sample_num)

        #渋滞開始点記録回数カウント
        self.Record_cnt = 0

    #グラフ保存&評価指標結果保存のフォルダを設定&作成
    def SetFolderpath(self,modelname,hyper_param,starttime,dt_now,learntype,pred_num):
        try:
            param_fname = "k{}_WLen{}_H{}".format(hyper_param["median"],
                                                  hyper_param["window_len"],
                                                  hyper_param["layerH_unit"])

        except KeyError as e:
            raise("{} keyword is missing from the hyperparameter".format(e.args))
        dt_st_fname = starttime.strftime(('%Y%m%d-T%H'))
        dt_fname    = dt_now.strftime('_%m%d-T%H%M')

        #グラフと評価指標結果を入れるためのフォルダを作る
        self.learntype = str(learntype)
        self.folder_path = os.getcwd() + '/result/pred_' + dt_st_fname + '/' + modelname + '_' + param_fname + dt_fname + '/' + self.learntype + '_{}step'.format(pred_num) + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    #設定したフォルダのパスを取得
    def GetFolderpath(self):
        return self.folder_path

    #モデルの損失を出力
    def ExportLossGraph(self,history):
        loss_graphpath = self.folder_path + '/' + self.learntype + '_loss.png'
        #loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        #plt.savefig(loss_graphpath,tight_layout=True) #tight_layout=Trueは古い記述方法。bbox_inches='tight'が新しい記述方法。
        plt.savefig(loss_graphpath, bbox_inches='tight')
        plt.close()

    #RecordMetricsのヘルパーメソッド
    #渋滞の開始を見つけ出し、開始時刻(実際はステップ数)を記録
    def RecordJamStart(self,y_metrics,jamspeed,detect_start):
        #渋滞発生の時間を検出して、その時間を返す
        #jamspeedは渋滞と判断する基準値である
        def DetectJamStart(series,jamspeed,detect_start):
            jam_start = 0
            for i in range(detect_start,len(series)):
                with np.errstate(invalid='ignore'): #Nanだけのnumpy配列をスライスした時の警告文を表示しないようにする
                    if jamspeed > series.iat[i] and series.iat[i] > 0:
                        jam_start = i
                        break

            return jam_start

        #渋滞開始点取得
        jam_start    = DetectJamStart(y_metrics['y_true'],jamspeed,detect_start)
        fc_jam_start = DetectJamStart(y_metrics['y_pred'],jamspeed,detect_start)

        #渋滞開始点記録
        try:
            self.jam_start[self.Record_cnt]    = jam_start
            self.fc_jam_start[self.Record_cnt] = fc_jam_start
        except IndexError:
            raise IndexError('There are too many times to record compared to the number of sample data.')

    #RecordMetricsのヘルパーメソッド
    #予測結果をグラフにして出力する
    def ExportPredGraph(self,y_metrics,y_metrics2,origindata):
        #保存するグラフのファイル名指定
        graph_fname         = '/' + self.learntype + str(self.Record_cnt) + '.png'
        graphdetail_fname   = '/' + self.learntype + str(self.Record_cnt) + 'detail' + '.png'

        #予測結果全体のグラフ描画
        Mytools.graphdrawer.savegraphdraw(x1=range(0,len(origindata)),
                                            y1=origindata.iloc[:,0],
                                            label1="car_speed",
                                            y2=origindata.iloc[:,1],
                                            label2="average_speed",
                                            #y3=origindata.iloc[:,2],
                                            #label3="car3",
                                            #y4=origindata.iloc[:,3],
                                            #label4="car4",
                                            #y5=origindata.iloc[:,4],
                                            #label5="car5",
                                            y6=y_metrics['y_pred'],#自車速度の予測値
                                            label6="predicted_car_speed",
                                            y7=y_metrics2['y_pred'],#前方車両の平均速度の予測値
                                            label7="predict_average_speed",
                                            savepath=self.folder_path,
                                            figurename=graph_fname)

        #渋滞開始付近のグラフを拡大して表示する範囲。
        #例:10なら渋滞開始から前後10ステップずつ、20ステップを表示
        jam_graphrange = 10

        #渋滞開始付近のグラフ描画
        '''
        Mytools.graphdrawer.savegraphdraw(x1=range(0,len(origindata)),
                                            y1=origindata.iloc[:,0],
                                            label1="car_speed",
                                            # y2=origindata.iloc[:,1],
                                            # label2="average_speed",
                                            #y3=origindata.iloc[:,2],
                                            #label3="car3",
                                            #y4=origindata.iloc[:,3],
                                            #label4="car4",
                                            #y5=origindata.iloc[:,4],
                                            #label5="car5",
                                            y6=y_metrics['y_pred'],
                                            label6="predict",
                                            xlim=(self.jam_start[self.Record_cnt]-jam_graphrange,
                                                  self.jam_start[self.Record_cnt]+jam_graphrange),
                                            savepath=self.folder_path,
                                            figurename=graphdetail_fname)
        '''

    #RecordMetricsのヘルパーメソッド
    #評価指標を計算して各記録用配列に格納する
    def ComputeMetrics(self,y_metrics,pred_num=0):
        #渋滞開始点から前後何個の値を使って指標評価するかを指定
        #ex:5の場合は渋滞開始時の値+前後5個のデータの計11個のデータで指標を計算する
        #TODO 渋滞後のデータよりもその手前のデータに重きを置きたい
        
        metrics_range11 = 5
        jam11_first         = self.jam_start[self.Record_cnt]-metrics_range11
        jam11_last          = self.jam_start[self.Record_cnt]+metrics_range11
        jam11naive_first    = self.jam_start[self.Record_cnt]-metrics_range11+1
        jam11naive_last     = self.jam_start[self.Record_cnt]+metrics_range11+1

        #RMSE等はnanがあると結果が出ないため、nanを落とす
        y_metrics.dropna(inplace=True)
        #渋滞開始が検出できなかった場合は無視して次に進む
        #RMSE計算
        try:
            self.Entire_naive_RMSE[self.Record_cnt] = np.sqrt(mean_squared_error(y_metrics['y_true'].iloc[:-pred_num-1],
                                                                                y_metrics['y_true'].iloc[pred_num+1:]))
        except ValueError:
            self.Entire_naive_RMSE[self.Record_cnt] = None
        try:
            self.Entire_RMSE[self.Record_cnt]       = np.sqrt(mean_squared_error(y_metrics['y_true'], y_metrics['y_pred']))
        except ValueError:
            self.Entire_RMSE[self.Record_cnt]       = None
        try:    
            self.jam11_naive_RMSE[self.Record_cnt]  = np.sqrt(mean_squared_error(y_metrics['y_true'].iloc[jam11_first:jam11_last],
                                                                                y_metrics['y_true'].iloc[jam11naive_first:jam11naive_last]))
        except ValueError:
            self.jam11_naive_RMSE[self.Record_cnt]  = None
        try:
            self.jam11_RMSE[self.Record_cnt]        = np.sqrt(mean_squared_error(y_metrics['y_true'].iloc[jam11_first:jam11_last],
                                                                                y_metrics['y_pred'].iloc[jam11_first:jam11_last]))
        except ValueError:
            self.jam11_RMSE[self.Record_cnt]        = None

        #MAE計算
        try:
            self.Entire_naive_MAE[self.Record_cnt]  = mean_absolute_error(y_metrics['y_true'].iloc[:-pred_num-1],
                                                                         y_metrics['y_true'].iloc[pred_num+1:])
        except ValueError:
            self.Entire_naive_MAE[self.Record_cnt]  = None
        try:
            self.Entire_MAE[self.Record_cnt]        = mean_absolute_error(y_metrics['y_true'], y_metrics['y_pred'])
        except ValueError:
            self.Entire_MAE[self.Record_cnt]        = None
        try:
            self.jam11_naive_MAE[self.Record_cnt]   = mean_absolute_error(y_metrics['y_true'].iloc[jam11_first:jam11_last],
                                                                         y_metrics['y_true'].iloc[jam11naive_first:jam11naive_last])
        except ValueError:
            self.jam11_naive_MAE[self.Record_cnt]   = None
        try:
            self.jam11_MAE[self.Record_cnt]         = mean_absolute_error(y_metrics['y_true'].iloc[jam11_first:jam11_last],
                                                                         y_metrics['y_pred'].iloc[jam11_first:jam11_last])
        except ValueError:
            self.jam11_MAE[self.Record_cnt]         = None
            #pass



    #評価指標を計算して記録するメソッド
    #初期化したインスタンスに評価指標の記録をしたい場合はこのメソッドを使用すること
    def RecordMetrics(self,pred_speed,origindata,jamspeed,pred_index=0,pred_range=0,pred_num=0):
        #元データと予測結果の2列で行列を作成
        try:
            #自車速度
            y_metrics = pd.DataFrame({'y_true':origindata.iloc[:,0].reset_index(drop=True),
                                      'y_pred':pd.Series(pred_speed[:,0].reshape(len(pred_speed)),
                                      index=range(self.hyper_param["window_len"]+pred_index+pred_num,len(origindata)))})
            #前方範囲平均速度
            y_metrics2 = pd.DataFrame({'y_true':origindata.iloc[:,1].reset_index(drop=True),
                                      'y_pred':pd.Series(pred_speed[:,1].reshape(len(pred_speed)),
                                      index=range(self.hyper_param["window_len"]+pred_index+pred_num,len(origindata)))})
        except KeyError as e:
            raise KeyError("{} keyword is missing from the hyperparameter".format(e.args))

        #予測結果を元データとともに出力
        #pred_fname = '/' + self.learntype + str(self.Record_cnt) + '_pred-result.csv'
        #y_metrics.to_csv(self.folder_path + pred_fname)

        #渋滞の開始を見つけ出し、開始時刻(実際はステップ数)を記録
        self.RecordJamStart(y_metrics=y_metrics,jamspeed=jamspeed,detect_start=self.hyper_param["window_len"])

        #予測結果をグラフにしたものを出力
        self.ExportPredGraph(y_metrics=y_metrics,y_metrics2=y_metrics2,origindata=origindata)

        #評価指標を計算&記録
        self.ComputeMetrics(y_metrics=y_metrics, pred_num=pred_num)

        self.Record_cnt += 1

    #評価指標計算結果をファイルに出力する
    def ExportMetrics(self):
        #TODO 20191104 変数過多なので減らす方法を検討する必要あり
        #指標計算結果を整理して結合
        output_colname = ['trainNo.',
                          '渋滞開始',
                          '渋滞開始予測',
                          'Entire_naive_RMSE',
                          'Entire_naive_MAE',
                          'Entire_RMSE',
                          'Entire_MAE',
                          'jam11_naive_RMSE',
                          'jam11_naive_MAE',
                          'jam11_RMSE',
                          'jam11_MAE']
        output_metrics = np.stack([self.output_index,
                                   self.jam_start,
                                   self.fc_jam_start,
                                   self.Entire_naive_RMSE,
                                   self.Entire_naive_MAE,
                                   self.Entire_RMSE,
                                   self.Entire_MAE,
                                   self.jam11_naive_RMSE,
                                   self.jam11_naive_MAE,
                                   self.jam11_RMSE,
                                   self.jam11_MAE],
                                   axis=1)
        output_average = ["average",
                          "",
                          "",
                          np.nanmean(self.Entire_naive_RMSE),
                          np.nanmean(self.Entire_naive_MAE),
                          np.nanmean(self.Entire_RMSE),
                          np.nanmean(self.Entire_MAE),
                          np.nanmean(self.jam11_naive_RMSE),
                          np.nanmean(self.jam11_naive_MAE),
                          np.nanmean(self.jam11_RMSE),
                          np.nanmean(self.jam11_MAE)
                          ]
        #ファイル出力
        metrics_fname = self.folder_path + '/' + self.learntype + 'metrics.csv'
        hparam_fname  = self.folder_path + '/' + self.learntype + 'hyperparam.json'
 
        with open(metrics_fname, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_colname)
            writer.writerows(output_metrics)
            writer.writerow(output_average)
            for key,value in self.hyper_param.items():
                f.write("{}:{}\n".format(key,value))

        with open(hparam_fname,'w') as hf:
            json.dump(self.hyper_param,hf)
    
    #TODO 予測結果をまとめて出力するファイルを作成
    def ExportPredSpeed(self, pred_speed):
        #ファイル出力
        metrics_fname = self.folder_path + '/pred_speed.csv'
        index = os.listdir(os.getcwd()+"/ReadData/車両追跡データ")
        y_metrics = pd.DataFrame(pred_speed, index=index[0:-1])
        y_metrics.to_csv(metrics_fname)
        # os.getlistとか使ってデータのファイル名を取得してインデックスにしたほうがいいかも
    
    def AdjustTimeLabel(self, pred_speed):
        metrics_fname = self.folder_path + '/pred_speed.csv'
        data_path =  os.getcwd()+"/ReadData/車両追跡データ"
        original_file_list = os.listdir(data_path)
        i = 0
        for fname in original_file_list:
            df = pd.read_csv(data_path+"/"+fname, nrows=1)
            #TODO 5秒1ステップで読み込むので位置をそろえるために5で割って切り捨て除算
            #最終的な1ステップ分のサイズで割る
            beginning_time = df.iat[0,0] // 5
            for j in range(beginning_time):
                pred_speed[i].insert(0, None)
            i = i + 1
        
        y_metrics = pd.DataFrame(pred_speed, index=original_file_list)
        y_metrics.to_csv(metrics_fname)
    
    def ExportOriginSpeed(self, origin_speed):
        metrics_fname = self.folder_path + '/origin_speed.csv'
        data_path =  os.getcwd()+"/ReadData/車両追跡データ"
        original_file_list = os.listdir(data_path)
        i = 0
        for fname in original_file_list:
            df = pd.read_csv(data_path+"/"+fname, nrows=1)
            #TODO 5秒1ステップで読み込むので位置をそろえるために5で割って切り捨て除算
            #最終的な1ステップ分のサイズで割る
            beginning_time = df.iat[0,0] // 5
            for j in range(beginning_time):
                origin_speed[i].insert(0, None)
            i = i + 1
        
        y_metrics = pd.DataFrame(origin_speed, index=original_file_list)
        y_metrics.to_csv(metrics_fname)

    #TODO 渋滞を予見した位置を取得する
    def GetPredJamPoint(self, jam_step_list=[], fname=""):
        data_path =  os.getcwd()+"/ReadData/車両追跡データ"
        pred_jam_point_list = []
        df = pd.read_csv(data_path+"/"+fname)
        start_time =  df.iat[0, 0] // 5

        for jam_step in jam_step_list:
            pred_jam_point_list.append(df.iat[jam_step - start_time + 1, 2])

        return pred_jam_point_list
    
    #TODO 渋滞の予見を判断するプログラム
    def JudgePredJam(self, pred_range):
        threadspeed = 50/3.6    #この速度を下回ると渋滞を予見したとする
        threadsteps = 3         #このステップ数だけ連続でthreadspeedを下回ると渋滞を予見したとする
        threadpoint = 0         #渋滞を予見した車両    

        metrics_fname = self.folder_path + '/pred_position.csv'
        pred_data_path = self.folder_path + "/pred_speed.csv"
        data_path =  os.getcwd()+"/ReadData/車両追跡データ"
        original_file_list = os.listdir(data_path)

        ##予測結果のファイルを読み込む
        #with open(pred_data_path, 'r') as f:
        #    reader = csv.reader(f)
        #    header = next(reader)
        #    #渋滞と判断した時間と位置を保持
        #    #pred_jam_data = {}
        #    pred_jam_data = []
        #    for row in reader:
        #        print(row)
        #        #渋滞を予見した時間を保持
        #        jam_step_list = []
        #        for i in range(1, len(row)):
        #            if float(row[i] or "22.3") < threadspeed:
        #                jam_step_list.append(i-pred_range)
        #        
        #        #渋滞を予見した位置を保持
        #        jam_point_list = self.GetPredJamPoint(jam_step_list, fname=row[0])
        #        #pred_jam_data[row[0]] = {"pred_step": jam_step_list, "pred_point": jam_point_list}
        #        pred_jam_data.append([row[0], jam_step_list, jam_point_list])

        #TODO ファイル出力をこの後に記述する
        with open(pred_data_path, 'r') as fp:
            pred_data = list(csv.reader(fp))
        origin_data = copy.deepcopy(pred_data)

        for i in range(1, len(pred_data)):
            jam_step_list = []

            for j in range(1, len(pred_data[i])-threadsteps+1):
                if pred_data[i][j] == '':
                    continue
                elif float(pred_data[i][j]) > threadspeed:
                    pred_data[i][j] = ''
                else:
                    for n in range(1, threadsteps):
                        if pred_data[i][j+n] == '':
                            pred_data[i][j] = ''
                            break
                        elif float(pred_data[i][j+n]) > threadspeed:
                            pred_data[i][j] = ''
                            break
                        elif n == threadsteps - 1:
                            jam_step_list.append(j+threadsteps-pred_range-2)
                            pred_data[i][j] = ''

                    #jam_step_list.append(j-1-pred_range)

            jam_point_list = self.GetPredJamPoint(jam_step_list, fname=pred_data[i][0])

            for (j, jam_point) in zip(jam_step_list, jam_point_list):
                pred_data[i][j] = jam_point                

        #TODO 渋滞の検知をここでpython上で行うか、Excel上でマクロで行う

        #df = pd.DataFrame(pred_data[1:][1:], index=original_file_list)
        #これでは二次元リストのスライスはできない
        df = pd.DataFrame([line[1:] for line in pred_data[1:]], index=original_file_list)
        df.to_csv(metrics_fname)

    def JudgeOriginJam(self, pred_range):
        threadspeed = 50/3.6    #この速度を下回ると渋滞を予見したとする
        threadsteps = 3         #このステップ数だけ連続でthreadspeedを下回ると渋滞を予見したとする
        threadpoint = 0         #渋滞を予見した車両    

        metrics_fname = self.folder_path + '/origin_position.csv'
        origin_data_path = self.folder_path + "/origin_speed.csv"
        data_path =  os.getcwd()+"/ReadData/車両追跡データ"
        original_file_list = os.listdir(data_path)

        ##予測結果のファイルを読み込む
        #with open(pred_data_path, 'r') as f:
        #    reader = csv.reader(f)
        #    header = next(reader)
        #    #渋滞と判断した時間と位置を保持
        #    #pred_jam_data = {}
        #    pred_jam_data = []
        #    for row in reader:
        #        print(row)
        #        #渋滞を予見した時間を保持
        #        jam_step_list = []
        #        for i in range(1, len(row)):
        #            if float(row[i] or "22.3") < threadspeed:
        #                jam_step_list.append(i-pred_range)
        #        
        #        #渋滞を予見した位置を保持
        #        jam_point_list = self.GetPredJamPoint(jam_step_list, fname=row[0])
        #        #pred_jam_data[row[0]] = {"pred_step": jam_step_list, "pred_point": jam_point_list}
        #        pred_jam_data.append([row[0], jam_step_list, jam_point_list])

        #TODO ファイル出力をこの後に記述する
        with open(origin_data_path, 'r') as fp:
            origin_data = list(csv.reader(fp))

        for i in range(1, len(origin_data)):
            jam_step_list = []

            for j in range(1, len(origin_data[i])-threadsteps+1):
                if origin_data[i][j] == '':
                    continue
                elif float(origin_data[i][j]) > threadspeed:
                    origin_data[i][j] = ''
                else:
                    for n in range(1, threadsteps):
                        if origin_data[i][j+n] == '':
                            origin_data[i][j] = ''
                            break
                        elif float(origin_data[i][j+n]) > threadspeed:
                            origin_data[i][j] = ''
                            break
                        elif n == threadsteps - 1:
                            jam_step_list.append(j+threadsteps-1)
                            origin_data[i][j] = ''

                    #jam_step_list.append(j-1-pred_range)

            jam_point_list = self.GetPredJamPoint(jam_step_list, fname=origin_data[i][0])

            for (j, jam_point) in zip(jam_step_list, jam_point_list):
                origin_data[i][j] = jam_point                

        #TODO 渋滞の検知をここでpython上で行うか、Excel上でマクロで行う

        #df = pd.DataFrame(pred_data[1:][1:], index=original_file_list)
        #これでは二次元リストのスライスはできない
        df = pd.DataFrame([line[1:] for line in origin_data[1:]], index=original_file_list)
        df.to_csv(metrics_fname)