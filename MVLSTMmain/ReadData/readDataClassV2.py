from abc import ABCMeta,abstractmethod

import math
import os
import sys
import pickle
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
from ReadData.split_data_process import split_data_func

class HandlePickel(object):
    def __init__(self,dataname,prepath,ADVrate=None):
        self.prePklPath = dataname
        self.prePath = prepath
        self.originPath = self.prePath+self.prePklPath+'/'+'originaldata.pkl'#訓練データのキャッシュ
        self.valoriginPath = self.prePath+self.prePklPath+'/'+'originalvaldata'+str(ADVrate)+'.pkl'#検証データのキャッシュ

    def LoadPickelData(self):
        #pklファイルがある場合は、pklファイルから元データ読み込み
        #pklファイルがない場合は、元データが格納されているフォルダ内のcsvから読み込み、pklファイルを作成するメソッドを発火させるためにNoneを代入        
        #訓練データに対する処理
        if not os.path.exists(self.originPath):
            datalist = None
        else:
            #pklファイル読み込み
            with open(self.originPath,'rb') as originpkl:
                datalist = pickle.load(originpkl)
        
        return datalist

    def LoadPickelData2(self):
        #検証データに対する処理
        if not os.path.exists(self.valoriginPath):
            valdatalist = None
        else:
            #pklファイル読み込み
            with open(self.valoriginPath,'rb') as valoriginpkl:
                valdatalist = pickle.load(valoriginpkl)

        return valdatalist

    def DumpPickelData(self,datalist):
        #pklファイルを入れるためのフォルダを作る
        if not os.path.exists(self.prePath+self.prePklPath):
            os.mkdir(self.prePath+self.prePklPath)
        #pklファイルを作る
        with open(self.originPath, 'wb') as originpkl:
            pickle.dump(datalist,originpkl)


    def DumpPickelData2(self,datalist):
        #pklファイルを入れるためのフォルダを作る
        if not os.path.exists(self.prePath+self.prePklPath):
            os.mkdir(self.prePath+self.prePklPath)
        #pklファイルを作る
        with open(self.valoriginPath, 'wb') as originpkl:
            pickle.dump(datalist,originpkl)

class ReadPortalTemplate(metaclass=ABCMeta):
    """
    Portalを読み込むためのテンプレートクラス
    """
    #Pickelを保存するフォルダの名前指定
    dataname = None
    #実行しているファイルの場所を取得
    prePath  = None 
    #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
    dataPath = None

    datalist = None

    valdatalist = None

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def readMatrix(self):
        """
        Portalのcsvファイルを読み込む前に、Pickelにキャッシュがあるかどうか確認する
        なければPortalのcsvファイルを読み込む
        """
        #訓練データ読み込み
        hp = HandlePickel(self.dataname,self.prePath)
        self.datalist = hp.LoadPickelData()#datalistに読み込んだデータを詰める。キャッシュが無いとNoneのまま。
        if self.datalist == None:#キャッシュが無い場合
            self.datalist = self.readFileCSV(self.prePath,self.dataPath)#CSVを読み込む
            hp.DumpPickelData(self.datalist)#キャッシュを作る
            if self.datalist == None:#CSVの読み込みに失敗した時
                print('Reading Original Data is failed.\n')
                sys.exit(0)   

    def readMatrix_for_val(self,ADVrate):
        #検証用データ読み込み
        hp2 = HandlePickel(self.dataname,self.prePath,ADVrate=ADVrate)
        self.valdatalist = hp2.LoadPickelData2()#datalistに読み込んだデータを詰める。キャッシュが無いとNoneのまま。
        if self.valdatalist == None:#キャッシュが無い場合
            self.valdatalist = self.readFileCSV(self.prePath,self.valdataPath)#CSVを読み込む
            hp2.DumpPickelData2(self.valdatalist)#キャッシュを作る
            if self.valdatalist == None:#CSVの読み込みに失敗した時
                print('Reading Original Data for validation is failed.\n')
                sys.exit(0) 


    def readFileCSV(self,prePath,dataPath):
        """
        readMatrixのサブクラス
        Portalのcsvファイルを読み込む
        """
        print()
        print("Now reading csv files...")
        #ファイル名のリストを取得
        dataPathlist = glob.glob(prePath+dataPath)
        #この表記については、リスト内包表記で調べるか、リスト内包表記フォルダ内の.txt参照
        #return [pd.read_csv(pathname) for pathname in tqdm(dataPathlist)]
        
        #内包表記を書き下した。
        res = []
        for pathname in tqdm(dataPathlist):
            if os.path.splitext(pathname)[1]=='.csv':#csvファイルのみ対応
                df = pd.read_csv(pathname) #csvを読み込み直後
                file_name = os.path.basename(pathname) #csvファイル名を取得
                date = file_name[1:9] #csvファイルの作成時間の部分を抽出
                df['date'] = int(date) #csv作成日時を記した列を追加
                res.append(df)

        return res
        
    @abstractmethod
    def getSpeedAll(self):
        raise NotImplementedError

class Glisan2loc20sec2hourResolution(ReadPortalTemplate):
    def __init__(self,window,ADVrate=None):
        #Pickelを保存するフォルダの名前指定
        #self.dataname = 'Glisan20sec2hour2locResolution3min' 
        #変更点 こちらもクラス名から分けるのがベスト
        #もしくは実行ファイルからまとめて変更できるようにするのがよいかも
        self.dataname = 'TrackedCarData'
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        # self.dataPath = '20180213-20180619Glisan-Halsey2hourSV_20sec/*'
        #変更点 本当はクラス名から分けたほうがよさそう
        self.dataPath = '車両追跡データ/*'
        self.valdataPath = '検証用車両追跡データ/ADV比率'+str(ADVrate)+'割/*'
        # 20秒ごとのデータを何個まとめるか指定
        self.window   = window   

    def adapterResolutionChange(self,df,window):
        #速度データとIDデータ、車間距離のデータを分ける
        #loc_val = int(len(df.columns) - 1)
        #TODO 位置を追加したため一つずれる
        loc_val = int(len(df.columns) - 2)
        res = {}

        for i in range(loc_val):
            #ここでspeedとvolumeを切り出している
            #TODO
            #速度の差分を利用しようとしている
            #elseは差分を計算する列
            #条件式で差分かそのままか制御を分けるようにする


            #TODO 位置情報を追加したバージョンからspeedとvolumeを取得
            changed_df = self.resolutionChange(speed  = df.iloc[:,2+i],
                                                volume = df.iloc[:,2+i],
                                                #speed = df.iloc[:,2*i],
                                                #volume = df.iloc[:,2*i+1],
                                                window = window)

            changed_dict = changed_df.to_dict(orient="list")
            res.update(changed_dict)

        return pd.DataFrame(res)

    @staticmethod
    def resolutionChange(speed,volume,window):
        """
        change to any resolution
        parameters
        ----------
            speed:   pd.Series

            volume:  pd.Series

            window: int
            何個分値をまとめるかを指定
        return
        ------
            計算されたpd.Series(ただし、各場所のvolume時系列は出力されません)
        """
        def summerize_list(speed,volume,window_size):
            res = []
            #math.ceil : float型の小数点以下切り上げ
            for i in range(math.ceil(len(speed)-1/window_size)-2):
                slice_begin = window_size * i + 1
                slice_last  = window_size * (i + 1) + 1
                speed_nd  = np.array(speed[slice_begin:slice_last])
                volume_nd = np.array(volume[slice_begin:slice_last])
                doted_speed = np.nansum(speed_nd*volume_nd)
                sum_volume  = np.nansum(volume_nd)
                #TODO 
                # 0除算への一時的な処置 NA(np.nan) のどちらかに設定したほうがよさそう
                """
                if sum_volume == 0:
                    res.append(None)
                else:
                    #res.append(doted_speed/sum_volume)
                    res.append(speed_nd)
                """
                #TODO 追跡データへの一時的な対応
                res.append(np.average(speed_nd))

            return res

        list_speed  = list(speed)
        list_volume = list(volume)

        dict_summerized = {}
        dict_summerized[speed.name] = summerize_list(list_speed,
                                                     list_volume,
                                                     window)

        res = pd.DataFrame(dict_summerized)
        return res

    @staticmethod
    def resolutionChange2(speed,volume,window):
        """
        change to any resolution
        parameters
        ----------
            speed:   pd.Series

            volume:  pd.Series

            window: int
            何個分値をまとめるかを指定
        return
        ------
            計算されたpd.Series(ただし、各場所のvolume時系列は出力されません)
        """
        def summerize_list(speed,volume,window_size):
            res = []
            #math.ceil : float型の小数点以下切り上げ
            for i in range(math.ceil(len(speed)-1/window_size) - 2):
                slice_begin = window_size * i + 1
                slice_last  = window_size * (i + 1) + 1
                speed_nd  = np.array(speed[slice_begin:slice_last]) - np.array(speed[slice_begin-1:slice_last-1])
                volume_nd = np.array(volume[slice_begin:slice_last])
                doted_speed = np.nansum(speed_nd*volume_nd)
                sum_volume  = np.nansum(volume_nd)
                #TODO 
                # 0除算への一時的な処置 NA(np.nan) のどちらかに設定したほうがよさそう
                """
                if sum_volume == 0:
                    res.append(None)
                else:
                    #res.append(doted_speed/sum_volume)
                    res.append(speed_nd)
                """
                #TODO 追跡データへの一時的な対応
                res.append(np.average(speed_nd))

            return res

        list_speed  = list(speed)
        list_volume = list(volume)

        dict_summerized = {}
        dict_summerized[speed.name] = summerize_list(list_speed,
                                                     list_volume,
                                                     window)

        res = pd.DataFrame(dict_summerized)
        return res

    #Glisan-Halsey(0.9km間)の20秒毎の全速度データを1日ごと16日分取得
    def getSpeedAll(self):
        print()
        print("Now splitting the Time, ID, Position and Velocity from the Data")
        # iloc[行, 列]でdataFrame型からデータを切り出せる
        #origin = [self.adapterResolutionChange(self.datalist[i].iloc[:,1:],self.window) for i in tqdm(range(0,len(self.datalist)))]
        origin = split_data_func(self.datalist,self.window)
        # origin = [self.adapterResolutionChange(self.datalist[i].iloc[:,1:5],self.window) for i in range(0,len(self.datalist))]
        #originはDataFrame型
        return origin

    def getSpeedAll2(self):
        print()
        print("Now splitting the Time, ID, Position and Velocity from the Data")
        # iloc[行, 列]でdataFrame型からデータを切り出せる
        #origin = [self.adapterResolutionChange(self.valdatalist[i].iloc[:,1:],self.window) for i in tqdm(range(0,len(self.valdatalist)))]
        origin = split_data_func(self.valdatalist,self.window)
        # origin = [self.adapterResolutionChange(self.valdatalist[i].iloc[:,1:5],self.window) for i in range(0,len(self.valdatalist))]
        #originはDataFrame型
        return origin

class ReadMatrixContext(object):
    #戦略を与える
    def __init__(self,strategy):
        self.strategy = strategy

    def ReadSpdMatrixlist(self):
        self.strategy.readMatrix()
        res = self.strategy.getSpeedAll()
        #resはDataFrame型
        return res

    def ReadSpdMatrixlist2(self,ADVrate):
        self.strategy.readMatrix_for_val(ADVrate)
        res = self.strategy.getSpeedAll2()
        #resはDataFrame型
        return res
