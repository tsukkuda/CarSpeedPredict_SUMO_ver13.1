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
    def __init__(self,dataname,prepath):
        self.prePklPath = dataname
        self.prePath = prepath
        self.originPath = self.prePath+self.prePklPath+'/'+'originaldata.pkl'

    def LoadPickelData(self):
        #pklファイルがある場合は、pklファイルから元データ読み込み
        #pklファイルがない場合は、元データが格納されているフォルダ内のcsvから読み込み、pklファイルを作成するメソッドを発火させるためにNoneを代入        
        if not os.path.exists(self.originPath):
            datalist = None
        else:
            #pklファイル読み込み
            with open(self.originPath,'rb') as originpkl:
                datalist = pickle.load(originpkl)

        return datalist

    def DumpPickelData(self,datalist):
        #pklファイルを入れるためのフォルダを作る
        if not os.path.exists(self.prePath+self.prePklPath):
            os.mkdir(self.prePath+self.prePklPath)
        #pklファイルを作る
        with open(self.originPath, 'wb') as originpkl:
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

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def readMatrix(self):
        """
        Portalのcsvファイルを読み込む前に、Pickelにキャッシュがあるかどうか確認する
        なければPortalのcsvファイルを読み込む
        """
        hp = HandlePickel(self.dataname,self.prePath)
        self.datalist = hp.LoadPickelData()
        if self.datalist == None:
            self.datalist = self.readFileCSV(self.prePath,self.dataPath)
            hp.DumpPickelData(self.datalist)
            if self.datalist == None:
                print('Reading Original Data is failed.\n')
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
'''
class Glisan2loc(ReadPortalTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Glisan2loc'
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath = '20180213-20180605Glisan-Halsey_sv/*'  

    #Glisan-Halsey(0.9km間)の全速度データを1日ごと15日分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,0:2] for i in range(0,len(self.datalist))]

        return origin

class Glisan2loc20sec12hour(ReadPortalTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Glisan20sec12hour2loc' 
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath = '20180213-20180619Glisan-Halsey12hour_20sec/*'    
            
    #Glisan-Halsey(0.9km間)の20秒毎の全速度データを1日ごと16日分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,1:3] for i in range(0,len(self.datalist))]

        return origin

class Glisan2loc20sec2hour(ReadPortalTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Glisan20sec2hour2loc' 
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath = '20180213-20180619Glisan-Halsey2hour_20sec/*'   

    #Glisan-Halsey(0.9km間)の20秒毎の全速度データを1日ごと16日分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,1:3] for i in range(0,len(self.datalist))]

        return origin
'''
class Glisan2loc20sec2hourResolution(ReadPortalTemplate):
    def __init__(self,window):
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
            """
            if i == 0:
                changed_df = self.resolutionChange(speed  = df.iloc[:,1+i],
                                                   volume = df.iloc[:,1+i],
                                                   #speed = df.iloc[:,2*i],
                                                   #volume = df.iloc[:,2*i+1],
                                                   window = window)
            else:
                changed_df = self.resolutionChange2(speed  = df.iloc[:,1+i],
                                                   volume = df.iloc[:,1+i],
                                                   #speed = df.iloc[:,2*i],
                                                   #volume = df.iloc[:,2*i+1],
                                                   window = window)
            """
            """
            changed_df = self.resolutionChange(speed  = df.iloc[:,1+i],
                                                volume = df.iloc[:,1+i],
                                                #speed = df.iloc[:,2*i],
                                                #volume = df.iloc[:,2*i+1],
                                                window = window)
            """

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
'''
class Glisan20sec1hour2loc(ReadPortalTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Glisan20sec1hour2loc' 
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath = '20180213-20180619Glisan-Halsey1hour_20sec/*'    
            
    #Glisan-Halsey(0.9km間)の20秒毎の全速度データを1日ごと16日分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,1:3] for i in range(0,len(self.datalist))]

        return origin

class Norwood2loc(ReadPortalTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Norwood2loc'
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath = '20180123-20180619Norwood-Wilson/*'  

    #Glisan-Halsey(0.9km間)の全速度データを1日ごと15日分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,0:2] for i in range(0,len(self.datalist))]

        return origin
'''

class ReadSimMatrixTemplate(metaclass=ABCMeta):
    """
    Simulation dataを読み込むためのテンプレートクラス
    """
    #Pickelを保存するフォルダの名前指定
    dataname = None
    #実行しているファイルの場所を取得
    prePath  = None 
    #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
    dataPath1 = None
    dataPath2 = None

    datalist = None

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def readMatrix(self):
        """
        Simulation dataのcsvファイルを読み込む前に、Pickelにキャッシュがあるかどうか確認する
        なければSimulation dataのcsvファイルを読み込む
        """
        hp = HandlePickel(self.dataname,self.prePath)
        self.datalist = hp.LoadPickelData()
        if self.datalist == None:
            self.datalist = self.readFileCSV(self.prePath,self.dataPath1,self.dataPath2)
            hp.DumpPickelData(self.datalist)
            if self.datalist == None:
                print('Reading Original Data is failed.\n')
                sys.exit(0)   

    def readFileCSV(self,prePath,dataPath1,dataPath2):
        """
        readMatrixのサブクラス
        Simulation dataのcsvファイルを読み込む
        """
        #ファイル名のリストを取得
        dataPathlist_up  = glob.glob(prePath+dataPath1)
        dataPathlist_dn  = glob.glob(prePath+dataPath2)

        #この表記については、リスト内包表記で調べるか、リスト内包表記フォルダ内の.txt参照
        datalist_up  = [pd.read_csv(pathname,names = ('Times','Speed'),usecols = [4,10]) for pathname in dataPathlist_up]
        datalist_dn  = [pd.read_csv(pathname,names = ('Times','Speed'),usecols = [4,10]) for pathname in dataPathlist_dn]

        timeUnit = 10 #何秒間隔でデータを区切るかを決める

        def timeAggregate2locData(datalist_up,datalist_dn,timeUnit):
            """
            timeAggregate2locData(datalist500,datalist2500,timeUnit)
            500m,2500mのデータそれぞれにおいてtimeAggregateを適用するために関数
            入力
            ----
            datalist500,datalist2500:TimesとSpeedを列ラベルに持つDataframe型
            timeUnit:経過時間の区切る間隔
            出力
            ----
            500m,2500m両方のデータが平均処理されて結合されたDataframe型。中身の要素はSpeedデータ。
            """
            def timeAggregateSPD(inputdata,timeUnit,dataname):
                """
                timeAggregateSPD(inputdata,timeUnit,dataname)
                inputdataのデータを、timeUnitで指定した値ごとにTimeを区切り、それぞれの平均を出して結果を返す。
                入力
                ----
                inputdata:TimesとSpeedを列ラベルに持つDataframe型
                timeUnit:経過時間の区切る間隔
                dataname:出力のSeriesの列ラベル名を決める

                出力
                ----
                平均処理されたSeries型。中身の要素はSpeedデータ。
                """
                endTime = int(round(inputdata.max()[0])) // timeUnit + 1 #終了時間
                #timeUnitで指定した値ごとにTimeを区切り、それぞれの平均を出してリスト化
                result = []
                for level in range(0,endTime):
                    temp = inputdata[(inputdata['Times'] >= level * timeUnit) & (inputdata['Times'] < (level + 1) * timeUnit)]
                    result.append(temp.mean()[1])
                result = pd.Series(result,name = dataname)
                result.fillna(0,inplace=True) #欠損値を0に置き換える(車が通っていないため、speed=0の扱い)
                return result

            def timeAggregateVOL(inputdata,timeUnit,dataname):
                """
                timeAggregateVOL(inputdata,timeUnit,dataname)
                inputdataのデータを、timeUnitで指定した値ごとにTimeを区切り、それぞれの通過した台数を出して結果を返す。
                入力
                ----
                inputdata:TimesとSpeedを列ラベルに持つDataframe型
                timeUnit:経過時間の区切る間隔
                dataname:出力のSeriesの列ラベル名を決める
                出力
                ----
                平均処理されたSeries型。中身の要素はVolumeデータ。
                """
                endTime = int(round(inputdata.max()[0])) // timeUnit + 1 #終了時間
                #timeUnitで指定した値ごとにTimeを区切り、それぞれの平均を出してリスト化
                result = []
                for level in range(0,endTime):
                    temp = inputdata[(inputdata['Times'] >= level * timeUnit) & (inputdata['Times'] < (level + 1) * timeUnit)]
                    result.append(len(temp))
                result = pd.Series(result,name = dataname)
                result.fillna(0,inplace=True) #欠損値を0に置き換える(車が通っていないため、speed=0の扱い)
                return result

            upstream_SPD   = timeAggregateSPD(datalist_up,timeUnit = timeUnit,dataname = "upstream_SPD")
            downstream_SPD = timeAggregateSPD(datalist_dn,timeUnit = timeUnit,dataname = "downstream_SPD")
            upstream_VOL   = timeAggregateVOL(datalist_up,timeUnit = timeUnit,dataname = "upstream_VOL")
            downstream_VOL = timeAggregateVOL(datalist_dn,timeUnit = timeUnit,dataname = "downstream_VOL")
            data_concat = pd.concat([upstream_SPD,downstream_SPD,upstream_VOL,downstream_VOL],axis='columns')
            data_concat.dropna(inplace = True) #元データのうち、500m,2500mの片方だけdataがある場合が存在する。その場合は、短いほうにデータ長を合わせる

            return data_concat

        #学習用のデータ
        datalist = [timeAggregate2locData(datalist_up[i],datalist_dn[i],timeUnit = timeUnit) for i in range(0,len(datalist_up))]

        return datalist

    @abstractmethod
    def getSpeedAll(self):
        raise NotImplementedError

'''
class Sim201907(ReadSimMatrixTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Sim201907'
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath1 = 'simdata500seed/*'
        self.dataPath2 = 'simdata2500seed/*'

    #シナジーのシミュレーションデータから抽出した速度データを1シード値ごと15個分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,0:2] for i in range(0,len(self.datalist))]

        return origin

class Sim201912(ReadSimMatrixTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Sim201912'
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath1 = 'sim201912point1500/*'
        self.dataPath2 = 'sim201912point2000/*'

    #シナジーのシミュレーションデータから抽出した速度データを1シード値ごと15個分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,0:2] for i in range(0,len(self.datalist))]

        return origin

class Sim202001(ReadSimMatrixTemplate):
    def __init__(self):
        #Pickelを保存するフォルダの名前指定
        self.dataname = 'Sim202001'
        #実行しているファイルの場所を取得
        self.prePath  = os.path.dirname(__file__) + '/' 
        #同じ階層内に置いてある読み込みたいデータが入っているフォルダの名前指定
        self.dataPath1 = 'Sim202001_ScenargielogExtract_1500/*'
        self.dataPath2 = 'Sim202001_ScenargielogExtract_2000/*'

    #シナジーのシミュレーションデータから抽出した速度データを1シード値ごと15個分取得
    def getSpeedAll(self):
        origin = [self.datalist[i].iloc[:,0:2] for i in range(0,len(self.datalist))]

        return origin
'''
class ReadMatrixContext(object):
    #戦略を与える
    def __init__(self,strategy):
        self.strategy = strategy

    def ReadSpdMatrixlist(self):
        self.strategy.readMatrix()
        res = self.strategy.getSpeedAll()
        #resはDataFrame型
        return res

if __name__ == '__main__':
    #読み込んだデータを出力する
    # strategy = Glisan2loc()
    # strategy = Norwood2loc()
    # strategy = Sim201907()
    # strategy = Sim201912()
    # strategy = Sim202001()
    # strategy = Glisan20sec1hour2loc()
    strategy = Glisan2loc20sec2hourResolution(window=6)

    context = ReadMatrixContext(strategy)
    original_data = context.ReadSpdMatrixlist()

    for i in range(len(original_data)):
        filename = os.getcwd() + '/data_no{}.csv'.format(str(i))
        original_data[i].to_csv(filename)