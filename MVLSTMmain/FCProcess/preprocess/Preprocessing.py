import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.cleandata import smoothing 

def DataPreprocessing(ts_data,MFwindow):#欠損データに未対応
    #訓練データ平滑化 LI:線形補間,MF:メディアンフィルタ
    #TODO 欠損値の影響を改善する必要がありそう
    #欠損値のせいで時系列とデータがずれている。補完方法を検討
    ts_data = smoothing.LinearInterpolation(ts_data)                #線形補完 ただ隙間を詰めただけ
    ts_data = smoothing.MovingAverage(ts_data,window_size=5)        #移動平均
    ts_data = [np.array(ts_data[i]) for i in range(len(ts_data))]

    return ts_data

def DataPreprocessing2(ts_data,MFwindow):#線形補完も平滑化もしない。numpy型に変換するだけ。
    #訓練データ平滑化 LI:線形補間,MF:メディアンフィルタ
    print("")
    print("Now processing data into numpy type...")
    ts_data = [np.array(ts_data[i]) for i in tqdm(range(len(ts_data)))]

    return ts_data

#[x]
def DataPreprocessing3(ts_data,MFwindow):
    #訓練データ平滑化 LI:線形補間,MF:メディアンフィルタ
    print()
    print("===START Smoothing process===")

    #欠損補完して平滑化した後に再度欠損させる
    ts_data_origine = [np.array(ts_data[i]) for i in range(len(ts_data))]           #欠損補完前・平滑化前のnumpy型ts_data
    print("Now during linear interpolation...")
    ts_data         = smoothing.LinearInterpolation(ts_data)                        #線形補完(全欠損未対応/全欠損のまま返される) 中間のNANは前後の数字を線形に結んで補完。始端のNANは、始めに来た数字で補完。終端のNANは、最後に表れた数字で補完。
    #=====全欠損の補完のみここで施す=======
    for i in range(len(ts_data)):
        #DataFrame型がlist型で束ねられているので、ts_dataはlist型だが、ts_data[i]はDataFrame型
        ts_data[i] = ts_data[i].fillna(0) #すべてのNANを0に置換
    #=====ここまで全欠損補完=============
    
    print()
    print("Now smoothing...")
    ts_data         = smoothing.MovingAverage(ts_data,window_size=MFwindow)         #移動平均 始端はwindow_sizeになるまで1個平均、2個平均...window_size個平均、となる。引数の配列にNANがあるとおかしくなるから注意。
    ts_data_maked   = [np.array(ts_data[i]) for i in range(len(ts_data))]           #欠損補完・平滑化されたデータの型をnumpy配列にリメイク

    #元々の欠損部をもう一度欠損させる(NANを代入)
    for i in range(len(ts_data_origine)): #csvの個数分ループ
        for k in range(len(ts_data_origine[i])): #入力データの個数(行数)分ループ
            if len(ts_data_maked[i]) != 0:
                if all(~np.isnan(ts_data_origine[i][k])): #その行にNanが含まれているとFalseが返される
                    pass
                else:
                    for j in range(ts_data_origine[i].ndim): #全ての列を一つずつチェック　Nanならts_data_makedの同箇所もNanに変更（再欠損）
                        if np.isnan(ts_data_origine[i][k][j]): #要素がNanならTrueが返される
                            ts_data_maked[i][k][j] = np.nan #Nanを代入。int型のnumpyには適用不可。

    print("===FINISH Smoothing process===")
    return ts_data_maked


def Normarizing(ts_data):
    #TODO 20191103 最大値を保持する必要はないため要改善
    #平滑化データを正規化する
    #データ中の最大値を調べ、その値で割る(0～1にする)
    #TODO 一時的に空配列に対応
    #前方車両のデータをまとめて利用する場合はこのデータが空配列に対応する必要あり
    """
    ts_data_max_list = []
    for i in range(len(ts_data)):
        if ts_data[i]:
            ts_data_max_list.append(ts_data[i].max(axis=0))
            continue
        ts_data_max_list.append(1)
    """

    ts_data_max_list = [ts_data[i].max(axis=0) for i in range(len(ts_data))]
    array_ts_data_maxlist = np.array(ts_data_max_list)
    ts_data_max = np.max(array_ts_data_maxlist)
    normarized_ts_data = [ts_data[i]/ts_data_max for i in range(len(ts_data))]

    return normarized_ts_data,ts_data_max

#[x]
def Normarizing2(ts_data):
    """
    正規化する。正規化データと正規化に利用した最大値をreturnする。
    """
    #np.maxではnanを含む配列で最大値がnanで返されてしまうため、nanを含む配列で処理場合はnp.nanmaxを使う。行列指定は引数axis=0,axis=1で指定する。
    #ts_data_max_list = [np.nanmax(ts_data[i],axis=0) for i in range(len(ts_data)) if len(ts_data[i]) != 0]
    #array_ts_data_maxlist = np.array(ts_data_max_list)#arrayに変形

    #最大値リスト内の最大値を求める。自車速度と前方範囲平均の両方の中での最大値を求める。nanを含むリストなのでnanmaxを用いる。
    #ts_data_max = np.nanmax(array_ts_data_maxlist)
    #正規化に用いる最大値は交通流simで取り得る最高速度の値
    ts_data_max = 120/3.66 #CHANGED 現在最高速度は第2走行車線が110km/km

    #最大値で生データを割って、正規化する。最大値が1となる。最小値は元々0。
    normarized_ts_data = [ts_data[i]/ts_data_max for i in range(len(ts_data))]

    return normarized_ts_data,ts_data_max
