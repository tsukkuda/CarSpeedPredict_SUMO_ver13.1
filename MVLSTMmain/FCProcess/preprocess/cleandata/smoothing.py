from abc import ABCMeta,abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm

#平滑化クラスの集まり(Template Method Pattern)
class SmoothingTemplate(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def process(data,k):
        raise NotImplementedError

    #型ごとに処理分岐
    def adapter(self,matrix,k):
        def DataFrame(matrix,k):
            #matrix = matrix.dropna() #Nanを含む行を除外。NANを補完したいのになぜ除外する？？20220906

            res = pd.DataFrame()
            for i in range(len(matrix.columns)):
                colname = matrix.columns[i]
                res[colname] = self.process(matrix.iloc[:,i],k)

            return res

        #まずここで引数のデータ型を判別。それぞれに合った補完プロセスを行う。
        #Seriesに対して平滑化するとき
        if type(matrix) is pd.Series:
            result = self.process(matrix,k)
            return result   
        #DataFrameの各列に対して平滑化するとき
        elif type(matrix) is pd.DataFrame:
            result = DataFrame(matrix,k)
            return result
        #各リストに入っているDataFrameの各列に対して平滑化するとき
        elif type(matrix) is list: 
            result = [self.adapter(matrix[i],k) for i in tqdm(range(len(matrix)))]
            return result 
        else:
            raise TypeError('input data type is not "pd.Series or pd.DataFrame\
                             or list".Please change value type.')

class MF(SmoothingTemplate):
    @staticmethod
    def process(data,k,*args):
        #フィルタをかけた結果をcleandataに代入。
        cleandata = []

        #メディアンフィルタの係数によって場合分けしている。
        #係数が奇数か偶数の場合で処理が変わる。具体的には、メディアンを取るときの
        #配列の中心の位置(middle)の出し方を変えている。
        if k % 2 == 0 :
            middle = int(k / 2)
        elif k % 2 == 1:
            middle = int((k - 1) / 2)
        else:
            raise ValueError("error:window size is not integer")

        #実際の処理部分
        #len()で長さを取り、その回数だけ処理を行う
        for i in range(len(data)):
            if  middle > i or len(data) - middle <= i:
                #メディアンを取る配列の要素数が係数kより少ない場合、元データをそのまま代入
                cleandata.append(data[i])
            else:
                #それ以外の場合はメディアンを取り、処理結果を代入
                #係数kだけ要素をtempに
                temp = [data[j+i-middle] for j in range(k)]
                #欠損値処理のためlist型からpandasにあるSeries型に変換
                temp = pd.Series(temp)
                #欠損値除去
                temp = temp.dropna()
                #Series型からndarray型に変換
                temp = np.array(temp)
                datamedian = np.median(temp)
                if np.isnan(datamedian) == True:
                    datamedian = cleandata[-1]
                cleandata.append(datamedian)
        
        cleandata = pd.Series(cleandata)
        return cleandata

class LI(SmoothingTemplate):
    @staticmethod
    def process(data,*args):
        #data = data.interpolate(axis=0)
        data = data.interpolate(axis=0,limit_direction='both')
        return data

class MA(SmoothingTemplate):
    @staticmethod
    def process(data,k,*args):
        res = data.rolling(k,min_periods=1).mean()
        return res

def MedianFilter(matrix,window_size=3):
    """
    Apply a Median Filter

    Parameters
    ----------
    matrix : pandas.Series or pandas.DataFrame or List with pandas.DataFrame in elements
        Object to be median filtered
    window_size : int
        default : 3
        Window size for calculating median

    Return
    ------
    out : pandas.Series or pandas.DataFrame or List with pandas.DataFrame in elements
        Median filtered input data
    """
    matrix = [matrix[i].dropna() for i in range(len(matrix))]
    matrix = [matrix[i].reset_index(drop=True) for i in range(len(matrix))]

    ins = MF()
    res = ins.adapter(matrix,window_size)

    return res

def LinearInterpolation(matrix,window_size=0):
    """
    Apply a Linear Interpolation

    Parameters
    ----------
    matrix : pandas.Series or pandas.DataFrame or List with pandas.DataFrame in elements
        Object to be linearly interpolated

    Return
    ------
    out : pandas.Series or pandas.DataFrame or List with pandas.DataFrame in elements
        Linearly interpolated input data
    """
    ins = LI()
    res = ins.adapter(matrix,window_size)

    return res

def MovingAverage(matrix,window_size=3):
    """
    Apply a Moving Avarage

    Parameters
    ----------
    matrix : pandas.Series or pandas.DataFrame or List with pandas.DataFrame in elements
        Object to be moving avaraged
    window_size : int
        default : 3
        Window size for calculating moving avarage

    Return
    ------
    out : pandas.Series or pandas.DataFrame or List with pandas.DataFrame in elements
        Moving avaraged input data
    """
    matrix = [matrix[i].dropna() for i in range(len(matrix))]#NANの除外/どれかの列にNANが含まれていると、その行自体が削除される。欠損補完されていないと、欠損を含む時間の行だけ間引かれて時系列がズレるので注意。
    matrix = [matrix[i].reset_index(drop=True) for i in range(len(matrix))]
    ins = MA()
    res = ins.adapter(matrix,window_size)

    return res
    
if __name__ == '__main__':
    a = np.array([[1,2,np.nan,4,5,8,7,np.nan,9,10],[10,20,np.nan,40,50,60,70,np.nan,90,100]])
    pda = pd.DataFrame(a.T,
                    columns = ['one','two']
                    )
    a2 = np.array([[8,4,7,4,5,8,7,np.nan,9,10],[10,40,70,40,50,60,20,np.nan,90,100]])
    pda2 = pd.DataFrame(a2.T,
                    columns = ['one','two']
                    )
    listpd = [pda,pda2]

    LIpda=LinearInterpolation(pda)
    MFpda=MedianFilter(pda,3)
    MApda=MovingAverage(pda,3)

    print(LIpda)
    print(MFpda)
    print(MApda)

    LIlistpda=LinearInterpolation(listpd)
    MFlistpda=MedianFilter(listpd,3)
    MAlistpda=MovingAverage(listpd,3)

    print(LIlistpda)
    print(MFlistpda)
    print(MAlistpda)