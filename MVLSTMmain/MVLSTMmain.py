import backtrace
import copy
import datetime

from ReadData.readDataClassV2 import ReadMatrixContext
from ReadData.readDataClassV2 import Glisan2loc20sec2hourResolution
#from FCProcess.Process_V2MVLSTM import ProcessMVLSTM
#from FCProcess.Process_V3MVLSTM import ProcessMVLSTM
#from FCProcess.Process_V4MVLSTM import ProcessMVLSTM
from FCProcess.Process_V5MVLSTM import ProcessMVLSTM
#from FCProcess.Process_Predict import ProcessPredict

#エラー表示を短くする
backtrace.hook(
    reverse=True,         # 逆順
    strip_path=True    # ファイル名のみ
)

def main():
    #開始時刻取得
    dt_st = datetime.datetime.now()

    #何ステップ先を予測するかを指定。閉ループ（再帰）予測を行う。
    #CHANGED 直接6ステップ先の予測を行う
    pred_range = 1

    #CHANGED Rの半径50m固定
    #Rの半径を連続処理。1から6の整数で選択。R_list[x]*50がRの半径の大きさとなる。
    #R_list =[1,2,3,4,5,6]
    R_list =[1]

    #CHANGED ADV3割固定
    #検証に用いるデータの自動運転車両比率を配列で指定（単位[割]）
    #出力では割合に関わらず左から順にrate1,rate2,rate3...となるので、間違えないように注意
    # val_ADV_list = [1,2,3,4]
    val_ADV_list=[3]

    #Hyper-Parameter設定
    hyper_parameter = { "train_cut"      :0,            #テストデータの指定番目以降を使用してテストさせるか
                        "median"         :1,  #5        #メディアンフィルタ(平滑化)において、メディアンをとる長さ
                        "window_len"     :15, #5       #学習における時系列データの窓長　#CHANGED 15にしとく
                        "layerH_unit"    :30,           #隠れ層のニューロン(ユニット)の数
                        "dropout_rate"   :0.2,          #Dropoutにおいて何割ニューロンを非活性化させるか
                        "epoch"          :150,          #何周学習データを使って学習させるか
                        "batch_size"     :1200,          #重みの更新間隔をバッチ何個分ずつにするか #CHANGED 鉢嶺さんのスライドに合わせた 総数/5にしとく
                        "optimizer"      :"RMSprop"     #最適化関数をどれにするか
    }
    #window_size_list     = [30,60,120]
    #window_size_list     = [10,15,20,25,30]
    #layerH_unit_list     = [5,15,30,60,120]
    #batch_size_list     = [2,4,8,16,32]
    #dropout_rate_list   = [0.05,0.1,0.15,0.2,0.25]

    #あらかじめ既定のハイパーパラメータを追加しておく
    hyperparam_list = [hyper_parameter]
    def MakeParamList(default_param,hyperparam_list,key,valuelist):
        add_hyperparam = copy.copy(default_param)
        for value in valuelist:
            add_hyperparam[key]=value
            if default_param == add_hyperparam: continue
            hyperparam_list.append(copy.copy(add_hyperparam))

    # MakeParamList(hyper_parameter,hyperparam_list,key="window_len",valuelist=window_size_list)
    # MakeParamList(hyper_parameter,hyperparam_list,key="layerH_unit",valuelist=layerH_unit_list)
    # MakeParamList(hyper_parameter,hyperparam_list,key="batch_size",valuelist=batch_size_list)
    # MakeParamList(hyper_parameter,hyperparam_list,key="dropout_rate",valuelist=dropout_rate_list)

    #データ読み込み
    # strategy = Glisan2loc()
    # strategy = Norwood2loc()
    # strategy = Sim201907()
    # strategy = Sim201912()
    # strategy = Sim202001()
    # strategy = Glisan20sec1hour2loc()

    #TODO
    #ここですでに平均化されている
    #元データが5秒間隔のため、現在は15秒位置ステップで実装しようとしていたっぽい
    #差分を実装しようとした関係上、origindataも結果も時系列が元データの1ステップ分ずれている
    #先頭のデータは差分をとるために使うのでorigindataに含まれず、学習および予測には使用しない
    #strategy = Glisan2loc20sec2hourResolution(window=3)
    #strategy = Glisan2loc20sec2hourResolution(window=6)  
    print()
    print("===START Reading Process For Training Data===")

    strategy = Glisan2loc20sec2hourResolution(window=1)

    context = ReadMatrixContext(strategy)
    #original_data : DataFrame
    original_data = context.ReadSpdMatrixlist()#訓練データの元データを読み込み

    print()
    print("===FINISH Reading Process For Training Data===")
    print()
    print("===START Reading Process For Validation Data===")

    original_valdata_list=[]#ADV比率毎に分かれた検証用データをリストで保持
    for rate in val_ADV_list:#比率でループ
        strategy_val = Glisan2loc20sec2hourResolution(window=1,ADVrate=rate)
        context_val = ReadMatrixContext(strategy_val)
        original_valdata = context_val.ReadSpdMatrixlist2(rate)#検証用データの元データを読み込み
        original_valdata_list.append(original_valdata)#配列にまとめる

    print()
    print("===FINISH Reading Process For Validation Data===")

    #予測ステップ数を指定してシミュレーションを行う
    for hi in hyperparam_list:
        #1ステップずつずらして予測
        #CHANGED 6ステップ後,30秒後のデータ予測
        #bookmark stepnum変えてみる？
        for R_num in R_list:
            ProcessMVLSTM(original_data=original_data, original_valdata_list=original_valdata_list,
                          starttime=dt_st, hyper_parameter=hi, pred_step=pred_range, stepnum=6,R_num=R_num)

    dt_ed = datetime.datetime.now()
    print("time={}s".format(dt_ed-dt_st))

if __name__ == "__main__":
    print("test_branch")
    main()


