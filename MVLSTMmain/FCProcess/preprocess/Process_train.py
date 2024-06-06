from evaluate.PredEval import PredEval
import NNModel.NN
import NNkeras.kerasInit as kInit
import numpy as np
from process_cross_validation import cross_validation , cross_val
from Process_validation import final_validation

def train_func(trainset,trainLabset,whole_data,sample_size,valInset_list,valLabset_list,valSampleSize_list,hyper_parameter,stepnum,starttime,dt_now,model_Input_Output,R_range):
    #データ形式をnumpyに変形
    trainIn  = np.array(trainset)
    trainLab = np.array(trainLabset)

    valIn_list=[]
    valLab_list=[]
    for i in range(len(valInset_list)):
        valIn    = np.array(valInset_list[i])
        valLab   = np.array(valLabset_list[i])
        valIn_list.append(valIn)
        valLab_list.append(valLab)

    #trainIn  = np.concatenate(trainset,axis=0)
    #trainLab = np.concatenate(trainLabset,axis=0)


    print()
    print("　　準備データ数 : ",sample_size)
    print("必要学習データ数 : ",whole_data)
    print()

    if sample_size < whole_data:
        need_data_size = round((whole_data/sample_size),2)
        print("学習データ不足です。約" , need_data_size , "倍以上のデータを用意してください。")
    elif sample_size > whole_data:
        print("学習データ過多です。学習日数を調節してください。")
    else:
        #指定した学習データ数になるように、必要以上の学習データをカット。
        trainIn  = trainIn[:whole_data]
        trainLab = trainLab[:whole_data]

        #交差検証を行う(k-分割法：検証データ欠損なし)
        cross_val(trainIn,trainLab,valIn_list,valLab_list,hyper_parameter,starttime,dt_now,stepnum,model_Input_Output,R_range)

        #kerasGPU設定初期化(ここでGPUメモリ占有の設定を行う)
        kInit.kerasInit()

        #訓練データによる予測(fitting)
        #訓練データを入れて予測した場合、モデルが学習できているかをテストすることになる
        #サンプル数不明のためバグ回避で1を指定
        train_eval = PredEval(sample_num=1,
                                hyper_param=hyper_parameter)

        #結果を出力するフォルダを設定
        learntype = str(model_Input_Output)+'D_'+'TrainData'+str(whole_data)+'_R'+str(R_range)  #モデルに使用したデータ数
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
                                                    tensorboard_path=TensorBoard_path,
                                                    model_Input_Output = model_Input_Output)

        #学習のlossをグラフで出力
        train_eval.ExportLossGraph(history=history)



        whole_data = len(trainIn) #訓練データの数
        Final_Val = final_validation(hyper_parameter,starttime,dt_now,stepnum,model_Input_Output,whole_data,model,R_range)
        Final_Val.set_FolderPath()#出力先フォルダをセット
        #学習モデルの予測精度を検証データで検証
        RMSE_list=[]
        MAE_list=[]
        SDAE_list=[]
        RMSE2_list=[]
        MAE2_list=[]
        SDAE2_list=[]
        for index in range(len(valIn_list)):#用意した割合の種類分でループ
            rate = str(index + 1)
            #学習モデルの本検証
            RMSE,MAE,SDAE, RMSE2,MAE2,SDAE2 = Final_Val.Final_velify(valIn_list[index],valLab_list[index],"rate"+rate+"_val",model_Input_Output)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
            SDAE_list.append(SDAE)
            RMSE2_list.append(RMSE2)
            MAE2_list.append(MAE2)
            SDAE2_list.append(SDAE2)
        #評価指標をまとめる
        valid_score = [RMSE_list,MAE_list,SDAE_list]
        Final_Val.make_valid_scores_csv(valid_score,"1") #検証の結果をまとめたcsvを書き出す
        valid_score2 = [RMSE2_list,MAE2_list,SDAE2_list]
        Final_Val.make_valid_scores_csv(valid_score2,"2") #検証の結果をまとめたcsvを書き出す