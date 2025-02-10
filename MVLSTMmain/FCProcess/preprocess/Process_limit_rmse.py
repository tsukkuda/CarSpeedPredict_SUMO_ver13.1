import os
import matplotlib.pyplot as plt
import numpy as np

#*予測値と真値の差が40km/h以上の場合の入力値をグラフで出力する関数
def IOChecker(valIn,pred,valLab1,folder_path): #入力値，予測値，真値, 出力先パス
    print("IOChecker start")
    
    result_path1 = folder_path + "IOChecker/" #フォルダ作成
    if not os.path.exists(result_path1):
        os.makedirs(result_path1)
        
    result_path2 =result_path1+"graphs/"
    if not os.path.exists(result_path2):
        os.makedirs(result_path2)
    
    
    #誤差が大きいやつ出力
    if len(pred[0])==1: #1次元の場合
        for i,val in enumerate(pred):
            if i%1000 == 0:
                if abs(val[0]*120 - valLab1[i][0]*120)>=40: #! 予測値と真値の差が40km/h以上の場合
                    fig,ax=plt.subplots(figsize=(10,10)) #画像サイズ
                    fig.set_figheight(5) #高さ調整
                    ax.tick_params(labelbottom=True, bottom=False) #x軸設定
                    ax.tick_params(labelleft=True, left=False) #y軸設定
                    ax.set_xlim(0,88)
                    ax.set_ylim(0, 120)
                    plt.tight_layout() #グラフの自動調整
                    ax.set_axisbelow(True)
                    plt.minorticks_on()
                    plt.grid(which = "both", axis="both")
                    plt.hlines(y=0,xmin=0,xmax=88, color="grey") #横軸
                    plt.vlines(x=0,ymin=0,ymax=120, color="grey") #縦軸
                    plt.xlabel("time[s]", fontname="MS Gothic", labelpad=0, fontsize="large") #x軸ラベル
                    plt.ylabel("speed[km/h]", fontname="MS Gothic", labelpad=0, fontsize="large") #y軸ラベル

                    for j in range(len(valIn[i])):
                        if j==0:
                            plt.scatter(0,valIn[i][j]*120, c='blue', s=100,marker='.', label='入力値')
                        else:
                            plt.scatter(j*5,valIn[i][j]*120, c='blue', s=100,marker='.')
                    plt.scatter((j+1)*5, val[0]*120, c='red', s=100,marker='.', label='予測値')
                    plt.scatter((j+1)*5, valLab1[i][0]*120, c='orange', s=100,marker='.', label='真値')

                    plt.legend(prop={"family":"Meiryo", "weight":"bold", "size":"xx-large"}) #凡例

                    plt.savefig(result_path2 + str(i) + ".png")#グラフを保存
                    plt.close()  # 現在の図を閉じる

    elif len(pred[0])==2: #2次元の場合
        for i,val in enumerate(pred):
            if i%1000 == 0:
                if abs(val[0]*120 - valLab1[i][0]*120)>=40: #!予測値と真値の差が40km/h以上の場合
                    fig,ax=plt.subplots(figsize=(10,10)) #画像サイズ
                    fig.set_figheight(5) #高さ調整
                    ax.tick_params(labelbottom=True, bottom=False) #x軸設定
                    ax.tick_params(labelleft=True, left=False) #y軸設定
                    ax.set_xlim(0,88)
                    ax.set_ylim(0, 120)
                    plt.tight_layout() #グラフの自動調整
                    ax.set_axisbelow(True)
                    plt.minorticks_on()
                    plt.grid(which = "both", axis="both")
                    plt.hlines(y=0,xmin=0,xmax=88, color="grey") #横軸
                    plt.vlines(x=0,ymin=0,ymax=120, color="grey") #縦軸
                    plt.xlabel("time[s]", fontname="MS Gothic", labelpad=0, fontsize="large") #x軸ラベル
                    plt.ylabel("speed[km/h]", fontname="MS Gothic", labelpad=0, fontsize="large") #y軸ラベル

                    for j in range(len(valIn[i])):
                        if j==0:
                            plt.scatter(0,valIn[i][j][0]*120, c='blue', s=100,marker='.', label='入力[自車]')
                            plt.scatter(0,valIn[i][j][1]*120, c='green', s=100,marker='.', label='入力[前方]')
                        else:
                            plt.scatter(j*5,valIn[i][j][0]*120, c='blue', s=100,marker='.')
                            plt.scatter(j*5,valIn[i][j][1]*120, c='green', s=100,marker='.')
                    plt.scatter((j+1)*5, val[0]*120, c='red', s=100,marker='.', label='予測値')
                    plt.scatter((j+1)*5, valLab1[i][0]*120, c='orange', s=100,marker='.', label='真値')

                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={"family":"Meiryo", "weight":"bold", "size":"xx-large"}) #凡例
                    plt.tight_layout()  # レイアウト調整

                    plt.savefig(result_path2 + str(i) + ".png")#グラフを保存
                    plt.close()  # 現在の図を閉じる


    #でてきたグラフを表にして出力したい
    
    
    print("IOChecker done")

#* 条件(予測値<40km/hかつ真値<40km/h)を満たす値の場合にRMSEを計算する関数
def LimitLoss(pred,valLab1): #予測値、真値
    limVal=40.0/120 #この値以下のRMSEは計算しない
    loss=[] #結果いれるやつ
    
    for i,val in enumerate(pred):
        if val[0]>limVal and valLab1[i][0]>limVal:
            loss.append(val-valLab1[i])
        
    return np.array(loss)


# #test用
# if __name__ == "__main__":
#     IOChecker([[[0.5,0.3,0.7]],[[0.2,0.5,0.1]]],[[0.2,0.5]],[[0.6]],os.getcwd()+"/MVLSTMmain/")
