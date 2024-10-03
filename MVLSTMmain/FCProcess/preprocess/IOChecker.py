import os
import matplotlib.pyplot as plt
import numpy as np

def IOChecker(valIn,pred,valLab1,folder_path): #入力値，予測値，真値
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
            if abs(val[0] - valLab1[i][0])>=40: #予測値と真値の差が40km/h以上の場合
                fig,ax=plt.subplots(figsize=(10,10)) #画像サイズ
                fig.set_figheight(1) #高さ調整
                ax.tick_params(labelbottom=True, bottom=False) #x軸設定
                ax.tick_params(labelleft=False, left=False) #y軸設定
                ax.set_xlim(0, 120)
                ax.set_ylim(-0.1,0.1)
                plt.tight_layout() #グラフの自動調整
                ax.set_axisbelow(True)
                plt.minorticks_on()
                plt.grid(which = "both", axis="x")
                plt.hlines(y=0,xmin=0,xmax=120, color="grey") #横軸
                
                plt.scatter(val[0],0, c='red', s=100,marker='v', label='予測値')
                plt.scatter(valIn[i],[0]*15, c='blue', s=100,marker='.', label='入力値')
                plt.scatter(valLab1[i][0],0, c='orange', s=100,marker='v', label='真値')
                
                plt.savefig(result_path2 + str(i) + ".png")#グラフを保存


    elif len(pred[0])==2: #2次元の場合
        for i,val in enumerate(pred):
            if abs(val[0] - valLab1[i][0])>=40: #予測値と真値の差が40km/h以上の場合
                #以下に処理記述
                print()

    else:
        print("Error in IOChecker!")
        
    #でてきたグラフを表にして出力したい
    
    
    print("IOChecker done")
    
    
#test用
if __name__ == "__main__":
    IOChecker([[25,30,50,89,90,51,46,29,59,44,13,66,77,99,10]],[[20]],[[60]],os.getcwd()+"/MVLSTMmain/")

