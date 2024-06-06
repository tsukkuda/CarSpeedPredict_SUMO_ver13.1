import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#graphdraw:4つまでグラフを引くことができる関数
#x軸はすべて固定である必要がある
#グラフの色は、それぞれ異なる色になる(ex:1つめは赤等)
#引数
#   x:x軸の値が入った時系列ベクトルデータ
#   y1,y2...:y軸の値が入った時系列ベクトルデータ
#       warning!:xとyのデータの長さは同じにする必要あり
#   label1,label2...:グラフにつける名前(凡例に出てくる)
#TODO 一時的に5台分入力しても対応できるように設定した
#いずれは配列等を渡すようにして10台くらいまで対応させたい
def graphdraw(x1,y1,label1,x2=None,y2=None,label2=None,x3=None,y3=None,label3=None,x4=None,y4=None,label4=None,x5=None,y5=None,label5=None,x6=None,y6=None,label6=None,x7=None,y7=None,label7=None):
    if x2 is None:
        x2=x1
    if x3 is None:
        x3=x1
    if x4 is None:
        x4=x1
    if x5 is None:
        x5=x1
    if x6 is None:
        x6=x1
    if x7 is None:
        x7=x1

    plt.figure()
    plt.plot(x1,y1,color="r",label=label1)
    if y2 is not None:
        plt.plot(x2,y2,color="b",lw=1.0,label=label2)
    if y3 is not None:
        plt.plot(x3,y3,color="g",label=label3)
    if y4 is not None:
        plt.plot(x4,y4,color="k",label=label4)
    if y5 is not None:
        plt.plot(x5,y5,color="m",label=label5)
    if y6 is not None:
        plt.plot(x6,y6,color="y",label=label6)
    if y7 is not None:
        plt.plot(x7,y7,color="c",lw=1.0,label=label7)

    plt.minorticks_on()
    plt.grid(which = "both", axis="x")
    plt.grid(which = "both", axis="y")
    plt.legend()
    plt.show()

def savegraphdraw(x1,y1,label1,x2=None,y2=None,label2=None,x3=None,y3=None,label3=None,x4=None,y4=None,label4=None,x5=None,y5=None,label5=None,x6=None,y6=None,label6=None,x7=None,y7=None,label7=None,xlim=None,ylim=None,savepath="",figurename="hogehoge"):
    if x2 is None:
        x2=x1
    if x3 is None:
        x3=x1
    if x4 is None:
        x4=x1
    if x5 is None:
        x5=x1
    if x6 is None:
        x6=x1
    if x7 is None:
        x7=x1

    plt.figure()
    plt.plot(x1,y1,color="r",label=label1)
    if y2 is not None:
        plt.plot(x2,y2,color="b",lw=1.0,label=label2)
    if y3 is not None:
        plt.plot(x3,y3,color="g",label=label3)
    if y4 is not None:
        plt.plot(x4,y4,color="k",label=label4)
    if y5 is not None:
        plt.plot(x5,y5,color="m",label=label5)
    if y6 is not None:
        plt.plot(x6,y6,color="y",label=label6)
    if y7 is not None:
        plt.plot(x7,y7,color="c",lw=1.0,label=label7)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.minorticks_on()
    plt.grid(which = "both", axis="x")
    plt.grid(which = "both", axis="y")
    plt.legend()
    fname = savepath+figurename
    #plt.savefig(fname,tight_layout=True) #tight_layout=Trueは古い記述方法。bbox_inches='tight'が新しい記述方法。
    plt.savefig(fname, bbox_inches='tight')
    plt.close()