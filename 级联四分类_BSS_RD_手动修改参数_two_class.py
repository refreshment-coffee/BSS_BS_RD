'''''''''
大致思路：
将四分类 转化为二分类（该类，非该类）
通过如下方式：
switcher = {
     #格式如： "类别":"开始列下标 结束列下标" ##只针对于正样本，采用从开始列下标到结束列下标概率相加 求得正样本概率 （负样本概率=1-正样本概率）
   "N": "01",  # 原来四分类中csv 的第一列
    "C": "12",
    "M": "23",
    "X": "34",
    "≥C": "14",  # 原来四分类中csv 的第二列，第三列，第四列 （C,M,X）
    "≥M": "24"
} ###字典
'''''''''

import numpy as np  ##以下为计算BS和BSS得分（2分类）
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve, auc, roc_auc_score

import pandas as pd
import matplotlib.pyplot as plt


_PRED_CSV = "./two_class_pre_true_csv/pred.csv"
_TRUE_CSV = "./two_class_pre_true_csv/true.csv"
save_path = "./"


_MODEL = "Bi-GRU_A"   #主要用来做title内容和图片的内容 ###可选 _jilian_,_putong_,_RFC_
_CLASS = "N"  #可选  N,C,M,X,≥C,≥M 其余情况直接报错 #手动修改
BEST_TIME_STEP=40



_biaoji =  "_≥M" + "_Prediction"

'''''''''
大致思路:
四分类转为二分类
'''''''''

strat_index = 0
end_index = 0

switcher = {
    #格式如： "类别":"开始列下标 结束列下标" ##只针对于正样本，采用从开始列下标到结束列下标概率相加 求得正样本概率 （负样本概率=1-正样本概率）
    "N": "01",  # 原来四分类中csv 的第一列
    "C": "12",
    "M": "23",
    "X": "34",
    "≥C": "14",  # 原来四分类中csv 的第二列，第三列，第四列 （C,M,X）
    "≥M": "24"
}

'''''''''
    array([[0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11]])
    print(np.sum(b))  # 默认对所有元素进行求和
    ##66
    print(np.sum(b, axis=0))  # 在第一个轴展开方向上求和
    ##array([6, 8, 10, 12, 14, 16])
    print(np.sum(b, axis=1))
    ##array([15, 51])
   
'''''''''


def Preprocess_csv(_true_csv, _pred_csv, _class="N"):
    global strat_index, end_index

    temp_str = switcher.get(_class)  # 返回start_index   end_index 的值
    strat_index = (int)(temp_str[0])
    end_index = (int)(temp_str[1])
    # print("strat: {} ".format(strat_index))
    # print("end : {} ".format(end_index))

    ##仅仅是为了获得两列的数据，
    # print(_true_csv)
    true_csv = _true_csv[:, 0:2]
    pred_csv = _pred_csv[:, 0:2]

    true_csv[:, 1] = np.sum(_true_csv[:, strat_index:end_index], axis=1)  # 列求和   #默认第二列为正样本
    true_csv[:, 1]=np.clip(true_csv[:, 1],0,1)          ##防止上下界越界
    true_csv[:, 0] = 1 - true_csv[:, 1]

    pred_csv[:, 1] = np.sum(_pred_csv[:, strat_index:end_index], axis=1)
    pred_csv[:, 1]=np.clip(pred_csv[:, 1],0,1)          ##防止上下界越界
    pred_csv[:, 0] = 1 - pred_csv[:, 1]

    return true_csv, pred_csv


def draw_rd(true_y_list, pred_y_list, label="No_Define", bins=8):
    '''''''''
    true_y 包含了十个数据集
    pred_y 包含了十个数据集
    '''''''''
    print("\n\n##################___DRAW_RD___#######################\n\n")
    plt.figure(figsize=(15, 15))

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for count in range(0, 10, 1):
        print("\n第 {} 个数据集".format(count + 1))
        true_y = np.array(true_y_list[count])
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pred_y = np.array(pred_y_list[count])

        fraction_of_positives, mean_predicted_value = calibration_curve(true_y, pred_y,
                                                                        n_bins=bins)  ##calibration_curve()适用于2分类器！！！  n_bins: int, default=5. Number of bins to discretize the [0, 1] interval. A bigger number requires more data. Bins with no samples (i.e. without corresponding values in y_prob) will not be returned, thus the returned arrays may have less than n_bins values.
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("fraction_of_positives:",
              fraction_of_positives)  ##注意：在本例子中，n_bins=100或11，Reliability curve曲线一样！直方图hist函数中bins相应地进行调整为100或11。
        print("mean_predicted_value:", mean_predicted_value)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (label +"__"+ str(count + 1)))

        plt.ylabel("Fraction of positives")

        plt.ylim([-0.05, 1.05])
        # plt.legend(loc="lower right")
        plt.legend ( bbox_to_anchor=(0.01, 0.99),loc="upper left",borderaxespad = 0.)

        plt.title('Calibration plots  (Reliability curve)'  +_biaoji)
        plt.xlabel("Mean predicted value")

        # plt.tight_layout()


def calculate_bs_bss(true_y, pred_y, label=" "):
    print("\n\n##################___" + label + "_BS_BSS___#######################")
    bs_score = brier_score_loss(true_y, pred_y, pos_label=1)
    print(label + " brier_score_loss(y_true, y_prob) pos_label=defaule(=1):", bs_score)

    y_mean = true_y.mean()
    print("y_mean : {} ".format(y_mean))
    temp = 0
    for i in range(len(true_y)):
        temp += ((true_y[i] - y_mean) * (true_y[i] - y_mean)) / len(true_y)
    BSS = 1 - brier_score_loss(true_y, pred_y, pos_label=1) / temp  ##计算BSS得分（2分类）
    print("BSS:", BSS)

    return bs_score, BSS


_pred = []
_true = []
_pred_all = np.array([])
_true_all = np.array([])

_avg_bss = []
_avg_bs = []

if __name__ == '__main__':

    for count in range(1, 11, 1):

        _pred_csv_path = _PRED_CSV[:-4] + "_" + str(count) + _PRED_CSV[-4:]
        _true_csv_path = _TRUE_CSV[:-4] + "_" + str(count) + _TRUE_CSV[-4:]

        _pred_csv = np.array(pd.read_csv(_pred_csv_path, header=None))
        _true_csv = np.array(pd.read_csv(_true_csv_path, header=None))

        _true_csv, _pred_csv = Preprocess_csv(_true_csv, _pred_csv, _class=_CLASS)

        _true_csv = _true_csv[:, 1]  # 默认第二列为正样本
        _pred_csv = _pred_csv[:, 1]  # 默认第二列为正样本

        temp_bs, temp_bss = calculate_bs_bss(_true_csv, _pred_csv, label=_MODEL + str(count) + "_")
        _avg_bs.append(temp_bs)
        _avg_bss.append(temp_bss)

        _pred.append(_pred_csv)
        _true.append(_true_csv)
        #
        _pred_all = np.concatenate((_pred_all, _pred_csv))
        _true_all = np.concatenate((_true_all, _true_csv))

    print("\n\n##############################################")   ###计算BSS
    print("_AVG_BS: {}".format(np.mean(_avg_bs)))
    print("_VAR_BS: {} ".format(np.var(_avg_bs)))
    print("_AVG_BSS: {}".format(np.mean(_avg_bss)))
    print("_VAR_BSS: {} ".format(np.var(_avg_bss)))
    print("_VAR_BSS: {} ".format(np.var(_avg_bss)))
    print("_std_BSS: {} ".format(np.array(_avg_bss).std()))

    print(np.array(_avg_bs).mean())
    print(np.array(_avg_bs).std())
    print(np.array(_avg_bss).mean())
    print(np.array(_avg_bss).std())

    draw_rd(_true, _pred, label=_MODEL)                     ###生成RD图
    plt.savefig(save_path + _MODEL + _biaoji + "_two_class_RD_.png")
    # plt.show()
