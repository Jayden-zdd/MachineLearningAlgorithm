# MachineLearningAlgorithm
PCA:
１．中心化
２.计算中心化后的数据的协方差矩阵（np.dot(xT,x)）
３．对协方差矩阵进行特征分解
４．取最大的特征值所对应的特征向量ｄ（样本点xi在新空间超平面上的投影是dTxi,投影后样本点的方差"求和dTxiTxid"最大化，即max tr(dTxTxd)）（最小的特征值对应的特征向量往往与噪声相关<<机器学习>>）
