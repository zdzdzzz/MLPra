#-*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
    fr = open(fileName)
    # readline：返回txt第一行结果，结果是str
    # 1.000000	0.067732	3.176513
    numFeat = len(fr.readline().split('\t'))-1
    dataMat = []
    labelMat = []
    # readlines：返回txt所有内容，list
    # ['1.000000\t0.067732\t3.176513\n', '1.000000\t0.427810\t3.816464\n',。。。
    for line in fr.readlines():
        lineArr = []
        # 将字符串按\t分隔
        curLine = line.strip().split('\t')
        # i从0到numFeat-1
        # range为range类型
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        # append extend均是list的函数
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standTegres(xArr, yArr):
    # numpy中将list转为matrix
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print('This matrix is singlar, cannot do inverse')
        return
    # 矩阵.T为转置矩阵，.I为逆矩阵
    ws = xTx.I*(xMat.T*yMat)
    return ws
# 数据为什么第一列是1？？
xArr, yArr = loadDataSet("ex0.txt")
ws = standTegres(xArr, yArr)

xMat = mat(xArr)
yMat = mat(yArr)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
# flatten为平整array数据，降维
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])

xCopy = xMat.copy()
# 从小到大排序
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
#plt.show()

y = xMat*ws
print(corrcoef(y.T,yMat))

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singlar, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights*yMat))
    return testPoint*ws