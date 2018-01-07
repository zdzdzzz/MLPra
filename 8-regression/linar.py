#-*- coding: utf-8 -*-
__author__ = 'acer1'

from numpy import *
from math import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 计算testPoint点的ws值，并返回结果；综合考虑其他点对testPoint的影响
def lwlr(testPoint,xArr,yArr,k):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    # 确定数组维度，不然没有yHat[i]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

xArr, yArr = loadDataSet('ex0.txt')
yHat = lwlrTest(xArr, xArr, yArr, 0.01)
xMat = mat(xArr)
# 由小到大，返回数值的下标
srtInd = xMat[:,1].argsort(0)
# xMat为200*2，xMat[srtInd]为200*1*2，增加了一个维度，[:,0,:]选中某一维度
xSort = xMat[srtInd][:,0,:]

import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s = 2,c='red')
# plt.show()

def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()




