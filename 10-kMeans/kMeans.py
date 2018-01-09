#-*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map对迭代器内的每一项进行运算
        # fltLine = map(float, curLine)
        for i in range(len(curLine)):
            curLine[i] = float(curLine[i])
        dataMat.append(curLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB,2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # 随机产生质心
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids

def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex, minDist*2
        print(centroids)
        for cent in range(k):
            # .A将矩阵转化为array类型
            pstInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(pstInClust, axis=0)
    return centroids,clusterAssment

dataMat = mat(loadDataSet("testSet.txt"))
km = kMeans(dataMat,4)
print(km)




