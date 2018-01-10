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

# 待分类的数据集，簇的个数，距离算法，初始化质心方式
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
            # 找出距离i点距离最小的质心点，将它划分到clusterAssment中
            for j in range(k):
                distJI = distMeans(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex, minDist*2
        # print(centroids)
        for cent in range(k):
            # .A将矩阵转化为array类型；取clusterAssment中对应质心点的的x轴的值
            pstInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            # 取x轴均值
            centroids[cent,:] = mean(pstInClust, axis=0)
    # 返回质心位置和各个点所属质心及误差
    return centroids,clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssement = mat(zeros((m,2)))
    # tolist数组转列表
    centroid0=mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    # 初始化各个点到第一个簇的距离
    for j in range(m):
        clusterAssement[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while(len(centList)<k):
        lowestSSE = inf
        # 对每一个簇进行遍历，得到最小的ssr
        for i in range(len(centList)):
            # 第i个簇对应的数据点
            pstInCurrCluster = dataSet[nonzero(clusterAssement[:,0].A==i)[0],:]
            # 对第i个簇二分
            centroidMat,splitClustAss = kMeans(pstInCurrCluster,2,distMeas)
            # 二分后的该簇ssr
            sseSplit = sum(splitClustAss[:,1])
            # 未二分的簇ssr
            sseNotSplit = sum(clusterAssement[nonzero(clusterAssement[:,0].A != i)[0],1])
            # print('sseSplit,and notSplit: ',sseSplit,sseNotSplit)
            # 比较切分第i个簇的sse
            if(sseSplit+sseNotSplit)<lowestSSE:
                bestCentToSplit = i
                # 新的质心（两个）
                bestNewCents = centroidMat
                # 新质心（两个0，1）对应的误差
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将新误差分至相应
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        # 赋值最佳划分点及添加新点
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        # 添加新点的误差
        clusterAssement[nonzero(clusterAssement[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return centList, clusterAssement

dataMat = mat(loadDataSet("testSet2.txt"))
cl,ca = biKmeans(dataMat,4)
# print(cl)
# print(ca)

