#-*- coding: utf-8 -*-

from math import *

# 计算集合的香农值
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1, 0,'no'],
               [0, 1,'no'],
               [0, 0, 'no'],]
    labels = ['no surface','flippers']
    return dataSet, labels

# 获得某一维度值为value的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 获取数据集熵最大的维度
def chooseBestFeatureTosplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #[]1,1,1,0,0];[1,1,0,1,0]
        featList = [example[i] for example in dataSet]
        # 第一列；list转为set，无重复
        # {0，1},{0,1}
        uniquevals = set(featList)
        newEntropy = 0.0
        #将数据集按0，1划分
        for value in uniquevals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

import operator
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter[1], reverse = True)
    return sortedClassCount[0][0]

# labels:dataSet中每一维度对应的标签
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择熵最大的维度
    bestFeat = chooseBestFeatureTosplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #划分好维度后，将标签值删除
    del(labels[bestFeat])
    #按选取的维度（即熵最大的维度）划分子树
    featValues = [example[bestFeat] for example in dataSet]
    uniquevals = set(featValues)
    for value in uniquevals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value),subLabels)
    return myTree

myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print(myTree)

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.key()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key])._name_=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel



