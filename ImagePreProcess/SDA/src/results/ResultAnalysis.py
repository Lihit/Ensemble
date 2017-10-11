import json
import os
from collections import defaultdict
import numpy as np
import csv


def readCSVfile(CsvFile):
    ret = {}
    if not os.path.exists(CsvFile):
        print('CsvFile can not be found...')
        return
    fp = open(CsvFile, 'r')
    r = csv.reader(fp)
    for line in r:
        ret[int(line[0])] = line[-1]
    return ret


def main(jsonPath):
    if not os.path.exists(jsonPath):
        print('jsonPath can not be found...')
        return
    with open(jsonPath, 'r') as fp:
        LoadResult = json.load(fp)
        # print(LoadResult)
    ret = defaultdict(lambda: [0] * 6)
    errorRet = defaultdict(list)
    for oneDict in LoadResult:
        if oneDict != None:
            trueLabel = int(oneDict['image_id'].split('_')[0])
            try:
                trueIndex = oneDict['label_id'].index(trueLabel)
                ret[trueLabel][trueIndex] += 1
            except Exception as e:
                ret[trueLabel][5] += 1
                errorRet[trueLabel].append(oneDict['label_id'][:3])
    return ret,errorRet


if __name__ == '__main__':
    jsonPath = 'result.json'
    ret,errorRet = main(jsonPath)
    tmp = {}
    labelDict = readCSVfile('scene_classes.csv')
    for key in ret.keys():
        tmp[key] = list(np.array(ret[key]) / (1.0 * sum(ret[key])))
        # if tmp[key][0] <= 0.5:
        #     print(labelDict[key])
        #     print(tmp[key])
    for key in errorRet.keys():
        print(key)
        print(errorRet[key])

