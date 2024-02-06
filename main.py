# Code LSH Duplicate Detection

# Packages to import
import random
import json
import numpy as np
import re
from sklearn.cluster import AgglomerativeClustering


# Prepare the data
def dataPrepare():
    data = json.load(open("Data\TVs-all-merged.json"))

    modelIDs = list(data.keys())
    productList = []
    duplicateAmount = []

    for modID in modelIDs:
        modelProducts = data.get(modID)
        productList.extend(modelProducts)
        duplicateAmount.append(len(modelProducts))

    productAmount = len(productList)
    duplicateMatrix = np.zeros((productAmount, productAmount))
    i = 0

    for dupAmount in duplicateAmount:
        duplicateMatrix[i:i + dupAmount, i:i + dupAmount] = 1
        i += dupAmount

    for i in range(len(productList)):
        duplicateMatrix[i, i] = 0

    for product in productList:
        # Delete unused information
        del product['url']
        del product['modelID']

        # Cleaning of data
        product['featuresMap'] = {key.lower(): value.lower() for key, value in product['featuresMap'].items()}
        replaceInch = ['\"', 'inches', '-inch']
        for replacement in replaceInch:
            product['featuresMap'] = {key: value.replace(replacement, 'inch') for key, value in product['featuresMap'].items()}
        replaceBrand = ['brand name']
        for replacement in replaceBrand:
            product['featuresMap'] = {key: value.replace(replacement, 'brand') for key, value in product['featuresMap'].items()}

    return productList, duplicateMatrix


def createModelWords(productList, includeBrand = True):
    def encoder(productTitle, mwVocabulary):
        patternMatches = re.findall('[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*', productTitle.lower())
        return {encoderVocabulary(mwVocabulary, match) for match in patternMatches}

    def encoderVocabulary(mwVocabulary, match):
        result = mwVocabulary.get(match)
        if result is None:
            result = len(mwVocabulary)
            mwVocabulary[match] = len(mwVocabulary)
        return result

    mwVocabulary = {}
    mwList = []

    for product in productList:
        if includeBrand and product['featuresMap'].get('brand') is not None:
            productTitle = "".join([product['title'], product['featuresMap'].get('brand')])
        else:
            productTitle = product['title']
        mwProduct = encoder(productTitle, mwVocabulary)
        mwList.append(mwProduct)

    return mwVocabulary, mwList


def createProductShingles(product, shingleLength):
    shingleSet = set()

    keyValues = set(product['featuresMap'].items())
    keyValues.add(('title', product['title'].lower()))

    for key, value in keyValues:
        keyString = ''.join(map(str,key))
        if shingleLength <= len(keyString):
            for i in range(len(keyString) - shingleLength + 1):
                currentShingle = keyString[i:i+shingleLength]
                shingleSet.add(currentShingle)

        valueString = ''.join(map(str, value))
        if shingleLength <= len(valueString):
            for i in range(len(valueString) - shingleLength + 1):
                currentShingle = valueString[i:i + shingleLength]
                shingleSet.add(currentShingle)

    return shingleSet


def main():
    random.seed(2024)
    productList, duplicateMatrix = dataPrepare()
    resultsPQLSH = []
    resultsPCLSH = []
    resultsF1LSH = []
    resultsF1StarLSH = []
    resultsPQClust = []
    resultsPCClust = []
    resultsF1Clust = []
    resultsF1StarClust = []
    thresholds = []

    #Parameters
    n = 1000
    N = len(productList)
    bootstrapAmount = 20
    shingleLength = 3

    #Create a list for the rows (from rows and bands)
    rList = []
    for r in range(2,n):
        if n % r == 0:
            rList.append(r)

            b = int(n/r)
            thresholds.append(((1/b)**(1/r)))

    index_range = range(len(productList))

    for bootstrap in range(1, bootstrapAmount+1):
        bootstrapIndices = np.random.choice(index_range, size=N, replace=True)

        #Want to remove duplicate product draws from the bootstrap indices
        bootstrapIndices = list(set(bootstrapIndices))

        # Obtain products and duplicates in current bootstrap
        currentProductList = []
        currentDuplicateMatrix = np.zeros((len(bootstrapIndices), len(bootstrapIndices)))
        for i in bootstrapIndices:
            currentProductList.append(productList[i])

        for i in range(len(bootstrapIndices)):
            for j in range(len(bootstrapIndices)):
                currentDuplicateMatrix[i, j] = duplicateMatrix[bootstrapIndices[i], bootstrapIndices[j]]

        #Create model word representations
        mwVocabulary, mwList = createModelWords(currentProductList)

        for r in rList:
            b = int(n/r)
            thresholdClustering = 0.9

            primeNumber = 59263579
            aValues = np.random.randint(0, 100000000, size=n)
            bValues = np.random.randint(0, 100000000, size=n)

            hashValues = []
            for x in range(len(mwVocabulary)):
                h = (aValues + x * bValues) % primeNumber
                hashValues.append(h)
            hashValues = np.array(hashValues).T

            minhashSignatures = np.inf * np.ones((n, len(mwList)))

            for column in range(len(mwList)):
                colIndices = mwList[column]
                for j in colIndices:
                    for i in range(n):
                        currentValue = hashValues[i,j]
                        if minhashSignatures[i,column] > currentValue:
                            minhashSignatures[i,column] = currentValue

            bucketCountMatrix = np.zeros((len(mwList), len(mwList)))

            for band in range(b):
                buckets = {}
                for i in range(len(mwList)):
                    bandSignatures = ''.join(map(str, minhashSignatures[band * r : (band + 1) * r, i]))
                    buckets.setdefault(bandSignatures, set()).add(i)

                for bucket in buckets.values():
                    for i in bucket:
                        for j in bucket:
                            if j > i:
                                bucketCountMatrix[i, j] = bucketCountMatrix[i, j] + 1
                                bucketCountMatrix[j, i] = bucketCountMatrix[j, i] + 1

            duplicateMatrixLSH = np.zeros((len(currentProductList), len(currentProductList)))
            amountComparisonsLSH = 0

            for i in range(len(currentProductList)):
                for j in range(len(currentProductList)):
                    if bucketCountMatrix[i, j] > 0:
                        duplicateMatrixLSH[i, j] = 1
                        amountComparisonsLSH += 1

            maxDistance = 9999
            distanceMatrix = maxDistance * np.ones((len(currentProductList), len(currentProductList)))
            amountComparisonsClustering = 0
            shingleDictionary = {}

            for i in range(len(distanceMatrix)):
                distanceMatrix[i, i] = 0

            #Create shingles beforehand for products that have at least one duplicate to save computation time
            for i in range(len(currentProductList)):
                if sum(duplicateMatrixLSH[i, :]) > 0:
                    productShingles = createProductShingles(currentProductList[i], shingleLength)
                    shingleDictionary[i] = productShingles

            for i in range(len(currentProductList)):
                for j in range(i+1, len(currentProductList)):
                    if duplicateMatrixLSH[i,j] == 0:
                        continue

                    iProduct = currentProductList[i]
                    jProduct = currentProductList[j]

                    if iProduct['shop'].lower() == jProduct['shop'].lower():
                        distanceMatrix[i, j] = maxDistance
                        distanceMatrix[j, i] = maxDistance
                        continue

                    iProductBrand = iProduct['featuresMap'].get('brand')
                    jProductBrand = jProduct['featuresMap'].get('brand')

                    if iProductBrand is not None and iProductBrand == jProductBrand:
                        distanceMatrix[i, j] = maxDistance
                        distanceMatrix[j, i] = maxDistance
                        continue

                    iProductShingles = shingleDictionary[i]
                    jProductShingles = shingleDictionary[j]

                    shingleIntersectionLength = len(iProductShingles.intersection(jProductShingles))
                    shingleUnionLength = len(iProductShingles.union(jProductShingles))

                    distanceJaccard = 1 - (shingleIntersectionLength / shingleUnionLength)

                    distanceMatrix[i, j] = distanceJaccard
                    distanceMatrix[j, i] = distanceJaccard

                    amountComparisonsClustering += 1

            clusteringModel = AgglomerativeClustering(distance_threshold=thresholdClustering,n_clusters = None,metric='precomputed',linkage='single')
            clusteringModel.fit(distanceMatrix)

            clusteringDuplicates = np.zeros((len(currentProductList), len(currentProductList)))
            for i in range(len(currentProductList)):
                for j in range(i+1, len(currentProductList)):
                    if clusteringModel.labels_[i] == clusteringModel.labels_[j]:
                        clusteringDuplicates[i, j] = 1
                        clusteringDuplicates[j, i] = 1

            correctDuplicatesLSH = 0
            correctDuplicatesClustering = 0
            currentDuplicateAmount = 0

            for i in range(len(mwList)):
                for j in range(i+1, len(mwList)):
                    if currentDuplicateMatrix[i,j] > 0:
                        currentDuplicateAmount += 1
                        if currentDuplicateMatrix[i,j] == duplicateMatrixLSH[i,j]:
                            correctDuplicatesLSH += 1
                        if currentDuplicateMatrix[i,j] == clusteringDuplicates[i,j]:
                            correctDuplicatesClustering += 1

            pairQualityLSH = correctDuplicatesLSH / amountComparisonsLSH
            pairQualityClustering = correctDuplicatesClustering / amountComparisonsClustering

            pairCompletenessLSH = correctDuplicatesLSH / currentDuplicateAmount
            pairCompletenessClustering = correctDuplicatesClustering / currentDuplicateAmount

            f1LSH = (2 * pairQualityLSH * pairCompletenessLSH) / (pairQualityLSH + pairCompletenessLSH)
            f1Clustering = (2 * pairQualityClustering * pairCompletenessClustering) / (pairQualityClustering + pairCompletenessClustering)

            TP_LSH = 0
            FP_LSH = 0
            FN_LSH = 0
            TP_Clust = 0
            FP_Clust = 0
            FN_Clust = 0

            for i in range(len(mwList)):
                for j in range(i + 1, len(mwList)):
                    if currentDuplicateMatrix[i, j] == 1:
                        if duplicateMatrixLSH[i, j] == 1:
                            TP_LSH += 1
                        else:
                            FN_LSH += 1
                        if clusteringDuplicates[i, j] == 1:
                            TP_Clust += 1
                        else:
                            FN_Clust += 1
                    if currentDuplicateMatrix[i, j] == 0:
                        if duplicateMatrixLSH[i, j] == 1:
                            FP_LSH += 1
                        if clusteringDuplicates[i, j] == 1:
                            FP_Clust += 1

            F1StarLSH = (2 * TP_LSH) / (2 * TP_LSH + FP_LSH + FN_LSH)
            F1StarClust = (2 * TP_Clust) / (2 * TP_Clust + FP_Clust + FN_Clust)

            resultsF1StarLSH.append(F1StarLSH)
            resultsF1StarClust.append(F1StarClust)

            resultsPQLSH.append(pairQualityLSH)
            resultsPCLSH.append(pairCompletenessLSH)
            resultsF1LSH.append(f1LSH)
            resultsPQClust.append(pairQualityClustering)
            resultsPCClust.append(pairCompletenessClustering)
            resultsF1Clust.append(f1Clustering)

    averagePQLSH = np.zeros(len(thresholds))
    averagePCLSH = np.zeros(len(thresholds))
    averageF1LSH = np.zeros(len(thresholds))
    averagePQClust = np.zeros(len(thresholds))
    averagePCClust = np.zeros(len(thresholds))
    averageF1Clust = np.zeros(len(thresholds))
    averageF1StarLSH = np.zeros(len(thresholds))
    averageF1StarClust = np.zeros(len(thresholds))

    print("Best F1-measure for LSH" + str(max(resultsF1LSH)))
    print("Best F1-measure for Clustering" + str(max(resultsF1Clust)))
    print("Best F1Star-measure for LSH" + str(max(resultsF1StarLSH)))
    print("Best F1Star-measure for Clustering" + str(max(resultsF1StarClust)))

    print("Thresholds:")
    print(thresholds)
    for i in range(len(resultsPCLSH)):
        averagePQLSH[i % len(thresholds)] += (1 / bootstrapAmount) * resultsPQLSH[i]
        averagePCLSH[i % len(thresholds)] += (1 / bootstrapAmount) * resultsPCLSH[i]
        averageF1LSH[i % len(thresholds)] += (1 / bootstrapAmount) * resultsF1LSH[i]
        averagePQClust[i % len(thresholds)] += (1 / bootstrapAmount) * resultsPQClust[i]
        averagePCClust[i % len(thresholds)] += (1 / bootstrapAmount) * resultsPCClust[i]
        averageF1Clust[i % len(thresholds)] += (1 / bootstrapAmount) * resultsF1Clust[i]
        averageF1StarLSH[i % len(thresholds)] += (1 / bootstrapAmount) * resultsF1StarLSH[i]
        averageF1StarClust[i % len(thresholds)] += (1 / bootstrapAmount) * resultsF1StarClust[i]

    print("Average Pair Quality LSH:")
    print(averagePQLSH)

    print("Average Pair Completeness LSH:")
    print(averagePCLSH)

    print("Average F1-measure LSH:")
    print(averageF1LSH)

    print("Average Pair Quality Clustering:")
    print(averagePQClust)

    print("Average Pair Completeness Clustering:")
    print(averagePCClust)

    print("Average F1-measure Clustering:")
    print(averageF1Clust)

    print("Average F1Star-Measure LSH:")
    print(averageF1StarLSH)

    print("Average F1Star-Measure Clust:")
    print(averageF1StarClust)


if __name__ == '__main__':
    main()