import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
from collections import Counter
import random 

path ="E:\\FCI\\fourth year\\Machine Learning\\Assignments\\Assignment2\\part1\\Part 1 Dataset\\house-votes-84.data.csv" 
dataset = pd.read_csv(path,header=None)
dataset.columns = ["c0","c1","c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9","c10","c11","c12","c13","c14", "c15", "c16"]


def fillMissing(data,colNames):
    #filling the missing vote
    for i in colNames:
        values = data[i].value_counts().keys().tolist()
        value = values[0]
        data[i] = data[i].replace(['?'],value)
    return data 
         
#fill missing votes
data = fillMissing(dataset,dataset.columns)



def check_purity(data):
    label = data.iloc[:,-1]
    unique_values = np.unique(label)
    if len(unique_values) == 1:
        return True
    return False


def divideData(data,percentage):
    temp= data
    chosen = []
    k = len(data) * percentage // 100
    chosen = random.sample(range(0,434),k)
    indexes_to_keep = set(range(temp.shape[0])) - set(chosen)
    testdata = temp.take(list(indexes_to_keep))
    traindata = temp.take(list(chosen))
    return  traindata , testdata   


def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    Entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return Entropy

def calculateInformationGain(data,colName):
    Entropy = entropy(data["c0"])
    vals,counts= np.unique(data[colName],return_counts=True)
    AvgEntropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[colName]==vals[i]).dropna()["c0"]) for i in range(len(vals))])
    Information_Gain = Entropy - AvgEntropy
    return Information_Gain
       

def create_decision_tree(dataset,data,colNames,leaf = None, size = 0):
    #base case
    if check_purity(dataset):
        return np.unique(dataset["c0"])[0] 

    elif len(colNames)==0:
        return leaf   

    #recursion
    else:
        v = np.unique(dataset["c0"])
        indx = np.argmax(np.unique(dataset["c0"],return_counts=True)[1])
        leaf = v[indx]
        InformationGain = []
        for colName in colNames:
            InformationGain.append(calculateInformationGain(dataset,colName))
        #print "Information gain ", InformationGain
        IndexOfCol = np.argmax(InformationGain)
        bestCol = colNames[IndexOfCol]
        decision_tree = {bestCol:{}}
        size = size + 1
        colNames = [i for i in colNames if i != bestCol]
        #print "size ", size
        for value in np.unique(dataset[bestCol]):
            sub_data = dataset.where(dataset[bestCol] == value).dropna()
            subtree = create_decision_tree(sub_data,data,colNames,leaf,size)
            decision_tree[bestCol][value] = subtree
        return(decision_tree)



    
def predict(query,tree,default):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result,default)

            else:
                return result

        
def test(data,tree):
    data = data.reset_index(drop=True)
    queries = data.iloc[:,1:].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    default = np.unique(data["c0"])[np.argmax(np.unique(data["c0"],return_counts=True)[1])]
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,default) 
    correctPrediction = [1 if (a == b) else 0 
                        for (a, b) in zip(predicted["predicted"], data["c0"])]

    accuracy = (np.sum(correctPrediction)*1.0 / len(correctPrediction)*1.0) * 100.0
    accuracy = int(accuracy)
    return accuracy


def main():  
    percentage = [30,40,50,60,70]
    for p in percentage:
        print "\n Percentage ", p
        accuracies = [] 
        for i in range(0,5): 
            training_data, testing_data = divideData(data,p)
            decision_tree = create_decision_tree(training_data,training_data,training_data.columns[:-1])
            #print "\n \n Decision Tree \n"
            #print decision_tree
            test(testing_data,decision_tree)
            accuracies.append(test(testing_data,decision_tree))
        #print accuracies    
        print "mean accuracy: ", np.mean(accuracies), " maximum accuracy: ", max(accuracies), " minimum accuracy: ", min(accuracies)


main()
#print data.head()     