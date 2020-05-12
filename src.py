# Importer les librairies
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import math
import random
import matplotlib.pyplot as plt
import findspark
findspark.init()
from pyspark.sql import SparkSession

#creer session Spark
appName = "Analyse des sentiments Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# charger le dataset
data_set = []
labels = []
data_path = 'C:/Users/MAS/Downloads/amazon_cells/amazon_cells_labelled.txt'

# charger les donnees dans le tableau data_set
with open(data_path) as dp:
    line = dp.readline()
    while line:
        data_set.append(line.strip())
        line = dp.readline()

def createData(dataset):
    d = [i[:-3] for i in dataset]
    d = [item.lower() for item in d]
    for i in range(len(d)):
        d[i] = re.sub('[^A-Za-z0-9-\s]+', '', d[i])
    l = [i[-1:] for i in dataset]
    for i in range(0, len(l)): 
        l[i] = int(l[i])
    dataset = list(zip(d , l))   
    return dataset

data = createData(data_set)

def train_test_split(dataset, value):
    index = int(value*len(dataset))
    train = dataset[:index]
    test = dataset[index:]  
    return train, test

train, test = train_test_split(data,0.8)

# converter train_data a un tableau
train_data=np.asarray(train)

# converter test_data a un tableau
test_data=np.asarray(test)

# afficher les donnees
print(train_data)
print('\n\n\n\n\n')
print(test_data)


# compter la frequence des mots dans notre dataset
vectorizer=CountVectorizer()
counts=vectorizer.fit_transform(train_data[:,0])

# appliquer Naive Bayes avec MultinomialNB
classifer=MultinomialNB()
targets=train_data[:,1]
classifer.fit(counts,targets)

# faire une prediction sur les donnees du test
example_counts=vectorizer.transform(test_data[:,0])
predictions=classifer.predict(example_counts)
print(predictions)

# Calculer le degree de precision
x=0
for i in range(len(predictions)):
    if test_data[i][1]==predictions[i]:
        x+=1
print ("Precision : ", 100*x/len(predictions))