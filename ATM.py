from atm import ATM
import os
import numpy as np
import sqlite3
from sklearn.metrics import confusion_matrix
import pickle
from atm import Model
import random
from scipy import stats
import sys
import time
import pandas as pd
import shutil
def confusion(g_turth,predictions):
    tn, fp, fn, tp = confusion_matrix(g_turth,predictions).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    sensitivity = (tp)/(tp+fn)
    specificty = (tn)/(tn+fp)
    return accuracy,sensitivity,specificty

site = sys.argv[1]



if os.path.exists('models'):
    shutil.rmtree('models')


if os.path.exists('metrics'):
    shutil.rmtree('metrics')


np.random.seed(10)
random.seed(10)
res_true={}
start = time.time()
res = []
for i in range(5): 

    path="./features"
    path = path+site+"/"
    pathcont="_train.csv"
    pathconttest="_test.csv"
    fullpathtrain=path+str(i)+"_True"+pathcont
    fullpathtest = path+str(i)+"_True"+pathconttest
    atm = ATM()
    results = atm.run(train_path = fullpathtrain,budget=50,budget_type = 'classifier',metric = 'accuracy',methods = ['svm'],score_target='cv')
    sqlite_file = './atm.db'
    table_name = 'classifiers'
    col_name = 'cv_judgment_metric'
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    c.execute("Select model_location, id from {tn} order by {col} desc limit 10".format(tn = table_name,col = col_name))
    all_rows = c.fetchall()


    clfs_prdictions = []
    for fc in range(10):
        m = Model.load("./"+all_rows[fc][0])
        data = pd.read_csv(fullpathtest)
        clfs_prdictions.append(m.predict(data).values)
    clfs_prdictions = np.array(clfs_prdictions)   
    dftest = pd.read_csv(fullpathtest)
    acutal_labels = dftest.iloc[:,-1:]
    predicted_labels = stats.mode(clfs_prdictions,axis=0)[0][0]
    print(confusion(acutal_labels,predicted_labels))
    res.append(confusion(acutal_labels,predicted_labels))
    os.remove('atm.db')
print(res)
print("\n avergae result Auto-ASD-Network \n Accuracy Sensitivity Specificity: \n",site,np.mean(res,axis=0),"\n--------------------------------\n")
res_true[site]=res    



Description = 'Auto-ASD-Network'
with open("./results/"+Description, 'w') as file:
    file.write("Accuracy Sensitivity Specificity\n")
    np.savetxt(file,[np.mean(res,axis=0)],fmt='%1.2f')


if os.path.exists('models'):
    shutil.rmtree('models')


if os.path.exists('metrics'):
    shutil.rmtree('metrics')


np.random.seed(10)
random.seed(10)
res_true={}
import time
start = time.time()
res = []

for i in range(5):

    path="./features"
    path = path+site+"/"
    pathcont="_train.csv"
    pathconttest="_test.csv"
    fullpathtrain=path+str(i)+"_False"+pathcont
    fullpathtest = path+str(i)+"_False"+pathconttest
    atm = ATM()
    results = atm.run(train_path = fullpathtrain,budget=50,budget_type = 'classifier',metric = 'accuracy',methods = ['svm'],score_target='cv')
    sqlite_file = './atm.db'
    table_name = 'classifiers'
    col_name = 'cv_judgment_metric'
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    c.execute("Select model_location, id from {tn} order by {col} desc limit 10".format(tn = table_name,col = col_name))
    all_rows = c.fetchall()

    clfs_prdictions = []
    for fc in range(10):
        m = Model.load("./"+all_rows[fc][0])
        data = pd.read_csv(fullpathtest)
        clfs_prdictions.append(m.predict(data).values)
    clfs_prdictions = np.array(clfs_prdictions)
    dftest = pd.read_csv(fullpathtest)
    acutal_labels = dftest.iloc[:,-1:]
    predicted_labels = stats.mode(clfs_prdictions,axis=0)[0][0]
    print(confusion(acutal_labels,predicted_labels))
    res.append(confusion(acutal_labels,predicted_labels))
    os.remove('atm.db')

print("\navergae result MLP-SVM-ATM: \n Accuracy Sensitivity Specificity: \n",site,np.mean(res,axis=0),"\n------------------------------\n")
Description = 'MLP-SVM-ATM'

with open("./results/"+Description, 'w') as file:
    file.write("Accuracy Sensitivity Specificity\n")
    np.savetxt(file,[np.mean(res,axis=0)],fmt='%1.2f')



