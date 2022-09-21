from scipy.io import arff
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
import matplotlib.pyplot as plt

def trainNetwork(Feactures,labels):
    

    X_train=Feactures
    y_train=labels
    #svclassifier = SVC(kernel='sigmoid', C = 1.0)
    svclassifier = SVC(gamma='auto')
    print(X_train.shape)
    print(y_train.shape)
    print(np.unique(y_train))
    X_train1=np.reshape(X_train,(-1,1))

    Y_train1=np.reshape(y_train,(-1,1))
       

    svclassifier.fit(X_train1, Y_train1)
    y_pred = svclassifier.predict(X_train1)
    confusion_matrix(y_pred, Y_train1)
    
    print()
    
