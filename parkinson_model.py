import pandas as pd
"""
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
"""
df=pd.read_csv("Parkinson_dataset.csv")
pd.set_option('display.max_columns',None)
df.head()
"""
df.shape

df.info()

df.isnull().sum()

# df.types()


df['status'].value_counts()

temp = df["status"].value_counts()
temp_df=pd.DataFrame({"status":temp.index,"values":temp.values})
print(sns.barplot(x="status",y="values",data=temp_df))
plt.show()

x=df.drop(["status","name"],axis=1)
y=df["status"]

list_accuracy=[]

#Applyling all the Algorithms

Lr = []
Svm = []
Knn = []

# creating arrays to store predictions of all the models.
lr_pred = []
svm_pred = []
knn_pred = []
# appling train test in 80-20 split
for i in range(3):
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20 +(i*0.05),random_state=2)
  #____________________Applying logistic Regression_____________________

  from sklearn.linear_model import LogisticRegression
  lr = LogisticRegression(C=0.4,max_iter=1000,solver='liblinear')
  lr.fit(x_train,y_train)
  #prediction
  lr_pred.append(lr.predict(x_test))
  #accuracy
  Lr.append(accuracy_score(y_test,lr_pred[i]))

  #____________Similarly Applying SVM__________________________
  from sklearn.svm import SVC
  svm = SVC(cache_size=100)
  svm.fit(x_train,y_train)
  #prediction
  svm_pred.append(svm.predict(x_test))
  #accuracy
  Svm.append(accuracy_score(y_test,svm_pred[i]))

  #_____________________Applying KNN______________________
  from sklearn.neighbors import KNeighborsClassifier

  knn =  KNeighborsClassifier(n_neighbors=3)
  knn.fit(x_train,y_train)
  #predicting test N=3
  knn.predict(x_test)
  #accuracy
  # Knn.append(accuracy_score(y_test,k))

for i in range(3):
  print(f"Accuracy of Logistic regression at {80 - (i*5)}-{20 + (i*5)} split        :",Lr[i])
print()
for i in range(3):
  print(f"Accuracy of SVM at {80 - (i*5)}-{20 + (i*5)} split                        :",Svm[i])
print()
for i in range(3):
  print(f"Accuracy of KNN at {80 - (i*5)}-{20 + (i*5)} split                        :",Knn[i])
print()

for i in range(3):
  list1=['logisticRegression','SVM','KNN']
  list2=[Lr[i],Svm[i],Knn[i]]
  list3=[lr,svm,knn]  
  df_Accuracy=pd.DataFrame({"Method Used":list1,"Accuracy":list2})
  print(df_Accuracy)
  chart=sns.barplot(x='Method Used',y='Accuracy',data=df_Accuracy)
  chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
  print(chart)
  plt.show()

# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_test,x_test)

# from sklearn.metrics import f1_score
# f1_score(y_test,model_xx_test),average='binary'

# from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report,accuracy_score
# print(classification_report(y_test,model_xg.predict(x_test)))
# print('confusion Matrix')
# print(cm)

# for i in list3:
#   print("______________________",i,"__________________")
#   print(classification_report(y_test,i.predict(x_test)))
#   print("Confusion Matrix")
#   print(cm)

# for i in range(3):
#   print(f"train test split at {80 - (i*5)}-{20 + (i*5)}")
#   print("Classification report of Logistic regression \n",classification_report(y_test,lr_pred[i]))
#   print("Classification report of SVM \n",classification_report(y_test,svm_pred[i]))
#   # print("Classification report of KNN \n",classification_report(y_test,knn_pred[i]))

# using svm model 
# using pickle to use svm model in app.py
import pickle

pickle.dump(svm, open('model.pkl', 'wb'))
pickled_model=pickle.load(open('model.pkl', 'rb'))
print(x_test[:2])
"""
