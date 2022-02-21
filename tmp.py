
from  main import *

if __name__ == "__main__":
   df=pd.read_csv('sonar.csv',header=None)    # Name of the input numeric feature file in .csv format
   shuffle(df)
   data=df.sample(frac=1)
   n,m=data.shape
   print(n,m)

   x=data.drop(data.columns[len(data.columns)-1],1)
   Y=data[data.columns[len(data.columns)-1]]

   X=x.as_matrix()
   y=Y.as_matrix()
   print("Features in Original Dataset:")
   p,pp=X.shape
   print(pp)


   # Dividing data into 5 parts where 4 parts are used for training and 1 for testing in each iteration

   train1=X[:(int)(0.8*n),:]
   test1=X[(int)(0.8*n):,:]

   train2=X[(int)(0.2*n):,:]
   test2=X[:(int)(0.2*n),:]

   train3=np.concatenate((X[:(int)(0.6*n),:],X[(int)(0.8*n):,:]),axis=0)
   test3=X[(int)(0.6*n):(int)(0.8*n),:]

   train4=np.concatenate((X[:(int)(0.4*n),:],X[(int)(0.6*n):,:]),axis=0)
   test4=X[(int)(0.4*n):(int)(0.6*n),:]

   train5=np.concatenate((X[:(int)(0.2*n),:],X[(int)(0.4*n):,:]),axis=0)
   test5=X[(int)(0.2*n):(int)(0.4*n),:]

   train1Y=y[:(int)(0.8*n)]
   test1Y=y[(int)(0.8*n):]

   train2Y=y[(int)(0.2*n):]
   test2Y=y[:(int)(0.2*n)]

   list1=y[:(int)(0.6*n)]
   list2=y[(int)(0.8*n):]
   train3Y=np.append(list1,list2)
   test3Y=y[(int)(0.6*n):(int)(0.8*n)]

   list1=y[:(int)(0.4*n)]
   list2=y[(int)(0.6*n):]
   train4Y=np.append(list1,list2)
   test4Y=y[(int)(0.4*n):(int)(0.6*n)]

   list1=y[:(int)(0.2*n)]
   list2=y[(int)(0.4*n):]
   train5Y=np.append(list1,list2)
   test5Y=y[(int)(0.2*n):(int)(0.4*n)]

   original={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
   			 'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   orig_ig={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
   				'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   new={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,'AdaBoost':0,\
        'Neural Network':0,'Decision Tree':0}
   new_fs={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,'AdaBoost':0,\
        'Neural Network':0,'Decision Tree':0}
   supplement={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
   			   'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   supplement_ig={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
                  'AdaBoost':0,'Neural Network':0,'Decision Tree':0}
   stable_ig={'kNN':0,'Logistic Regression':0,'Linear SVM':0,'Poly SVM':0,'Random Forest':0,\
                  'AdaBoost':0,'Neural Network':0,'Decision Tree':0}


   #############################################################################
                # Computing Accuracy for each fold of Cross Validation
   #############################################################################

   original_ig_train1, original_ig_test1=original_ig(train1,test1,train1Y)  # No normalization needed for original training & testing
   

   """original_ig_train1=original_ig_train1.as_matrix()
   original_ig_test1=original_ig_test1.as_matrix()"""

   linear_correlated_1,nonlinear_correlated_1=dependent(original_ig_train1, 0.7, 1)
   a2,a1=linear(original_ig_train1, original_ig_test1, 1)
   a4,a3=nonlinear(original_ig_train1, original_ig_test1, 1)

   """a1=pd.read_csv('sonar_related_lineartest_1.csv',header=None)          # all predicted feature files
   a2=pd.read_csv('sonar_related_lineartrain_1.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_1.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_1.csv',header=None)"""


   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])      # Train
   r3=np.hstack([a1,a3])      # Test

   scaler=StandardScaler().fit(r4) # Normalization  & fit only on training
   p2=scaler.transform(r4)     # Normalized Train
   p1=scaler.transform(r3)     # Normalized Test

   f1,f2,st_f1,st_f2=stable(p2,p1,train1Y)
   """f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)"""

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test1, f2])  # original test features, selected by IG, f2 is feature space after ensemble selection.
   x2X=np.hstack([train1, f1])

   scaler=StandardScaler().fit(x2X)  # Again normalization of the complete combined feature pool
   x2=scaler.transform(x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test1, f2])
   y2Y=np.hstack([train1, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   """st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)"""

   st_x1X=np.hstack([original_ig_test1, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train1, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("............................................................................................................................")

   print("Predicting Accuracies")

   names=['kNN','Logistic Regression','Linear SVM','Poly SVM','Random Forest','AdaBoost','Neural Network','Decision Tree']
   models=[KNeighborsClassifier(), LogisticRegression(), svm.LinearSVC(),SVC(C=1.0, kernel='poly'),
           RandomForestClassifier(),AdaBoostClassifier(), MLPClassifier(), tree.DecisionTreeClassifier()]

   print("....................Results on Original Features...............................")

   for i in range(0,len(models)):
      models[i].fit(train1,train1Y)
      y_out= models[i].predict(test1)
      print(models[i].score(test1,test1Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test1,test1Y)

   print("....................Results on (Original + IG) Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train1,train1Y)
      y_out= models[i].predict(original_ig_test1)
      print(models[i].score(original_ig_test1,test1Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test1, test1Y)

   print("...................Results on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train1Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test1Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test1Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train1Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test1Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test1Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train1Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test1Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test1Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train1Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test1Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test1Y)

   print("...................Results on (7).............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train1Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test1Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test1Y)

   rank(x2,train1Y) # - rank function is for plotting graph - sec 5 in paper

   print("################################################################################")
   print("################################################################################")

   original_ig_train2,original_ig_test2=original_ig(train2,test2,train2Y)  # No normalization needed for original training & testing
   """original_ig_train2=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test2=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train2=original_ig_train2.as_matrix()
   original_ig_test2=original_ig_test2.as_matrix()"""

   dependent(original_ig_train2, 0.7, 2)
   linear(original_ig_train2, original_ig_test2, 2)
   nonlinear(original_ig_train2, original_ig_test2, 2)

   a1=pd.read_csv('sonar_related_lineartest_2.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_2.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_2.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_2.csv',header=None)

   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])
   #r4=a4
   #r3=a3

   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   f1,f2,st_f1,st_f2=stable(p2,p1,train2Y)
   """f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)"""

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test2,f2])
   x2X=np.hstack([train2,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test2, f2])
   y2Y=np.hstack([train2, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   """st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)"""

   st_x1X=np.hstack([original_ig_test2, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train2, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print("....................Results on Original Features...............................")

   for i in range(0,len(models)):
      models[i].fit(train2,train2Y)
      y_out= models[i].predict(test2)
      print(models[i].score(test2,test2Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test2,test2Y)

   print("....................Results on (Original + IG) Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train2,train2Y)
      y_out= models[i].predict(original_ig_test2)
      print(models[i].score(original_ig_test2,test2Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test2,test2Y)

   print("...................Results on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train2Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test2Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test2Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train2Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test2Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test2Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train2Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test2Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test2Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train2Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test2Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test2Y)

   rank(x2,train2Y)

   print("...................Results on (7).............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train2Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test2Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test2Y)

   print("################################################################################")
   print("################################################################################")

   original_ig_train5,original_ig_test5=original_ig(train5,test5,train5Y)  # No normalization needed for original training & testing
   """original_ig_train5=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test5=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train5=original_ig_train5.as_matrix()
   original_ig_test5=original_ig_test5.as_matrix()"""

   dependent(original_ig_train5, 0.7, 5)
   linear(original_ig_train5, original_ig_test5, 5)
   nonlinear(original_ig_train5, original_ig_test5, 5)

   a1=pd.read_csv('sonar_related_lineartest_5.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_5.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_5.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_5.csv',header=None)
   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])

   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   f1,f2,st_f1,st_f2=stable(p2,p1,train5Y)
   """f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)"""

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test5,f2])
   x2X=np.hstack([train5,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test5, f2])
   y2Y=np.hstack([train5, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   """st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)"""

   st_x1X=np.hstack([original_ig_test5, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train5, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print(".................... Results on Original Features ...............................")

   for i in range(0,len(models)):
      models[i].fit(train5,train5Y)
      y_out= models[i].predict(test5)
      print(models[i].score(test5,test5Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test5,test5Y)

   print("....................Results on (Original + IG )Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train5,train5Y)
      y_out= models[i].predict(original_ig_test5)
      print(models[i].score(original_ig_test5,test5Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test5,test5Y)

   print("...................Results only on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train5Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test5Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test5Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train5Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test5Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test5Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train5Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test5Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test5Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train5Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test5Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test5Y)

   print("...................Results when full architecture of AutoLearn is followed.............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train5Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test5Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test5Y)

   rank(x2,train5Y)
   print("################################################################################")
   print("################################################################################")


   original_ig_train4,original_ig_test4=original_ig(train4,test4,train4Y)  # No normalization needed for original training & testing
   """original_ig_train4=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test4=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train4=original_ig_train4.as_matrix()
   original_ig_test4=original_ig_test4.as_matrix()"""

   dependent(original_ig_train4,0.7, 4)
   linear(original_ig_train4,original_ig_test4, 4)
   nonlinear(original_ig_train4,original_ig_test4, 4)

   a1=pd.read_csv('sonar_related_lineartest_4.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_4.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_4.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_4.csv',header=None)

   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])
   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   f1,f2,st_f1,st_f2=stable(p2,p1,train4Y)
   """f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)"""

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test4,f2])
   x2X=np.hstack([train4,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test4, f2])
   y2Y=np.hstack([train4, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   """st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)"""

   st_x1X=np.hstack([original_ig_test4, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train4, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print("....................Results on Original Features...............................")

   for i in range(0,len(models)):
      models[i].fit(train4,train4Y)
      y_out= models[i].predict(test4)
      print(models[i].score(test4,test4Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test4,test4Y)

   print("....................Results on (Original + IG) Stable Features...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train4,train4Y)
      y_out= models[i].predict(original_ig_test4)
      print(models[i].score(original_ig_test4,test4Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test4,test4Y)

   print("...................Results only on Newly constructed Features.........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train4Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test4Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test4Y)

   print("...................Results after R2R.........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train4Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test4Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test4Y)

   print("...................Results on (5).............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train4Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test4Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test4Y)

   print("...................Results on (6).............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train4Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test4Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test4Y)

   print("................... Results when full architecture of AutoLearn is followed .............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train4Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test4Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test4Y)

   rank(x2,train4Y)
   print("################################################################################")
   print("################################################################################")

   original_ig_train3,original_ig_test3=original_ig(train3,test3,train3Y)  # No normalization needed for original training & testing
   """original_ig_train3=pd.read_csv('sonar_original_ig_trainfeatures.csv',header=None)
   original_ig_test3=pd.read_csv('sonar_original_ig_testfeatures.csv',header=None)

   original_ig_train3=original_ig_train3.as_matrix()
   original_ig_test3=original_ig_test3.as_matrix()"""

   dependent(original_ig_train3, 0.7, 3)
   linear(original_ig_train3,original_ig_test3, 3)
   nonlinear(original_ig_train3,original_ig_test3, 3)

   a1=pd.read_csv('sonar_related_lineartest_3.csv',header=None)
   a2=pd.read_csv('sonar_related_lineartrain_3.csv',header=None)
   a3=pd.read_csv('sonar_related_nonlineartest_3.csv',header=None)
   a4=pd.read_csv('sonar_related_nonlineartrain_3.csv',header=None)
   #r4=a4
   #r3=a3
   r4=np.hstack([a2,a4])
   r3=np.hstack([a1,a3])
   scaler=StandardScaler().fit(r4)
   p2=scaler.transform(r4)
   p1=scaler.transform(r3)

   f1,f2,st_f1,st_f2=stable(p2,p1,train3Y)
   """f1=pd.read_csv('sonar_ensemble_trainfeatures.csv',header=None)
   f2=pd.read_csv('sonar_ensemble_testfeatures.csv',header=None)"""

   scaler=StandardScaler().fit(f1)
   e_f1=scaler.transform(f1)
   e_f2=scaler.transform(f2)

   x1X=np.hstack([test3,f2])
   x2X=np.hstack([train3,f1])

   scaler=StandardScaler().fit(x2X)
   x2=scaler.transform(x2X)
   x1=scaler.transform(x1X)

   y1Y=np.hstack([test3, f2])
   y2Y=np.hstack([train3, f1])

   scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
   y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
   y1=scaler.transform(y1Y)

   """st_f1=pd.read_csv('sonar_stable_trainfeatures.csv',header=None)
   st_f2=pd.read_csv('sonar_stable_testfeatures.csv',header=None)"""

   st_x1X=np.hstack([original_ig_test3, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
   st_x2X=np.hstack([original_ig_train3, st_f1])

   scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
   st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
   st_x1=scaler.transform(st_x1X)

   print("Predicting Accuracies")
   print(".................... Results on Original Features ...............................")

   for i in range(0,len(models)):
      models[i].fit(train3,train3Y)
      y_out= models[i].predict(test3)
      print(models[i].score(test3,test3Y)," ..... ",names[i])
      original[names[i]]+=models[i].score(test3,test3Y)

   print(".................... Results on (Original + IG) Stable Features ...............................")

   for i in range(0,len(models)):
      models[i].fit(original_ig_train3,train3Y)
      y_out= models[i].predict(original_ig_test3)
      print(models[i].score(original_ig_test3,test3Y)," ..... ",names[i])
      orig_ig[names[i]]+=models[i].score(original_ig_test3,test3Y)

   print("................... Results on just the Newly constructed Features .........................")

   for i in range(0,len(models)):
      models[i].fit(p2,train3Y)
      y_out= models[i].predict(p1)
      print(models[i].score(p1,test3Y)," ..... ",names[i])
      new[names[i]]+=models[i].score(p1,test3Y)

   print("................... Results after Newly constructed Features with feature selection .........................")

   for i in range(0,len(models)):
      models[i].fit(e_f1,train3Y)
      y_out= models[i].predict(e_f2)
      print(models[i].score(e_f2,test3Y)," ..... ",names[i])
      new_fs[names[i]]+=models[i].score(e_f2,test3Y)

   print("................... Results on (5) .............................")

   for i in range(0,len(models)):
      models[i].fit(y2,train3Y)
      output= models[i].predict(y1)
      print(models[i].score(y1,test3Y)," ..... ",names[i])
      supplement[names[i]]+=models[i].score(y1,test3Y)

   print("................... Results when full architecture of AutoLearn is followed (IG & then Stability in feature selection) .............................")

   for i in range(0,len(models)):
      models[i].fit(x2,train3Y)
      y_out= models[i].predict(x1)
      print(models[i].score(x1,test3Y)," ..... ",names[i])
      supplement_ig[names[i]]+=models[i].score(x1,test3Y)

   rank(x2,train3Y)
   print("................... Results when full architecture of AutoLearn is followed (Stability only in feature selection) .............................")

   for i in range(0,len(models)):
      models[i].fit(st_x2,train3Y)
      y_out= models[i].predict(st_x1)
      print(models[i].score(st_x1,test3Y)," ..... ",names[i])
      stable_ig[names[i]]+=models[i].score(st_x1,test3Y)


   print("################################################################################")
   print("################################################################################")
   #rank(Train1,y_train)
   #rank(Train,y_train)
   '''
   print("Original features", pp)
   print("Selected after IG (Avg)", len_orig_ig/5)
   print("---------------------------------------------")
   print("New Features Constructed (Avg)", nc_val/5)
   print("Features Selected after Stability Selection(Avg)", stable_val/5)
   print("---------------------------------------------")
   print("Features selected after ensemble (Avg)", ensemble_val/5)
   '''
   
   print("Accuracies :")

   print("................... Average of results after 5 fold CV in the same order as above .............................")


   for i in range(0,len(models)):
       print(names[i])
       print((original[names[i]]/5)*100)
       print((orig_ig[names[i]]/5)*100)
       print((new[names[i]]/5)*100)
       print((new_fs[names[i]]/5)*100)
       print((supplement[names[i]]/5)*100)
       print((supplement_ig[names[i]]/5)*100)
       print((stable_ig[names[i]]/5)*100)
       print("--------------------------")

   print("DONE !!!")
