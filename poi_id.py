#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

##Dataset Exploration and Outlier Detection
#Print Number of data points
print "***********************************************************************************************************"
print "Number of Data Points: ", len(my_dataset)
#Investigate Features
all_features=[]
for key,value in my_dataset.iteritems():
    for k,v in value.iteritems():
      all_features.append(k)
    break
print "List of All Features available:",all_features

#Number of POIs and Non POIs
poi=0
npoi=0
invalid_poi=0
for key,value in my_dataset.iteritems():
    if value['poi'] == 1:
        poi = poi+1
    elif value['poi'] == 0:
        npoi = npoi+1
    else:
        invalid_poi = invalid_poi+1
print  "Number of POIs:" ,poi
print  "Number of Non POIs:",npoi
print "Number of Invalid POIs:",invalid_poi
print "***********************************************************************************************************"

#Find Missing Features
df=pd.DataFrame.from_dict(my_dataset,orient='index')
df=df.replace('NaN',np.nan)
missing_features_percentage = (df.isnull().sum()/146)*100
print "Percentage of missing fearures:"
print missing_features_percentage
remove_features=['loan_advances','director_fees','restricted_stock_deferred','deferral_payments','deferred_income','long_term_incentive','email_address']
features_list= [feat for feat in all_features if feat not in remove_features]

#Bringing 'poi' to first position in features_list
features_list.insert(0, features_list.pop(features_list.index('poi')))



        
print "***********************************************************************************************************"

#Remove THE TRAVEL AGENCY IN THE PARK
df=df.drop("THE TRAVEL AGENCY IN THE PARK")
#outlier Detection
sns.jointplot(y="total_payments",x='salary',data=df)
plt.show()
#print people with huge total payments

print "People with huge toatl payments(outlier detection):",df[df.total_payments > 1e8].index
print "***********************************************************************************************************"
#Drop TOTAL
df = df.drop('TOTAL')

#Creating new feature
df=df.apply(pd.to_numeric,errors='ignore',downcast='integer')
df['percentage_msg_to_poi'] =(df.from_this_person_to_poi/df.from_messages*100)
df['percentage_msg_from_poi'] =(df.from_poi_to_this_person/df.to_messages*100)
df=df.round()

#Add newly created features to features_list
features_list.append('percentage_msg_to_poi')
features_list.append('percentage_msg_from_poi')


#convert back Pnandas DataFrame to dict
df = df.replace(np.nan, 'NaN') 
data_dict = df.to_dict(orient = 'index')
my_dataset = data_dict

        
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Compute Feature Scores
selector =SelectKBest(f_classif,k='all')
selector.fit(features,labels)
scores = -np.log10(selector.pvalues_)  #get raw scores
print "Feature Score are:",scores
plt.bar(range(len(features_list)-1),scores)               #plot a bar graph of features and scores
plt.show()

#Final list of top 9 features
features_list =['poi','salary','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','total_stock_value','expenses',
                'percentage_msg_to_poi']
print "Final Feature List is:",features_list


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42,)
#GaussianNB
clf = GaussianNB()
test_classifier(clf,my_dataset,features_list)


print " Training using AdaBoost Classifer, it may take a while......"
#AdaBoost
AdaBoost=AdaBoostClassifier()
parameters={'n_estimators':[10,20,30,40,50,60,70,80,90,100],'learning_rate':[x/20.0 for x in range(1,31)]}
clf = GridSearchCV(AdaBoost,parameters,cv=10)
clf.fit(features,labels)
clf=clf.best_estimator_
test_classifier(clf,my_dataset,features_list)

#DecisionTree
DecisionTree = DecisionTreeClassifier(random_state=1)
parameters ={'criterion':['gini','entropy'],'min_samples_split':list(range(2,11)),'max_features':[9,'auto','sqrt','log2']}
clf =GridSearchCV(DecisionTree,parameters,cv=10)
clf.fit(features,labels)
clf=clf.best_estimator_
test_classifier(clf,my_dataset,features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
