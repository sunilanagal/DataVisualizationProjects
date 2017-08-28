
# coding: utf-8

# Goal of the project is to identify potential “Person of Interest (POI)”from the available Enron fraud data by using financial and email information which was made public by employing data investigation and machine learning algorithms. POIs are people who were found to have committed fraud by US government during trial. 
# 
# Machine learning algorithms can be trained on data to identify patterns and relationships exhibited by POIs. This learned pattern/relationship knowledge can be very useful in predicting POIs. 

# In[1]:

# Loading Enron data in a data dictionary...

from __future__ import division
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit

sys.path.append("../tools/")
sys.path.append("../final_project/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from tester import load_classifier_and_data

with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Understanding the Dataset and Question:

# In[2]:

# Print Basic information of data dictionary

print("Number of people: %d"%len(data_dict.keys()))
print("Number of features per person: %d"%len(list(data_dict.values())[0]))
print("Number of POI: %d"%sum([1 if x['poi'] else 0 for x in data_dict.values()]))


# In[3]:

# constructing numpy dataframe from data dictionary for analysing data
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)


# In[4]:

# Plotting scatter diagram for Enron data with salary feature vs poi
#%matplotlib inline
plt.scatter(df['poi'], df['salary'])
plt.suptitle("POI/Non-POI salary scatter plot Enron Data")
plt.show()


# In[5]:

# Plotting scatter diagram of bonus and salary 
plt.scatter(df['bonus'], df['salary'])
plt.suptitle("Spread of bonus vs salary for Enron Data with outliers")
plt.show()


# As it is very clear from the scatter plots above that there is an outlier.
# Let's find the outlier: criteria used for outlier detection is - if the difference 
# for a person's feature value from mean value of that feature is higher than 1.96 times it's standard deviation, it is considered outlier.

# In[6]:

# Finding Outliers

newdf = df.copy()

for col in ['salary', 'to_messages', 'deferral_payments', 
            'total_payments', 'exercised_stock_options', 
            'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
            'restricted_stock_deferred', 'total_stock_value', 
            'expenses', 'loan_advances', 'from_messages', 
            'other', 'from_this_person_to_poi', 'director_fees', 
            'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']:
    newdf['Outlier'] = abs(newdf[col] - newdf[col].mean()) > 1.96*newdf[col].std()
#   print newdf[newdf['Outlier'] == True][col]
#    print ""

print newdf.loc['BHATNAGAR SANJAY']
print newdf.loc['FREVERT MARK A']


# The key for the outlier is not a person's name but the total (it was printed and checked, not shown in results as it makes the output very long and not all the information is useful), and it is sum of the column, so removing it for classification.
# 
# Person "BHATNAGAR SANJAY" has unusually high restricted_stock_deferred value, which most likely is an error, so removing him from classification.
# 
# Person "FREVERT MARK A" displays unusual characteristic's in terms of email, which is
# outlier when considered with other people, hence removing it.

# In[7]:

# Removing Outliers from data

newdf = newdf.drop('TOTAL')
newdf = newdf.drop('BHATNAGAR SANJAY')
newdf = newdf.drop('FREVERT MARK A')

print len(newdf)
df = newdf


# In[8]:

# Plotting scatter diagram of bonus and salary after removing outlier
plt.scatter(df['bonus'], df['salary'])
plt.suptitle("Spread of bonus vs salary for Enron Data")
plt.show()


# The plot shows better spread of Enron Data now after removing the outliers. There were other outliers associated with the financial data for some of the POIs (i.e., Kenneth Lay, Jeffrey Skilling, etc.), but these are valid data points for training the machine learning algorithms. These were the top executives of Enron company. Hence retaining them.

# In[9]:

# To check the ranges of data for all features

df.describe()


# In[10]:

# Analyzing data availability for POIs null/not null information and plotting for visualization

criterion = df['poi'].map(lambda x: x == True)
print "Total no. of poi's: ", len(df[criterion])
print "no. of data points with no data for poi's: ", df[criterion].isnull().sum().sum()
print "no. of data points with no data for poi's for each feature: "
print df[criterion].count()


# In[11]:

# Plotting to visualize the distribution of null/not null data points for POIs

data_counts = df[criterion].count()
# create a quick bar chart
data_counts.plot(kind='bar');
plt.suptitle("POI: features with null/not null data points")
plt.show()


# In[12]:

# Analysing non-POI data point null/not null information and plotting for visualization
criterion = df['poi'].map(lambda x: x == False)
print "Total no. of non-poi's: ", len(df[criterion])
print "no. of data points with no data for poi's: ", df[criterion].isnull().sum().sum()
print "no. of data points with no data for poi's for each feature: "
print df[criterion].count()


# In[13]:

#Plotting to see distribution between features of null/not null data points

data_counts = df[criterion].count()
# create a quick bar chart
data_counts.plot(kind='bar');
plt.suptitle("Non-POI: features with null/not null data points")
plt.show()


# Optimize Feature Selection/Engineering:

# For the financial features; loan_advances, restricted_stock_deferred and director_fees, all had not enough POI data coverage. So, including these features could be  mis-classified as pattern of a POI. So, discarding these 3 features.
# 
# All of the email features had sufficient coverage for analysis use. Although email_address doesn't provide any information for classifying as POI or non-POI, I decided to discard it.
# 
# From above 2 scatter plots it's obvious that many features have missing information. The missing information is being filled by the mean values of the feature.

# In[14]:

# Removing features from the dataset that are either not useful for analysing or have
# missing information most of times. 

del df['email_address']
del df['loan_advances']
del df['restricted_stock_deferred']
del df['director_fees']

print df.shape


# In our dataset, we have 143 people and 18 features.

# In[15]:

# Checking the correlation of each feature in POI identification

corr = df.corr()
print('\nCorrelations between features to POI:\n ' +str(corr['poi'].sort_values()))


# In[16]:

# Plotting the most co-related features vs POI to analyse 

#%matplotlib inline
df.hist(column='exercised_stock_options',by='poi',bins=25,sharex=True,sharey=True)
plt.suptitle("exercised_stock_options by POI")


# In[17]:

# Plotting the most co-related features vs POI to analyse 
df.hist(column='total_stock_value',by='poi',bins=25,sharex=True,sharey=True)
plt.suptitle("total_stock_value by POI")


# In[18]:

# Plotting the most co-related features vs POI to analyse 
df.hist(column='bonus',by='poi',bins=25,sharex=True,sharey=True)
plt.suptitle("bonus by POI")


# In[19]:

# Plotting the most co-related features vs POI to analyse 
df.hist(column='salary',by='poi',bins=25,sharex=True,sharey=True)
plt.suptitle("salary by POI")


# Feature selection process is key in Machine Learning problems, the idea behind it is that you want to have the minimum number of features than capture trends and patterns in your data. A good feature sets contain features that are highly correlated with the class, yet uncorrelated with each other.
# 
# Since salary, stock options, bonus, emails from/to person to/from poi would provide a perspective related to their parent feature (eg. fractional salary of a person related to total payments he/she is receiving), new features were created, which are following:
# 
# 1. fraction_salary=salary/total_payments: this new feature can help interpret patterns of fraction of salary with respect to total_payments. If total_payments are way too much in comparison to salary then it's a sign of something suspicious, could be one of the signs of POI.
# 2. fraction_exercised=exercised_stock_options/total_stock_value: this new feature can tell if they are exercised_stock_options are relatively high as compared to total_stock_value, which could be a sign that this person is not interested in holding onto company stocks. Again, this could be a sign of POI.
# 3. fraction_bonus=bonus/total_payments: this new feature can tell what fraction of total_payments is his bonus. If a person is receiving very high bonus as compared to his total_payments, that could be a sign calling for investigation. This might be one of the patterns of POI.
# 4. fraction_from_poi=from_poi_to_this_person/to_messages: this new feature can tell how many emails are being received from POIs in comparison to total incoming emails. If most incoming emails are received from POIs then that's a reason for inquiry. This could be a sign of POI.
# 5. fraction_to_poi=from_this_person_to_poi/from_messages: this new feature can tell how many emails are being sent to POIs in comparison to total outgoing emails. If most outgoing emails are sent to POIs then that's a sign for investigation. It might be a characteristic of POI.
# 6. fraction_shared_receipt_with_poi=shared_receipt_with_poi/to_message: This new feature can recognize the pattern of sharing email information with POI in comparison to the total number of emails being received. 

# In[20]:

# New Feature creation

df['fraction_salary']=df['salary']/df['total_payments']
df['fraction_exercised']=df['exercised_stock_options']/df['total_stock_value']
df['fraction_bonus']=df['bonus']/df['total_payments']
df['fraction_from_poi']=df['from_poi_to_this_person']/df['to_messages']
df['fraction_to_poi']=df['from_this_person_to_poi']/df['from_messages']
df['fraction_shared_receipt_with_poi']=df['shared_receipt_with_poi']/df['to_messages']

#Filling NaNs with mean

df = df.fillna(df.mean())

df.shape

# Converting Dataframe to Data dictionay to be used in Udacity provided modules later
Enron_Data = df.to_dict('index')

# storing the list of poi information to be used in Udacity provided modules later
poi = df['poi']

# Removing features poi and Outlier from the dataframe for feature selection in  SelectKBest
del df['poi']
del df['Outlier']


# In[21]:

# Adding new features to the feature list and splitting them into features and labels for classification

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
           'exercised_stock_options', 'bonus', 'restricted_stock',
           'shared_receipt_with_poi', 'total_stock_value', 'expenses',
           'from_messages', 'other', 'from_this_person_to_poi',
           'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',
           'fraction_salary', 'fraction_exercised', 'fraction_bonus',
           'fraction_from_poi', 'fraction_to_poi',
           'fraction_shared_receipt_with_poi']
dataset = featureFormat(Enron_Data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(dataset)


# Scaling before applying classifiers is very important in classifiers like SVM. The main advantage of scaling is to avoid attributes in greater numeric ranges dominating those in smaller numeric ranges. In Enron data, financial data ranges are 0 to 1000,000,000, whereas the email data ranges between 50 - 20000. Another advantage is to avoid numerical difficulties during the calculation. Because kernel values usually depend on the inner products of feature vectors, e.g. the linear kernel and the polynomial kernel, large attribute values might cause numerical problems. I used linear scaling each attribute to the range [0, 1].
# I used scaling on the complete data here. If scaling is done on training data then it's important to remember to use the same method to scale testing data. For example, suppose that we scaled the first attribute of training data from [0, 10000] to [0, +1]. If the first attribute of testing data lies in the range [0, 8000], we must scale the testing data to [0, +0.8].
# Hence, MinMaxScaler functionality from sklearn was used to scale Enron data for all classifiers, although DecisionTree, GuassianNB don't call for it, it's good practice to scale for keeping the data robust for all types of classifiers.

# Validation is the process of testing a machine learning algorithm in order to assess its performance, and to prevent overfitting. Overfitting can occur when the algorithm is fit too closely to the training data, in which case it performs really well on the training data, but poorly on any other new unseen data. This is why it is important to always set aside data for testing from training data. Since machine learning is often used to try to form predictions about new data, a proper validation strategy is critical.
# 
# Aside from not actually perform validation by splitting data into testing and training sets, another classic mistake is to not shuffle and split the testing and training sets adequately. If there are patterns in the way classes are represented or inserted into the main dataset (i.e., sequentially by class), this could lead to an uneven mix of data across classes in the training sets. As a result, the algorithm could wind up being training mainly on data from one class, but tested mainly on data from another, leading to poor performance.
# 
# For this effort, all of the testing and training datasets were created using StratifiedShuffleSplit in Scikit-Learn, which shuffled the data and split it into 1000 different sets, called folds. This was already implemented in the testing function provided by Udacity, but was also used in the GridSearchCV function in order to best identify the optimal combination of parameters.
# 
# Since Enron data set is small it is important to use StratifiedShuffleSplit instead of a simpler cross-validation method such as TrainTestSplit. StratifiedShuffleSplit will make randomly chosen training and test sets multiple times and average the results over all the tests.
# 
# The data is unbalanced with many more non-POIs than POIs. StratifidShuffleSplit also makes sure that the ratio of non-POI:POI is the same in the training and test sets as it was in the larger data set.

# In[22]:

# Shuffling and spilting the data into features and labels for classification and predictions
from sklearn.preprocessing import MinMaxScaler

y = labels
X = features

# Scaling data for classification by all types of classifiers (independent of their sensitivity to scale of features)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

## Try to use stratifieldshufflesplit to find the best subset of training and testing data to use

scv = StratifiedShuffleSplit(labels, 1000, random_state = 42)


# In[23]:

# function to shuffle and split data for training and testing
def train_test(labels):
    cv = StratifiedShuffleSplit(labels, n_iter=25, test_size=0.5, random_state = 42)

    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append(features[ii] )
            labels_train.append(labels[ii] )
        for jj in test_idx:
            features_test.append(features[jj] )
            labels_test.append(labels[jj] ) 
    return features_train, features_test, labels_train, labels_test


# Dataset we normally have can have many no. of feature information, but based on our goal for developing a predictive model there could be many features that don't have any influence or co-relation with what we are trying to predict. Hence, it's important to select the best features for developing predictive model. 
# 
# There are several methods in sklearn for selecting best features to train a classfier. Based on Enron dataset, which is composed of scalar components, I have picked Univariate feature selection method called SelectKBest. It works by selecting the best features based on univariate statistical tests. 

# In[24]:

# Function to SelectKBest features for a classifier based to accuracy, precision and recall
def SelectKBest_for_Classifier(clf):
    RF_acc = []
    RF_precision = []
    RF_recall = []    
    for i in range(len(features[0])):
        t0 = time()
        selector = SelectKBest(f_classif, k = i+1)
        selector.fit(features, labels)
        reduced_features = selector.fit_transform(features, labels)
        cutoff = np.sort(selector.scores_)[::-1][i]
        selected_features_list = [f for j, f in enumerate(features_list[1:]) if selector.scores_[j] >= cutoff]
        selected_features_list = ['poi'] + selected_features_list
        RF = clf
        acc, precision, recall = cvClassifier(RF, reduced_features, labels, scv)
        RF_acc.append(acc)
        RF_precision.append(precision)
        RF_recall.append(recall)
        print "fitting time for k = {0}: {1}".format(i+1, round(time()-t0, 3))
        print "Classifier accuracy: {0}  precision: {1}  recall: {2}".format(RF_acc[-1], RF_precision[-1], RF_recall[-1])
    rfdf = pd.DataFrame({'accuracy': RF_acc, 'precision': RF_precision, 'recall': RF_recall})                  
    rfdf.plot()
    plt.suptitle("Classifier accuracy, precision, recall vs no. of selectKBest features")
    plt.show()
    k_val=RF_recall.index(max(RF_recall))+1
    selector = SelectKBest(f_classif, k=k_val)
    selector.fit(features, labels)
    cutoff = np.sort(selector.scores_)[::-1][k_val]
    selected_features_list = [f for i, f in enumerate(features_list[1:]) if selector.scores_[i] > cutoff]
    selected_features_list = ['poi'] + selected_features_list
    selected_features = selector.fit_transform(features, labels)
    return k_val, selector, selected_features_list, selected_features


# While there are many different ways to measure the performance of a machine learning algorithm, the three metrics used in this effort were precision, recall, and the F1-score. The unbalanced data is also why we use precision and recall instead of accuracy as our evaluation metric.
# 
# These metrics are based on comparing the predicted values (POI or non-POI, in this case) to the actual ones. Precision is calculated by dividing the number of times the algorithm positively identifies a data point (known as a true positive) divided by the total number of positive identifications (regardless of whether they were correct). For this case, a high precision value means POIs identified by the algorithm tended to be correct, while a low value means there were more false alarms, where non-POIs were flagged as POIs.
# 
# Recall is calculated by dividing the number of true positives by the sum of the true positives and the number of times the algorithm incorrectly negatively identified a data point (known as a false negative). In this case, a false negative would be represented by incorrectly labeling a POI as a non-POI. Here, a high recall value means that if there were POIs in the test set, the algorithm would do a good job of identifying them, while a low value means that sometimes POIs slipped through the cracks and were not identified.
# 
# Finally, the F1 score is a weighted average of precision and recall. It’s calculated by multiplying the product of the recall and precision by two, and then dividing by the sum of the precision and recall. For this effort, both precision and recall values of 0.3 had to be achieved using one of the algorithms. The performance of all of the DecisionTree algorithm was tested using the test_classifier function included in the project.

# In[25]:

### function to split train/test data and evaluate classifiers using cross-validation

def cvClassifier(clf, features, labels, cv):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = round(1.0*(true_positives + true_negatives)/total_predictions,2)
    precision = round(1.0*true_positives/(true_positives+false_positives),2)
    recall = round(1.0*true_positives/(true_positives+false_negatives),2)

    return accuracy, precision, recall


# In[26]:

# function to shuffle and split data after passing selected features list
def selected_features_train_test(selected_features_list): 
    dataset = featureFormat(Enron_Data, selected_features_list, sort_keys = True)
    labels, selected_features = targetFeatureSplit(dataset)

    features_train, features_test, labels_train, labels_test=train_test(labels)
    return features_train, features_test, labels_train, labels_test


# To find the best features selection using SelectKBest; I have written an iterator that will iterate the value of K from 0 (0 being the first feature) to total no of features from dataset(21 being the last feature), the features selected in this process are passed into classifier, fitted, then evaluated for precision, recall, f1, f2 scores. 
# 
# Since, in most of the predictive models we would prefer a model with least false negatives result over least false positives (recall = true_positives/(true_positives+false_negatives)), recall method is used for picking the best performing score with selected features.
# 
# The no. of features selected by SelectKBest method differ with each type of Classifier. Hence, I will run SelectKBest to find best no. of features for each type of Classifier against best recall score. The results are plotted for easy visualization.

# Pick and Tune an Algorithm
# 
# I will be trying supervised algorithms: DecisionTree, Kneighbors, GuassianNB and SVM(with and without PCA) to cross-validate against precision and recall scores. The highest scoring algorithm will be the winner algorithm.
# 
# It is important to tune the parameters of algorithm where-ever possible. For this GridSearchCV module from sklearn was utilized for tuning parameters for DecisionTreeClassifier, KNeighborsClassifier, and SVM. GuassianNB doesn't have parameters to tune.
# 
# Grid search, true to its name, picks out a grid of hyperparameter values, evaluates every one of them, and returns the winner. For example, if the hyperparameter is the number of leaves in a decision tree, then the grid could be 10, 20, 30, …, 100. For regularization parameters, it’s common to use exponential scale: 1e-5, 1e-4, 1e-3, … 1. Some guess work is necessary to specify the minimum and maximum values. So sometimes people run a small grid, see if the optimum lies at either end point, and then expand the grid in that direction. This is called manual grid search. Best parameters for DecisionTreeClassifier, KNeighnorsClassifier, and SVM were selected by using this technique.

# In[27]:

import warnings
warnings.filterwarnings("ignore")

from time import time
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

print "DecisionTreeClassifiers accuracy, precision and recall plotted for the SelectKbest features"
DT = DecisionTreeClassifier()
dt_val, selector, selected_features_list, selected_features = SelectKBest_for_Classifier(DT)

print "No. of features selected: %i" %dt_val
DT.fit(selected_features, labels)
print "the feature importance for these selected features are: "
print DT.feature_importances_
print "the feature scores for these selected features from SelectKBest are: "
for f in selected_features_list[1:]:
	print f, "score is: ", selector.scores_[features_list[1:].index(f)]

features_train, features_test, labels_train, labels_test = selected_features_train_test(selected_features_list)
    
print "Tuning DecisionTree Classifier for improving precision, recall and f1 scores…"

from sklearn import tree
from sklearn import svm, grid_search
from sklearn.tree import DecisionTreeClassifier
parameters = {'class_weight': ['balanced', None], 'min_samples_split': [2, 4, 10], 'criterion': ['gini', 'entropy'],
              'max_depth':[1,2,3], 'max_features':['sqrt','log2','auto'],'splitter':['random', 'best'], 'random_state':[1,2,3]}

clf = grid_search.GridSearchCV(DT, parameters, scoring = 'recall')

clf.fit(features_train, labels_train) 
clf = clf.best_estimator_
best_clf = clf
selected_features_list_forBest_clf = selected_features_list

print "Tuned best performing DecisionTree Classifier: ", clf
test_classifier(clf, Enron_Data, selected_features_list, folds = 1000)


# Best performing DecisionTreeClassifier was found to use 8 features which after tuning has:
# accuracy: 0.77, Precision: 0.31, Recall:0.64, f1: 0.42, f2: 0.53

# For SVC, due to unbalanced dataset, assigned a class_weight of 8,1 before selecting k best features.

# In[28]:

from sklearn.svm import SVC

print "SVM Classifiers accuracy, precision and recall plotted for the SelectKbest features"
SV = SVC(class_weight={True: 8, False: 1})

sv_val, selector, selected_features_list, selected_features = SelectKBest_for_Classifier(SV)
print "No. of features selected: %i" %sv_val
SV.fit(selected_features, labels)

print "the feature scores for these selected features from SelectKBest are: "
for f in selected_features_list[1:]:
	print f, "score is: ", selector.scores_[features_list[1:].index(f)]

features_train, features_test, labels_train, labels_test = selected_features_train_test(selected_features_list)
    


# For reducing dimensionality of Enron Dataset for SVM classifier, PCA was employed.
# 
# PCA can be used to extract important features from a data set. These extracted features are low dimensional in nature. These features a.k.a components are a resultant of normalized linear combination of original predictor variables. These components aim to capture as much information as possible with high explained variance. The first component has the highest variance followed by second, third and so on. The components must be uncorrelated. Normalizing data becomes extremely important when the predictors are measured in different units. PCA works best on data set having 3 or higher dimensions (like Enron data, which has several features). Because, with higher dimensions, it becomes increasingly difficult to make interpretations from the resultant cloud of data. PCA is applied on a data set with numeric variables.

# In[29]:

print "Training SVM with PCA"
print "Warning: this is slow... it can take several minutes..."

# Training SVM 

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

clf_params= dict(pca__n_components=[2, 5, 10, 15],
                 pca__whiten=[True],
                 svm__C=[1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
                 svm__gamma=[0.0],
                 svm__kernel=['rbf', 'linear'],
                 svm__tol=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
                 svm__class_weight=[{True: 12, False: 1},
                                    {True: 10, False: 1},
                                    {True: 8, False: 1},
                                    {True: 15, False: 1},
                                    {True: 4, False: 1}])

#Pipeline PCA(for reducing high dimentional data) and SVM
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('svm', SVC())])

# Tuning and fitting SVM 
clf = GridSearchCV(pipe, param_grid = clf_params, scoring = 'f1')

features_train, features_test, labels_train, labels_test = selected_features_train_test(selected_features_list)

features_train=pca.fit_transform(features_train)
clf.fit(features_train, labels_train)

clf = clf.best_estimator_
print "Tuned SVM with PCA classifier: "
print clf


# In[30]:

# function to predict and calculate accuracy, precision and recall for support vector 
# machine's pca reduced features test data

def predict_svc():
    print("Predicting on the test set")

    predictions = clf.predict(features_test)
    print predictions

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    if true_positives == 0:
        print "Classifier could not predict any true positives"
    else:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print "accuracy: %.2f" %accuracy,  "precision: %.2f" %precision, "recall: %.2f" %recall
        print "f1: %.2f" %f1,"f2: %.2f" %f2
    print "total_predictions: %i" %total_predictions, "true_positives: %i" %true_positives, 
    print "false_positives: %i" %false_positives, "false_negatives: %i" %false_negatives, "true_negatives: %i" %true_negatives
    print ""


# In[31]:

predict_svc()


# In[32]:

print "Training SVM without PCA"
print "Warning: this is slow..."

# Training SVM 

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

param_grid= [{'C':[1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
              'gamma':[0.0],
              'kernel':['rbf', 'linear'],
              'tol':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
              'class_weight':[{True: 12, False: 1},
                            {True: 10, False: 1},
                            {True: 8, False: 1},
                            {True: 15, False: 1},
                            {True: 4, False: 1}]}]


# Tuning and fitting SVM 
clf = GridSearchCV(SVC(C=1), param_grid, scoring = 'f1')

features_train, features_test, labels_train, labels_test = selected_features_train_test(selected_features_list)

clf.fit(features_train, labels_train)

clf = clf.best_estimator_
print "Tuned SVM without PCA classifier: "
print clf
predict_svc()


# Best performing SVC without PCA was found to use 21 features, which after tuning has:
# accuracy: 0.12, Precision: 0.12, Recall:1.00, f1: 0.22, f2: 0.42

# In[33]:

from sklearn.neighbors import KNeighborsClassifier

print "KNeighbors accuracy, precision and recall plotted for the SelectKbest features"

KNN=KNeighborsClassifier()
knn_val, selector, selected_features_list, selected_features = SelectKBest_for_Classifier(KNN)

print "No. of features selected: %i" %knn_val
KNN.fit(selected_features, labels)
print "the feature scores for these selected features from SelectKBest are: "
for f in selected_features_list[1:]:
	print f, "score is: ", selector.scores_[features_list[1:].index(f)]

features_train, features_test, labels_train, labels_test = selected_features_train_test(selected_features_list)

clf = KNeighborsClassifier()
clf.fit(selected_features, labels)

print "the feature scores for these selected features from SelectKBest are: "
for f in selected_features_list[1:]:
	print f, "score is: ", selector.scores_[features_list[1:].index(f)]

# Tuning KNeighborsClassifier for improving precision, recall and f1 scores…

parameters = {'n_neighbors': [4,5,6], 'weights': ['uniform', 'distance']} 
knn = KNeighborsClassifier()
clf = grid_search.GridSearchCV(knn, parameters, cv=scv, scoring = 'recall')
clf.fit(selected_features, labels)
clf = clf.best_estimator_
print "Tuned KNeighborsClassifier classifier: "
print clf    
test_classifier(clf, Enron_Data, selected_features_list, folds = 1000)


# Best performing KNeighborsClassifier was found to use 2 features, which after tuning has:
# accuracy: 0.86, Precision: 0.42, Recall:0.22, f1: 0.29, f2: 0.24

# In[34]:

from sklearn.naive_bayes import GaussianNB

print "GaussianNBs accuracy, precision and recall plotted for the SelectKbest features"

GNB=GaussianNB()

gnb_val, selector, selected_features_list, selected_features = SelectKBest_for_Classifier(GNB)
clf = GaussianNB()
#clf.fit(selected_features, labels)
print "No. of features selected: %i" %gnb_val
print "the feature scores for these selected features from SelectKBest are: "
for f in selected_features_list[1:]:
	print f, "score is: ", selector.scores_[features_list[1:].index(f)]

features_train, features_test, labels_train, labels_test = selected_features_train_test(selected_features_list)

print "the feature scores for these selected features from SelectKBest are: "
for f in selected_features_list[1:]:
	print f, "score is: ", selector.scores_[features_list[1:].index(f)]

test_classifier(clf, Enron_Data, selected_features_list, folds = 1000)


# Best performing GuassianNB Classifier was found to use 6 features, which after tuning has:
# accuracy: 0.86, Precision: 0.49, Recall:0.34, f1: 0.40, f2: 0.36

# Best performing DecisionTreeClassifier was found to use 8 features which after tuning has:
# accuracy: 0.77, Precision: 0.31, Recall:0.64, f1: 0.42, f2: 0.53
# 
# Best performing SVC with PCA was found to use 21 features which after tuning has:
# accuracy: 0.15, Precision: 0.13, Recall:1.00, f1: 0.23, f2: 0.42
# 
# Best performing SVC without PCA was found to use 21 features, which after tuning has:
# accuracy: 0.12, Precision: 0.12, Recall:1.00, f1: 0.22, f2: 0.42
# 
# Best performing KNeighborsClassifier was found to use 2 features, which after tuning has:
# accuracy: 0.86, Precision: 0.42, Recall:0.22, f1: 0.29, f2: 0.24
# 
# Best performing GuassianNB Classifier was found to use 6 features, which after tuning has:
# accuracy: 0.86, Precision: 0.49, Recall:0.34, f1: 0.40, f2: 0.36

# Since DecisionTree Classifier performed the best it was decided to dump as the best classifier for Udacity evaluation.

# In[35]:

dump_classifier_and_data(best_clf, Enron_Data, selected_features_list_forBest_clf)


# Conclusion: 
# 
# 1. From the results, it's clear that best performing algorithm is DecisionTreeClassifier for Enron Data.
# 2. SVC with PCA or without PCA performed the same. 
# 3. GuassianNB Classifier although doesn't have parameters to be tuned, it performed very well.
# 
# The Machine Learning process is as follows:
# 
# Problem Definition: Understand and clearly describe the problem that is being solved.
# Analyze Data: Understand the information available that will be used to develop a model.
# Prepare Data: Remove outliers, eliminate bad data, replace missing data in the best possible way. Create new features where it calls for.
# Cross-Validate: Develop robust testing and training data sets. Evaluate with the help of predictions.
# Evaluate Algorithms: by predicting and checking performace
# Improve Results: tune algorithms for performance
# Present Results: Describe the problem and solution so that it can be understood by third parties.
# 
# It can take several iterations to come to the best performing classifier.
