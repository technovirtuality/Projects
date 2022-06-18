# Importing necessary packages 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Importing the following libraries for preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 

# Importing performance metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
import statsmodels.api as sm 

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Classification SVM Models
from sklearn.svm import LinearSVC, SVC


# Hyperparameter Tuning
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)

# For custom tuning functions for hyperparameter tuning
from sklearn.metrics import make_scorer

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


import os

os.chdir('E:\Anukriti\Advance Data Science-IIT Madras\Assignment#9')

bank_data = pd.read_csv('bank_marketing.csv', index_col=0) 
bank = bank_data.copy()

bank.info()

bank.isnull().sum()
# No missing values


# Checking special characters
categorical_data = bank.select_dtypes(include = ['object']).copy()

# There are no special characters present in columns with object data types
frequencies = categorical_data.apply(lambda x: x.value_counts()).T.stack()
print(frequencies)
# There are lot of unknown values in poutcome and contact columns


# Removing duplicates, if present
bank.drop_duplicates(keep = 'first', inplace = True)


# Summary 
summary = bank.describe()


# Correlation among numerical variables

corr = bank.corr()

plt.figure(figsize = (10,6))
sns.heatmap(data= corr, annot = True)
plt.show()
# Correlation among numerical variables is weak. We could not find any correlation. There is no correlation among
# independent variables


plt.plot(figsize = (15,15))
sns.pairplot(data = bank, hue = 'deposit')
plt.show()
# There are no clusters in data
# More duration of call leads to more subscriptions
# 


#----------------------EDA------------------------------------------------------------------

# Age

sns.boxplot(y=bank['age'])
sns.distplot(bank['age'], bins=10)
plt.show()
# Most of the people range between 30 and 50 as area under the curve within this region is more

sns.histplot(x='age', data=bank, hue='deposit', bins=10, kde=False)
# Customers who subscribed for term deposit are either less than 30 or more than 60yrs of age



# Job Type

pd.crosstab(index=bank['job'], columns='count', normalize=True)
# Most of the dataset involves customers from management job, blue collar, administration and technician 

# Job vs deposit(Target variable)
sns.countplot(x='job', hue = 'deposit', data = bank)

# Interpretation: People with Management jobs, retired, students and self employed opted 
#for term deposit as compared to other job types. 



# Education

pd.crosstab(index=bank['job'], columns= bank['education'], normalize=True)


sns.countplot(x= 'education', hue= 'deposit', data = bank)


sns.countplot(x= 'job', hue= 'education', data = bank)
# Interpretation: People get job type according to their education. Tertiary educated people get 
# high skilled jobs(management) whereas primary educated people get low shilled jobs(technicians, blue collar, services, unemployed)
# We can chose one variable among job and education. $$$$$$$$$$$$$$$$$$$$$$$$$$


# Marital Status

sns.countplot(x='marital', hue = 'deposit', data = bank)
# More Single customers opt for term deposit as compared to married or divorced  


# Defaulters

sns.countplot(x='default', hue = 'deposit', data = bank)
bank['default'].value_counts()
# INterpretation: The dataset contains most of non-defaulters. Only 1.53% of customers are defaulters. 
# There is almost no variation in this feature. Therefore, removing this variable. ######################


# Housing loan

sns.countplot(x='housing', hue = 'deposit', data = bank)
bank['housing'].value_counts()
# More customers without loan opted for term deposit in comparison to those with loan.


# Personal Loan
sns.countplot(x='loan', hue = 'deposit', data = bank)
bank['loan'].value_counts()
# There is not much effect on decision for deposit due to personal loan.
# Also, The dataset contains only 12% of customers with loan. There is not much variability in the variable.
# Therefore, removing 'loan' variable


# Yearly Balance

sns.boxplot(y=bank['balance'])
bank['balance'].describe()
# Interpretation: Mean is very high as compared to median. So, balance is a right skewed distribution.
# Therefore replacing the outliers with median. Keeping the threshold range from 0 to 20,000
bank['balance'] = np.where((bank['balance'] > 20000) | (bank['balance'] < 0), bank['balance'].median(), bank['balance'])

bank['balance'].describe()  

sns.histplot(x = bank['balance'], hue = bank['deposit'], bins = 10, kde = False)
# Customers who subscribed for term deposit had more balance as compared to those who did not subscribed.


# contact

sns.countplot(x='contact', hue = 'deposit', data = bank)
# Interpretation: Data contains those customers who are mostly contacted by cellular. Among all contact types 
# only those customers who got contacted by cellular(more of them) went for term deposit. 
# We will consider this variable.



# days

sns.countplot(x = 'day', hue = 'deposit', data = bank )

sns.distplot(bank['day'], bins=10)
plt.show()
# Could not find much effect of contact day on target variable 'deposit'. So, not considering this variable.


# Month

sns.countplot(x = 'month', hue = 'deposit', data = bank )


# Duration

sns.barplot(x = 'deposit', y='duration', data = bank)
# customers who went ahead with term deposit had more time duration in last call.
# The last call duration is less for those who rejected term deposit.


# pdays

bank['pdays'].value_counts()
# There are 4133 values of -1. -1 means customer was not contacted previously. There are 74% (4133/5581*100)
# values of -1. So we will not consider this variable.


# campaign

sns.boxplot(y = bank['campaign'])
bank.campaign.describe() # Mean and median has almost same value. So it is normal distributiion. Imputing outliers with
# mean values.
sns.countplot(x = bank['campaign'])
bank['campaign'].value_counts()
len(bank[bank['campaign'] > 6])/len(bank) * 100 # Percentage of customers contacted more than 6 times is 5.1%
# So we will impute values more than 6 with mean.

# Imputation:
bank['campaign'] = np.where(bank['campaign'] > 6, bank['campaign'].mean(), bank['campaign'])
sns.boxplot(y = bank['campaign'])
bank.campaign.describe()

sns.histplot(x = bank['campaign'], hue = bank['deposit'], bins = 10, kde = False)
# Customers who subscribed for term deposit had fewer number of contacts during the camaign.



# Previous Number of Contacts
sns.boxplot(y = bank['previous'])
bank.previous.value_counts()
sns.countplot(x = bank['previous'])
bank.previous.describe() # As mean and median are almost equal. So imputing outliers with mean
len(bank[bank['previous'] > 5])/len(bank) * 100 # Percentage of customers contacted more than 6 times is 3.7%
# So we will impute values more than 5 with mean.

# Imputation:
bank['previous'] = np.where(bank['previous'] > 6, bank['previous'].mean(), bank['previous'])
sns.boxplot(y = bank['previous'])
bank.previous.describe()



# Previous Outcome

bank.poutcome.value_counts()
# Approx. 74% of values are unknown. So we cannot consider this variable.


# Target variable deposit: 
bank['deposit'].value_counts()
ratio = bank[bank['deposit'] == 'yes'].shape[0] / bank.shape[0]
print(f"Ratio of term deposit subscribers in dataset = {round(ratio,4)}")
# The dataset is slightly imbalanced but we are considering it as balanced dataset. Target variable has almost equal count for both values of term deposit.
# So we can use accuracy for evaluating the model

#___________________________________________________________________________________________


X = bank.drop(columns = ['default','loan','day','pdays','poutcome','deposit'], axis = 1)
Y = bank['deposit']

# Encoding categorical variables numerically
X = pd.get_dummies(X, drop_first=True)

# Standardization of variables before applying models as all numerical variables are on different scale
scaler = StandardScaler() 
encoder = LabelEncoder()
X = scaler.fit_transform(X)
Y = encoder.fit_transform(Y)

# We need to decide which machine learning model to chose for solving the problem that we are getting different accuracy scores
# for different random_state parameter value. To solve this we should use K-Fold Cross-Validation. But K-Fold Cross Validation
# the training dataset may become imbalanced.  
# We chose Stratified cross validation because it maintains the same ratio of datapoints of each class of original data
# in train test after splitting. We are considering the dataset as balanced, that's why not applying upsampling or downsampling before applying Stratified Kfold cross validation  
# We will check the accuracies of each model using 3 splits
# We will consider the average of accuracies from each split for every model. 


# Function for obtaining scores from each model. Calling get_score for each model will return accuracy for that model 
def get_score(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


# IMporting Stratified K Fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 3) # Chosing number of splits = 3


# Creating empty list for appending accuracy scores from each type of model
scores_l = []                    # Logistic Regression
scores_lda = []                  # Linear Discriminant Analysis
scores_rf = []                   # Random Forest
scores_svm = []                  # Support Vector Machine
scores_knn = []                  # K Nearest Neighbour


# Appending accuracy scores for every model in their respective empty list of scores. Also applying Stratified K fold
# to data
for train_index, test_index in skf.split(X,Y):
    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    scores_l.append(get_score(LogisticRegression(), X_train, X_test, Y_train, Y_test))
    scores_lda.append(get_score(LinearDiscriminantAnalysis(), X_train, X_test, Y_train, Y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators = 200), X_train, X_test, Y_train, Y_test))
    scores_svm.append(get_score(SVC(), X_train, X_test, Y_train, Y_test))
    scores_knn.append(get_score(KNeighborsClassifier(n_neighbors = 3), X_train, X_test, Y_train, Y_test))


# Finding average of accuracy obtained from 3 splits

# Logistic Regression
scores_l
total = sum(scores_l)
print(total*100/3)


# Linear Discriminant Analysis
scores_lda
print(scores_lda)
print(sum(scores_lda)*100/3)


# Random Forest
scores_rf
print(sum(scores_rf)*100/3)


# Support Vector Machine
scores_svm
print(sum(scores_svm)*100/3)


# K Nearest Neighbours
scores_knn
print(sum(scores_knn)*100/3)

# We get highest accuracy of 83.2467% from random forest model.
# Still we will try other models to consider other parameters of models eg. no. of misclassified samples, minimum no. of
# false negatives because if subscriber is falsely predicted as non-subscriber then company can loose the customer.
# So, we need to select the model with classifier which reduce false negative values. We need to check for Recall value.
# F1 score is weighted average of precision and recall. More useful if we have unequal class distribution in data.
# here we considered class distribution as balanced so not looking F1 score for decision.

#________________________________________________________________________________________________________________


# Evaluation function

def evaluate_classification_model(Y_true, Y_pred):
    """
    Custom classification model evaluation function
    """

    # Confusion_matrix
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    sns.heatmap(data=conf_matrix, annot=True, fmt='g')
    plt.title("Confusion Matrix")
    plt.show()
    print(f"Number of missclassified samples: {(Y_test != Y_pred).sum()}")
    
    
    # Accuracy
    accuracy = accuracy_score(Y_true, Y_pred)
    print(f"Accuracy of the model: {accuracy * 100:.2f}%")
    
    # Precision
    precision = precision_score(Y_true, Y_pred)
    print(f"Precision of the model: {precision:.2f}")
    
    # Recall
    recall = recall_score(Y_true, Y_pred)
    print(f"Recall of the model: {recall:.2f}")
    
    # F1-Score
    f1score = f1_score(Y_true, Y_pred)
    print(f"F1 score of the model: {f1score:.2f}")
    
    # AUC
    auc = roc_auc_score(Y_true, Y_pred)
    print(f"Area under the ROC Curve: {auc:.2f}")


# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)


# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
evaluate_classification_model(Y_test, Y_pred)
# Number of missclassified samples: 279; No. of false negative samples = 154
# Accuracy of the model: 83.34%
# Precision of the model: 0.83
# Recall of the model: 0.80 Good recall value
# F1 score of the model: 0.82
# Area under the ROC Curve: 0.83

# Largest and Smallest coefficient
lr.coef_
lr.intercept_
np.argmax(lr.coef_, axis = 1) # largest coefficient index
X.columns[np.argmax(lr.coef_, axis=1)] # Column name with highest coefficient


np.argmin(lr.coef_, axis = 1) # smallest coefficient index
X.columns[np.argmin(lr.coef_, axis=1)] # Column name with lowest coefficient

#---------------------------------ROC-AUC Curve-----------------------------------------------------------


lr.predict_proba(X_test) # The probability of data points belonging to class 0 or 1



# Creating a random classifier. Random classifier randomly assigns any class value(0 and 1) to the prediction without considering independent features.
# So for all features we are creating a list. On that list everything belongs to category '1'. This is random prediction. This is our random classifier.
random_prediction = [1 for _ in range(len(Y_test))]
logistic_regression_prediction = lr.predict_proba(X_test)



# Logistic regression by default considers threshold of 0.5. roc_curve() will give false positive rate
# true poisitive values for various thresholds i.e 0.2, 0.3,....  
random_predictions_fpr, random_predictions_tpr, _ = roc_curve(Y_test, random_prediction)
logistic_regression_fpr, logistic_regression_tpr, logistic_regression_thresholds = roc_curve(Y_test, logistic_regression_prediction[:, 1])


logistic_regression_thresholds

# We will look for all thresholds. 

# We are creating a list accuracy_list and accuracy scores to it.
accuracy_list = []

for threshold in logistic_regression_thresholds:
    y_pred_thresh = np.where(logistic_regression_prediction[:, 1] > threshold, 1, 0) # We will look for all thresholds. If prediction probability is greater than threshold then function will assign class 1 else class 0. 
    
    accuracy_list.append(accuracy_score(Y_test, y_pred_thresh))

accuracy_list# The accuracy is low initially but gradually increases and then starts decreasing.


# Creating a dataframe by concatenating the threshold values and accuracy in it.
accuracy_thresh_df = pd.concat([pd.Series(logistic_regression_thresholds), pd.Series(accuracy_list)], axis=1)
accuracy_thresh_df.columns=['Thresholds', 'Accuracy'] # Assigning names to columns
accuracy_thresh_df.sort_values(by='Accuracy', ascending=False, inplace=True)

accuracy_thresh_df.head()


# Plotting ROC-AUC Curve

plt.figure(figsize=(10, 10))

# Plotting false positive rate and true positive rate to get plot of random classifier
plt.plot(random_predictions_fpr, random_predictions_tpr, linestyle='--', label="Random Predictions", color='darkblue')
# Plotting logistic regression false positive rate and logistic regression true positive rate to get plot at thresholds level
plt.plot(logistic_regression_fpr, logistic_regression_tpr, marker='.', label="Logistic Regression", color='orange')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - AUC Curve")
plt.show()

# For selecting the model which minimizes false positive rate and maximizes true positive rate, we select the
# threshold value at bend of the curve.



# Area under the curve

random_auc = roc_auc_score(Y_test, random_prediction)
logistic_regression_auc = roc_auc_score(Y_test, logistic_regression_prediction[:, 1])

print(f"Random Prediction - Area under the curve (AUC) = {random_auc}")
print(f"Logistic Regression - Area under the curve (AUC) = {logistic_regression_auc}")

# Our model did well as area under the curve is 89.812%

#____________________________________________________________________________________________________



#------------------------------------Linear Discriminant Analysis-------------------------------------

# Building a linear discriminant analysis model:  
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
Y_predlda = lda.predict(X_test)
evaluate_classification_model(Y_test, Y_predlda)
# Number of missclassified samples: 301; No. of false negative samples=186
#Accuracy of the model: 82.03%
#Precision of the model: 0.84
#Recall of the model: 0.76
#F1 score of the model: 0.80
#Area under the ROC Curve: 0.82

#_____________________________________________________________________________________________________



#-------------------------------------K Nearest Neighbour--------------------------------------------

# KNN
KNN_model = KNeighborsClassifier(n_neighbors = 3)
KNN_model.fit(X_train, Y_train)
Y_pred_KNN = KNN_model.predict(X_test)
evaluate_classification_model(Y_test, Y_pred_KNN)
#Number of missclassified samples: 423
#Accuracy of the model: 74.75% Low


# Elbow method for determining the value of K

misclassified_samples = [] # Creating empty list for misclassified samples


for i in range(1, 20, 2):
    KNN_model = KNeighborsClassifier(n_neighbors = i)
    KNN_model.fit(X_train, Y_train)
    Y_preds_KNN = KNN_model.predict(X_test)
    misclassified_samples.append((Y_test != Y_preds_KNN).sum())
    
print(misclassified_samples) 
# The number of misclassified samples for all values of K is more than number of misclassified samples in logistic regression which is 279.


# Elbow Plot
plt.figure(figsize = (10,6))
plt.plot(range(1,20,2), misclassified_samples, marker = 'o')
plt.xticks(range(1,20,2))
plt.title("Elbow Plot")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Number of misclassified samples")
plt.show() 


# Choosing the value of K = 5
# As the elbow curve is turning at 5 and has minimum vaue at K = 5. Higher K value leads to overfitting and lower values underfitting.


list1 = [5,7,9,11,13]

for i in range(len(list1)):
    print("\n\nEvaluation with K = ", list1[i])
    KNN_model = KNeighborsClassifier(n_neighbors = list1[i])
    KNN_model.fit(X_train, Y_train)
    Y_pred_KNN = KNN_model.predict(X_test)
    evaluate_classification_model(Y_test, Y_pred_KNN)

# Interpretation: Evaluation parameters obtained are best for KNN model with K=5 beacause accuracy is highest
# = 75.34% and no. of misclassified samples = 413 which is minimum. Also false negatives is 245 minimum.
#Number of missclassified samples: 413
#Accuracy of the model: 75.34%
#Precision of the model: 0.76
#Recall of the model: 0.68
#F1 score of the model: 0.72
#Area under the ROC Curve: 0.75

#_______________________________________________________________________________________________________




#-----------------------------------Support Vector Machine--------------------------------------------

# Baseline linear SVC model

linear_svc = LinearSVC(C = 10, max_iter=10000, random_state=0)
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
evaluate_classification_model(Y_test, Y_pred)

#Number of missclassified samples: 285; No. of false negatives = 167
#Accuracy of the model: 82.99%
#Precision of the model: 0.84
#Recall of the model: 0.78
#F1 score of the model: 0.81
#Area under the ROC Curve: 0.83

#--------------------------------Hyper Parameter Tuning------------------------------------------------

# Hyper parameter tuning tries different combinations of parameters 'C' and 'kernel', fit
# and evaluate the model to find out which combination gives highest accuracy. 
# Grid Search performs trial and error method considering all combinations of 'C' and 'kernel'. It takes long time for large dataset.
# whereas Randomized Search randomly generates points within a certain range. We are having large dataset. So we are selecting Random Search
# for hyperparameter tuning.

random_search = RandomizedSearchCV(SVC(random_state=0),{"C":[0.001,0.01,0.1,1,10], "kernel":['rbf','poly','linear']})
random_search.fit(X_train, Y_train)

# Getting the best parameters
random_search.best_params_ # 'kernel': 'linear', 'C': 0.01


# Building SVC model using best hyperparameters
svc_classifier = SVC(kernel = 'linear', C=0.01, random_state=0)
svc_classifier.fit(X_train, Y_train)
Y_pred_h = linear_svc.predict(X_test)  # Predictions
evaluate_classification_model(Y_test, Y_pred_h)   # Evaluating the model

#Number of missclassified samples: 285; No. of false negatives: 167
#Accuracy of the model: 82.99%
#Precision of the model: 0.84
#Recall of the model: 0.78
#F1 score of the model: 0.81
#Area under the ROC Curve: 0.83


#______________________________________________________________________________________________________




#------------------------------------Decision Tree with all variables-------------------------------------------------


X_dta = bank.drop(columns = ['deposit'], axis = 1)
X_dta = pd.get_dummies(X_dta, drop_first=True)
Y_dta = bank['deposit'].to_numpy()
print(type(Y_dta))


features_a = list(X_dta.columns)
target_a = ['deposit']


# Splitting the data

train_xdta, test_xdta, train_ydta, test_ydta = train_test_split( X_dta, Y_dta, test_size=0.3,random_state=1)


# Model Building

model_dta = tree.DecisionTreeClassifier()
model_dta = model_dta.fit(train_xdta,train_ydta)


# Decision Tree visualization

import pydot

tree.export_graphviz(model_dta, out_file='tree_dta.dot',  
            feature_names=X_dta.columns,  
              #class_names=Y_dt,  
                     filled=True, rounded=True,  
                     special_characters=True)

(graph,) = pydot.graph_from_dot_file('tree_dta.dot')
graph.write_png('decision_tree_dta.png')




y_pred_dta = model_dta.predict(test_xdta)


# Confusion Matrix and Accuracy
print(confusion_matrix(test_ydta, y_pred_dta))
print(accuracy_score(test_ydta, y_pred_dta))

# When all variables are included: Accuracy increased to 77.134%.
# No. of misclassified samples reduced to 210+173 = 383 samples
# Model seems to be overfitting as most of leaf nodes have 1 sample.



#-------------------------------Decision Tree with all variables(Reduced Complexity)-------------------------



# Higher depth of tree causes overfitting and lower depth causes underfitting. Choosing max_depth = 5
# min_samples_split = 10

# Creating Decision Tree object
clf1 = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 5, min_samples_split=10, max_leaf_nodes= 5) # By default criterion = "gini"
 
# Training Decision Tree classifier
clf1 = clf1.fit(train_xdta, train_ydta)

# Prediction
y_pred_dtanew = clf1.predict(test_xdta)



# Decision Tree visualization

tree.export_graphviz(clf1, out_file='tree_dtanew.dot',  
            feature_names=X_dta.columns,  
              #class_names=Y_dt,  
                     filled=True, rounded=True,  
                     special_characters=True)

(graph,) = pydot.graph_from_dot_file('tree_dtanew.dot')
graph.write_png('decision_tree_dtanew.png')




# Performance Measures 

print(confusion_matrix(test_ydta, y_pred_dtanew))
print(accuracy_score(test_ydta, y_pred_dtanew))
# Accuracy increased to 80.12% and no. of misclassified samples is reduced to 166 + 167 = 333 samples. 





# Bagging - It reduces overfitting (variance) of Decision Tree by averaging 

from sklearn.ensemble import BaggingClassifier

bc1 = BaggingClassifier(base_estimator= clf1, n_estimators=50, random_state=1)

bc1.fit(train_xdta, train_ydta)

y_pred_bc1 = bc1.predict(test_xdta)

acc_test1 = accuracy_score(test_ydta, y_pred_bc1)
acc_test1

print(confusion_matrix(test_ydta, y_pred_bc1))

# AFter bagging accuracy reduced to 79.34% and no. of misclassified samples has increased to 346.
 
#____________________________________________________________________________________________________________


#--------------------------------------------Random Forest--------------------------------------------------

# Trying with different number of estimators in a random forest
list = [50,150,200,250]

for i in range(len(list)):
    print("\n\nNo. of estimators = ",list[i])
    rf = RandomForestClassifier(n_estimators=list[i], random_state=0)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    evaluate_classification_model(Y_test, Y_pred)

# With number of estimators = 250, accuracy is high 83.33% and no.of misclassified samples is least = 269
#Accuracy of the model: 83.94%
#Precision of the model: 0.81
#Recall of the model: 0.85
#F1 score of the model: 0.83
#Area under the ROC Curve: 0.84

#________________________________________________________________________________________________________________


#----------------------------------------------------PCA---------------------------------------------------------

from sklearn.decomposition import PCA

data_cleaned = bank.drop(columns = ['marital','default','loan','pdays','poutcome'], axis = 1)

#numerical = pd.get_dummies(data_cleaned)
#numerical.drop(columns = ['housing_no','job_unknown','contact_unknown','education_unknown', 'month_jan'])
# Dropping one column of each feature because dummy values of other values if all 0, can tell the presence of remaining column. 



# Technically PCA works well for numerical data, so considering only numerical variables.
# We can't do scaling precisely on categorical features, noise will be present as we can't scale precisely the response on a scale of 1 to 3 or 1 to 5

original_data = data_cleaned.drop(['job', 'education','housing','contact','month','campaign','previous'], axis = 1)
numerical_data = original_data.drop(['deposit'], axis = 1)

y_pca = bank['deposit']

# Standardization
sc = StandardScaler(with_std=False)
input_data = sc.fit_transform(numerical_data)
input_data = pd.DataFrame(input_data, columns = ['Age', 'Balance', 'Day', 'Duration' ]) # Assigning column names to scaled numerical data

input_columns = list(original_data.iloc[:, :4].columns)


# Computing eigenvalues and eigenvectors
# Decomposing using SVD

u,s,v = np.linalg.svd(input_data)
pc = original_data[input_columns].dot(v.T)
pc.columns = ['PC1', 'PC2', 'PC3', 'PC4']

# Adding target variable to score matrix pc
pc['deposit'] = original_data.deposit
pc.head()


exp_var = s**2/np.sum(s**2) * 100
exp_var
# We got 97.67% variance with only 1st PC and 2% variance from  2nd PC. PCs are linear combination of all features.
# We are selecting PC1 and PC2 


# Scatterplot with PC1 and PC2 scores
plt.subplots(figsize = (8,6))
sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'deposit', data = pc)
plt.show()
# The subscription for deposit is linearly separable.

# We did not consider PCA for building models because our data includes many categorical variables also and PCA typically works best with
# numerical variables. 

#_________________________________________________________________________________________________________________


"""Conclusion:

Comparing evaluation parameters of all models considering high accuracy, minimum no. of misclassified samples and
high recall value to reduce the number of false negatives:
    
We concluded that Random Forest model performs best for the given dataset.

"""

































