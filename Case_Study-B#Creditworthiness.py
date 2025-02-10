# Importing necessary libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC



# Hyperparameter Tuning
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)


# Importing performance metrics
from sklearn import preprocessing,metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve, plot_confusion_matrix


import os

os.chdir('E:Advance Data Science-IIT Madras\Final Exam')

data_xlsx = pd.read_excel('CreditWorthiness.xlsx',sheet_name = 'Data')
loan = data_xlsx.copy()



loan.info()
loan.isnull().sum() # No missing values



# Checking special characters
categorical_data = loan.select_dtypes(include = ['object']).copy()

# There are no special characters present in columns with object data types
frequencies = categorical_data.apply(lambda x: x.value_counts()).T.stack()
print(frequencies)


# Percentage of frequencies of each level in all categorical variables
frequencies1 = categorical_data.apply(lambda x: x.value_counts(normalize=True)*100).T.stack()
print(frequencies1)


"""
Describing the dataset: Most of the data contains single applicants.
1. Dataset has only 6% clients with checking account balance >= 2000. Majority has no checking account
2. Cdur varies from 4 to 72 months. So data is for short term loan of max. 5 years
3. Camt varies from Rs 2,380 to Rs 1,84120. Seems data is for small amount of loan, thats why it is short-term

"""


# Percentage of frequency of each level in Cpur
loan['Cpur'].value_counts(normalize = True)*100


# Merging levels using mask() function where condition is given. If condition is true then replace the entries
# with 'Other' otherwise if condition is False keep the original value. The condition is:eg. loan['Cpur'].map(loan['Cpur'].value_counts(normalize=True)*100 < 5   
# Merging all levels with less than 5% of value counts into 'other' category for 'Cpur' column
# map() is a series method and gives error if applied to Dataframe
for i in loan['Cpur']:
  loan['Cpur'] = loan['Cpur'].mask(loan['Cpur'].map(loan['Cpur'].value_counts(normalize=True)*100) < 5, 'Other')
  

# Merging all levels with less than 5% of value counts into 'High' category for 'NumCred' column
for i in loan['NumCred']:
  loan['NumCred'] = loan['NumCred'].mask(loan['NumCred'].map(loan['NumCred'].value_counts(normalize=True)*100) < 5, 'High')
  

  
# Merging levels yes, co-applicant and yes, guarantor as they have less than 6% of value counts into 'yes' category
for i in loan['Oparties']:
  loan['Oparties'] = loan['Oparties'].mask(loan['Oparties'].map(loan['Oparties'].value_counts(normalize=True)*100) < 6, 'yes')
  

# Percentage of frequency of each level in Other parties
loan['Oparties'].value_counts(normalize = True)*100 

# Dropping columns 'telephone' as it does not have any logical connection with loan default. Dropping columns 'foreign' because of low variability. 
# Only 3.7% are foreign workers



#___________________________________________________________________________________________________

#--------------------------------------Function for bar plot------------------------------------------


def plotting_percentages(df, col, target):
    x, y = col, target
    
    # Temporary dataframe with percentage values
    temp_df = df.groupby(x)[y].value_counts(normalize=True) # First group by the selected column according to their unique values then finding value_counts for creditscore within each groupby category
    temp_df = temp_df.mul(100).rename('percent').reset_index() # resets the index from 0 to length of index of columns

    # Sort the column values for plotting    
    order_list = list(df[col].unique())
    order_list.sort()

    # Plot the figure
    sns.set(font_scale=1.5)
    g = sns.catplot(x=x, y='percent', hue=y,kind='bar', data=temp_df, 
                    height=8, aspect=2, order=order_list, legend_out=False)   # hue=y, here y is target column
    g.ax.set_ylim(0,100)

    # Loop through each bar in the graph and add/print the percentage values as labels in bar chart    
    for p in g.ax.patches:
        txt = str(p.get_height().round(1)) + '%'            # Get height of bar, round it to 1 decimal place
        txt_x = p.get_x() 
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)
        
    # Set labels and title
    plt.title(f'{col.title()} By Percent {target.title()}', 
              fontdict={'fontsize': 30})
    plt.xlabel(f'{col.title()}', fontdict={'fontsize': 20})
    plt.ylabel(f'{target.title()} Percentage', fontdict={'fontsize': 20})
    plt.xticks(rotation=75)
    return g


cols_list = categorical_data.columns
print(type(cols_list))
cols_list.delete(14) # Removing creditScore label as we dont want to make bar plot for it.

for i in range(0,15):
    plotting_percentages(loan, cols_list[i], 'creditScore')

"""
Interpretation: 
1. As the amount in checking account increases the percentage of good credit score also increases while bad credit score decreases.  
   50% among clients with negative account balance have good chances of promptly closing credit while 50% of them have bad chances of closing  
   Clients with no checking account has best credit score. 88.3% among them have high chance of closing credit
2. Loan taken for new vehicle has most percentage of good credit score and loan taken for education has worst credit score
3. Status (MSG) does not have much impact on default or not default. So we can remove the variable 
4. Oparties Removed:The percentage of good and bad credit score is same for both categories of Oparties. So presence or absence
   of guarantor or co-applicant has no impact on credit score. Removing this variable.
5. Rdur Removal: Duration of current residance has almost no effect on credit score as percentage of good or bad is same in all categories.
6. Prop Merge other cars and life insurance,building society levels: Clients with other property as real estate has best credit score and with unknown other property have minimum percentage of good credit scorers
   and maximum percentage of bad credit scorers.
7. Inplans: Clients with no other installments to pay have best credit score while other two categories have exact same percentage of good or bad credit scores.
   So, we can combine other two categories as one category because they both have same impact on credit scores   
8. Htype: 73% of Clients who owned a house has good credit score whereas 60% of those who rent or staying for free has good credit score. Other two categories has 
   similar impact on credit score so will merge these two categories as 'House not owned' 
9. Jobtype Removal: All categories have almost same percentage of good(70% approx.) or bad(30% approx. credit score. It seems that this variable does not have impact on credit score. 
"""

#InRate Removed: The good credit score for all categories varies from 67% to 75% and bad scrore varies from 25% to 33.4% for all categories. The variation is 8% for both good and bad score for all categories.
#The variation is very less so this variable does not seem to have much impact on credit score. Removing this variable.

plotting_percentages(loan, 'InRate','creditScore')


# Merging bank and stores levels of inPlans column and assigning the merged level as 'Yes' category
loan['inPlans'].value_counts(normalize=True)*100
for i in loan['inPlans']:
  loan['inPlans'] = loan['inPlans'].mask(loan['inPlans'].map(loan['inPlans'].value_counts(normalize=True)*100) < 14, 'Yes')
plotting_percentages(loan, 'inPlans','creditScore')



# Merging pays rent and free levels of Htype column and assigning the merged level as 'House not owned' category
loan['Htype'].value_counts(normalize=True)*100
for i in loan['Htype']:
  loan['Htype'] = loan['Htype'].mask(loan['Htype'].map(loan['Htype'].value_counts(normalize=True)*100) < 18, 'House not owned')
plotting_percentages(loan, 'inPlans','creditScore')
loan['Htype'].value_counts(normalize=True)*100
plotting_percentages(loan, 'Htype','creditScore')



# Merging other cars etc. and life insurance building society levels of Prop column and assigning the merged level as 'Other property' category
loan['Prop'].value_counts(normalize=True)*100

loan['Prop'] = loan.Prop.replace(["Other cars etc.","life insurance/building society"],"Other property")
loan['Prop'].value_counts(normalize=True)*100
plotting_percentages(loan, 'Prop','creditScore')




# Converting the data type of categorical variables 'InRate', 'NumCred', Ndepend' from int to object data type
loan['InRate'] = loan['InRate'].astype('object')
loan['NumCred'] = loan['NumCred'].astype('object')
loan['Ndepend'] = loan['Ndepend'].astype('object')




plotting_percentages(loan, 'Ndepend','creditScore') # Removing this column as both categories have similar percentages of good and bad credit scores.
#plotting_percentages(loan, 'NumCred','creditScore')

# Created temporary dataframe for plotting bar plot of 'NumCred'
temp = loan.groupby('NumCred')['creditScore'].value_counts(normalize=True)
temp=temp.mul(100).rename('percent').reset_index()
sns.catplot(x='NumCred', y='percent', hue='creditScore',kind='bar', data=temp, height=8, aspect=2,legend_out=False) # Removing variable because the percentage of good or bad credit score does not vary much with categories.
#sns.countplot(x= 'NumCred', hue = 'creditScore', data = loan) # Removing variable because the percentage of good or bad credit score does not vary much with categories.
plotting_percentages(loan, 'JobType','creditScore')


# Creating categories in Age variable

def categorize(row):
    if loan['Age'] > 18 and loan['Age'] < 27:
        return 'Student'
    if loan['Age'] >=27 and loan['Age'] < 40:
        return 'Middle aged'
    if loan['Age'] >40:
        return 'Upper age'
    
loan['Age'] = loan.apply(lambda row: categorize(row), axis=1)



# Dropping the variables
loan = loan.drop(columns = ['Oparties','Rdur','JobType','InRate','Ndepend','NumCred','telephone','foreign','MSG'], axis = 1)
#__________________________________________________________________________________________________
 

# Correlation of numerical variables
corr = loan.corr()

plt.figure(figsize = (10,6))
sns.heatmap(data= corr, annot = True)
plt.show()
# Interpretation: Camt which has average positive correlation with Cdur.
# As the loan amount increase the credit duration also increases. 

# Summary 
summary = loan.describe()
print(summary)
# The ranges of age, Camt and Cdur variables are possible.



# Pairplot
#plt.plot(figsize = (15,15))
sns.pairplot(data = loan, hue = 'creditScore')




# Cdur
sns.boxplot(y=loan['Cdur'])
sns.boxplot(y= 'Cdur', x = 'creditScore', data = loan)
# Good credit scorers have low credit duration whereas bad credit scorers have high credit duration and wide range of credit duration



# Camt
sns.boxplot(y=loan['Camt'])
sns.boxplot(y=loan['Camt'], x= loan['creditScore'])
# Good credit scorers have low credit amount whereas bad credit scorers have high credit amount and wide range of credit duration



# Age
sns.boxplot(y=loan['age'])
sns.boxplot(y=loan['age'], x = loan['creditScore'])
# age can vary in range of 19 to 75 as in our variable.
# There is not much variation in credit scores according to age as both categories of credit score belaongs to almost similar age ranges and
# 50% of good credit scorers are from age group 27 to 45 and bad credit scorers range from 25 to 40.
# 75% of good credit scorers are below age 45yrs and bad credit scorers below age 40yrs.
# Removing 'age' variable

""" 
Decision: We are not removing outliers because defaulters have behavious different from normal. So we want to consider the abnormal behaviours
which may be hidden in outliers"""


# Removing age variable
loan = loan.drop(columns = ['age'], axis = 1)


# Removing duplicates, if present
loan.drop_duplicates(keep = 'first', inplace = True)



# Checking the balance of dataset
sns.countplot(loan['creditScore'])
print(loan['creditScore'].shape)
default = loan[loan['creditScore']=='bad'].shape[0]/loan.shape[0] #.shape[0] gives number of rows with creditScore == 'bad'
print(default)
# The dataset is imbalanced. 70% values belongs to good credit score and only 30% values belong to bad credit score


#_________________________________________________________________________________________________________
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


X = loan.drop(columns=['creditScore'], axis=1)
y = loan['creditScore']

cat_data = loan.select_dtypes(include = ['object']).copy()
cat_data = cat_data.drop('creditScore', axis = 1)
dummies_cat_data = pd.get_dummies(cat_data, drop_first = True)


num_data = loan.select_dtypes(include = ['int64']).copy()
# Standardization of numerical variables before applying models as all numerical variables are on different scale
scaler = StandardScaler() 
num_data_scaled = scaler.fit_transform(num_data)
num_data_scaled = pd.DataFrame(num_data_scaled)
num_data_scaled.columns = ['Cdur','Camt'] # Assigned column names to dataframe


X = pd.concat([num_data_scaled.reset_index(drop=True), dummies_cat_data.reset_index(drop=True)], axis=1, ignore_index=True)


concatenated_dataframes_columns = [list(num_data_scaled.columns), list(dummies_cat_data.columns)]

flatten = lambda nested_lists: [item for sublist in nested_lists for item in sublist]
X.columns = flatten(concatenated_dataframes_columns)


# Label encoding of target variable
y=y.map({'good':0, 'bad':1}) # bad represents default and good represents non-default


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45)


#________________________________________________________________________________________________

# Oversampling
X_train_over_sampled, Y_train_over_sampled = SMOTE(random_state=0).fit_resample(X_train, y_train)


# Counting the no. of samples in each class of y_train using counter

from collections import Counter
Counter(y_train)

# After fitting SMOTE we get upsampled dataset
print(Counter(Y_train_over_sampled))

#________________________________________________________________________________________________

# Evaluation Function

def evaluate_classification_model(Y_true, Y_pred):
    """
    Custom classification model evaluation function
    """

    # Confusion_matrix
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    sns.heatmap(data=conf_matrix, annot=True, fmt='g')
    plt.title("Confusion Matrix")
    plt.show()
    print(f"Number of missclassified samples: {(y_test != Y_pred).sum()}")
    
    
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




sns.set('talk', 'whitegrid', 'dark', font_scale=1, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = metrics.roc_curve(truth,pred)       # Returns fpr, tpr and thresholds
    roc_auc = metrics.auc(fpr, tpr)                            # Returns area under the curve 
    lw = 2
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color= c,lw=lw, label= lab +'(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve') #Receiver Operating Characteristic 
    plt.legend(loc="lower right", fontsize=10)
#_______________________________________________________________________________________________

#------------------------------------------KNearest Neighbors-----------------------------------
# Elbow method for determining the value of K

misclassified_samples = [] # Creating empty list for misclassified samples


for i in range(1, 20, 2):
    KNN_model = KNeighborsClassifier(n_neighbors = i)
    KNN_model.fit(X_train_over_sampled, Y_train_over_sampled)
    Y_preds_KNN = KNN_model.predict(X_test)
    misclassified_samples.append((y_test != Y_preds_KNN).sum())
    
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
# The plot does not have an elbow so we could not choose best neihbour from elbow method.




param_grid = [{
    'kneighborsclassifier__n_neighbors': [ 3, 5, 7, 8, 9, 10],
    'kneighborsclassifier__p': [1, 2],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'kneighborsclassifier__metric': ['minkowski','euclidean','manhattan'],
    'kneighborsclassifier__leaf_size': [3,5,7,11,17,21,25]
}]

imba_pipeline1 = make_pipeline(SMOTE(random_state=42), KNeighborsClassifier())


# Create a grid search was taking a lot of time so we chose Randomized search as there are lot of parameters
# and no. of iterations will get 6 X 2 X 2X4X3X7 for each fold.
gs = RandomizedSearchCV(imba_pipeline1, param_distributions = param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=5,
                  verbose=1,
                  n_jobs=2)

# Fit the most optimal model
#
gs.fit(X_train_over_sampled, Y_train_over_sampled)

# Print the best model parameters and scores
gs.best_params_

gsknn = gs.best_estimator_
best_prediction_knn = gsknn.predict(X_test)


evaluate_classification_model(y_test, best_prediction_knn)
"""Number of missclassified samples: 71
Accuracy of the model: 71.60%
Precision of the model: 0.53
Recall of the model: 0.70
F1 score of the model: 0.60
Area under the ROC Curve: 0.71"""


y_preds_knn_prob = gsknn.predict_proba(X_test)[:,1]
plotAUC(y_test, y_preds_knn_prob, 'KNearest Neighbor') # AUC = 75%

#_______________________________________________________________________________________________

#---------------------------------------SVC--------------------------------------------------

pipelineSVC = make_pipeline(SMOTE(random_state=42), SVC(random_state=1, probability=True))


# Create the parameter grid
# C-Regularization Parameter. C tries to control the penalty for misclassified samples
# If data is non-linearly separable then rbf and polykernels do well. We need to find out which kernel is 
# best for our data i.e. whether data is linearly separable or non-linearly separable
# For a linear kernel, we just need to optimize the c parameter. However, if we want to use an RBF kernel, 
# both c and gamma parameter need to optimized simultaneously.  
param_grid_svc = [{
                    'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                    'svc__kernel': ['linear']
                  },
                 {
                    'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                    'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                    'svc__kernel': ['rbf']
                 }]

# Create an instance of GridSearch Cross-validation estimator
gsSVC = GridSearchCV(estimator=pipelineSVC,
                     param_grid = param_grid_svc,
                     scoring='accuracy',
                     cv=5,
                     refit=True,
                     n_jobs=1)

# Train the SVM classifier
gsSVC.fit(X_train_over_sampled, Y_train_over_sampled)
#
# Print the training score of the best model
print(gsSVC.best_score_)

# Print the model parameters of the best model
print(gsSVC.best_params_)
#{'svc__C': 10.0, 'svc__gamma': 0.5, 'svc__kernel': 'rbf'} same every time


clfSVC = gsSVC.best_estimator_
best_prediction = clfSVC.predict(X_test)


evaluate_classification_model(y_test, best_prediction)
""" All scores have improved when applied on upsampled data directly
Number of missclassified samples: 63
Accuracy of the model: 74.80%
Precision of the model: 0.59
Recall of the model: 0.57
F1 score of the model: 0.58
Area under the ROC Curve: 0.70"""


y_preds_svc = clfSVC.predict_proba(X_test)[:,1] # y_preds_svc will have two columns of probability default or non-default.
# So selecting only one column as it can tell the probability for another (1 - (prob of default))

plotAUC(y_test, y_preds_svc,'SVM')

#________________________________________________________________________________________________

#------------------------------------Logistic Regression-----------------------------------------

pipelineLR = make_pipeline(SMOTE(random_state=42), LogisticRegression(random_state=1, penalty='l2', solver='lbfgs'))

# Create the parameter grid

param_grid_lr = [{
    'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
}]

# Create an instance of GridSearch Cross-validation estimator

gsLR = GridSearchCV(estimator=pipelineLR,
                     param_grid = param_grid_lr,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)

# Train the LogisticRegression Classifier
gsLR = gsLR.fit(X_train_over_sampled, Y_train_over_sampled)

# Print the training score of the best model
print(gsLR.best_score_)

# Print the model parameters of the best model
print(gsLR.best_params_) # {'logisticregression__C': 0.1}

# Print the test score of the best model
clfLR = gsLR.best_estimator_
best_prediction_lr = clfLR.predict(X_test) # Predicting on test data with best parameters


evaluate_classification_model(y_test, best_prediction_lr)
"""Number of missclassified samples: 75
Accuracy of the model: 70.00%
Precision of the model: 0.51
Recall of the model: 0.60
F1 score of the model: 0.55
Area under the ROC Curve: 0.67"""

    
y_preds_lr = clfLR.predict_proba(X_test)[:,1] # y_preds_lr will have two columns of probability default or non-default.
# So selecting only one column as it can tell the probability for another (1 - (prob of default))

plotAUC(y_test, y_preds_lr,'Logistic Regression') # AUC=0.78

#___________________________________________________________________________________________________

#----------------------------------------Random Forest-------------------------------------------------------

params = {'n_estimators': [50, 100, 200],
 'max_depth': [4, 6, 10, 12],
 'random_state': [13]}


imba_pipeline = make_pipeline(SMOTE(random_state=42), RandomForestClassifier(random_state=13))


new_params = {'randomforestclassifier__' + key: params[key] for key in params} # Passing the parameters
grid_imba = GridSearchCV(imba_pipeline, param_grid=new_params, cv=5, scoring='accuracy', return_train_score=True)
grid_imba.fit(X_train_over_sampled, Y_train_over_sampled)

grid_imba.best_params_
"""'randomforestclassifier__max_depth': 12,
 'randomforestclassifier__n_estimators': 100,
 'randomforestclassifier__random_state': 13"""
 

clfrf = grid_imba.best_estimator_
best_prediction_rf = clfrf.predict(X_test) 



y_preds_rf = clfrf.predict(X_test)

y_preds_rf_prob = clfrf.predict_proba(X_test)[:,1]

plotAUC(y_test, y_preds_rf_prob,'Random Forest') # AUC = 0.79



evaluate_classification_model(y_test, best_prediction_rf)
"""Number of missclassified samples: 66
Accuracy of the model: 73.60%
Precision of the model: 0.57
Recall of the model: 0.61
F1 score of the model: 0.59
Area under the ROC Curve: 0.70"""

#___________________________________________________________________________________________________________

#---------------------------------------Neural Networks----------------------------------------

from sklearn.neural_network import MLPClassifier

# Normalization of the train and test data
scaler_nn = MinMaxScaler()
features_names = X_train.columns
X_train_nn = scaler_nn.fit_transform(X_train)
X_train_nn = pd.DataFrame(X_train_nn, columns = features_names)
X_test_nn = scaler_nn.transform(X_test)
X_test_nn = pd.DataFrame(X_test_nn, columns = features_names)

%%time
mlp_nn = MLPClassifier(solver = 'adam', random_state = 42, max_iter = 1000 )
parameters = {'hidden_layer_sizes': [(20,), (20,10), (20, 10, 2)], 'learning_rate_init':[0.0001, 0.001, 0.01, 0.1]}
clf_nn = GridSearchCV(mlp_nn, parameters, cv = 5).fit(X_train_nn, y_train)
#---
y_preds_nn = clf_nn.predict(X_test)
y_preds_nn_prob = clf_nn.predict_proba(X_test_nn)[:,1]
#---
plotAUC(y_test, y_preds_nn_prob,'Neural Networks')
evaluate_classification_model(y_test, y_preds_nn)

#_______________________________________________________________________________________________

#--------------------------------ALL ROC curves---------------------------------------------------------------

sns.set('talk', 'whitegrid', 'dark', font_scale=1, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
def plotAUC(truth, pred, lab):
    fpr, tpr, _ = metrics.roc_curve(truth,pred)
    roc_auc = metrics.auc(fpr, tpr)
    lw = 2
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color= c,lw=lw, label= lab +'(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve') #Receiver Operating Characteristic 
    plt.legend(loc="lower right", fontsize=10)


plotAUC(y_test, y_preds_svc, 'SVM')
plotAUC(y_test, y_preds_knn_prob, 'KNN')
plotAUC(y_test, y_preds_lr,'Logistic Regression')
plotAUC(y_test, y_preds_rf_prob, 'Random Forest')
plotAUC(y_test, y_preds_nn_prob, 'Neural Networks')
plt.show()


#___________________________________________________________________________________________________________


# Performance comparisons between models

results_1 = {'Classifier': ['AUC ROC (%)','TN (%)','FP (%)','FN (%)','TP (%)'],
'Logistic Regression (LR)': [aucroclr, (tn_lr/3956*100).round(2), (fp_lr/3956*100).round(2), (fn_lr/3956*100).round(2), (tp_lr/3956*100).round(2)],
'K Nearest Neighbour (KNN)': [aucrocknn, (tn_knn/3956*100).round(2),(fp_knn/3956*100).round(2), (fn_knn/3956*100).round(2),(tp_nn/3956*100).round(2)],
'Support Vector Machine (SVC)': [aucrocsvc, (tn_svc/3956*100).round(2),(fp_svc/3956*100).round(2), (fn_svc/3956*100).round(2),(tp_svc/3956*100).round(2)],
'Random Forest (RF)': [aucrocrf, (tn_rf/3956*100).round(2), (fp_rf/3956*100).round(2), (fn_rf/3956*100).round(2),(tp_rf/3956*100).round(2)]}


df1 = pd.DataFrame(results_1, columns = ['Classifier', 'Logistic Regression (LR)', 'K Nearest Neighbour (KNN)', 'Support Vector Machine (SVC)', 'Random Forest (RF)'])

df1.set_index("Classifier", inplace=True)
results = df1.T
results

from sklearn.metrics import classification_report
print("RF",classification_report(y_test, best_prediction_rf, target_names=None))
print("SVM",classification_report(y_test, best_prediction, target_names=None))
print("LR",classification_report(y_test, best_prediction_lr, target_names=None))
print("KNN",classification_report(y_test, best_prediction_knn, target_names=None))
print("MLP",classification_report(y_test, y_preds_nn, target_names=None))


#________________________________________________________________________________________________________________

# Important features in Random Forest model

importances = clf.feature_importances_

#from feature_importance import FeatureImportance
#feature_importance = FeatureImportance(pipe)
#feature_importance.plot(top_n_features=25)


d = {'Features': X_train.columns, 'FI': grid_imba.best_estimator_.named_steps["randomforestclassifier"].feature_importances_}
df = pd.DataFrame(d)

# Display the top 30 feature based on feature importance
topFI = df.sort_values(by='FI', ascending=False).head(30)

# Plot feature importance
plt.rcParams["figure.figsize"] = (10, 10)

y_axis = topFI['Features']
x_axis = topFI['FI']
plt.barh(y_axis, x_axis)
plt.title('Plot FI')
plt.ylabel('Features')
plt.xlabel('Feature Importance')
plt.gca().invert_yaxis()

#_______________________________________________________________________________________________


# Important features

"""importances = clf.feature_importances_

from feature_importance import FeatureImportance
feature_importance = FeatureImportance(pipe)
feature_importance.plot(top_n_features=25)"""


d = {'Features': X_train.columns, 'FI': gsLR.best_estimator_.named_steps["logisticregressionclassifier"].feature_importances_}
df = pd.DataFrame(d)

# Display the top 30 feature based on feature importance
topFI = df.sort_values(by='FI', ascending=False).head(30)

# Plot feature importance
plt.rcParams["figure.figsize"] = (10, 10)

y_axis = topFI['Features']
x_axis = topFI['FI']
plt.barh(y_axis, x_axis)
plt.title('Plot FI')
plt.ylabel('Features')
plt.xlabel('Feature Importance')
plt.gca().invert_yaxis()

#___________________________________________________________________________________________________________________


""" Understanding percentage plot
temp_df = loan.groupby(loan["Htype"])["creditScore"].value_counts(normalize=True) # First group by each column according to their unique values then finding value_counts for creditscore within each groupby category
temp_df = temp_df.mul(100).rename('percent').reset_index() """


