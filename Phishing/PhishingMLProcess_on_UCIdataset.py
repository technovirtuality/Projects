# Importing necessary libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from collections import Counter


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

# Importing performance metrics
from sklearn import preprocessing,metrics

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split 
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)

# Hyperparameter Tuning
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)

# Importing performance metrics
from sklearn import preprocessing,metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve, plot_confusion_matrix


import os

os.chdir('E:\Anukriti\Avrutti Research\Task 9 Phishing\Dataset')

data = pd.read_csv('Training Dataset.csv')
df = data.copy()

df.columns

df.isnull().sum() # There are no missing values present in dataset
df.info()

# Finding number of levels in each variable
unique = df.nunique().sort_values(ascending = False)
# The variables which has only one unique value have same fixed value for all samples so such variables needs to be removed as they cannot be considered as variable 

# Removing variables which has same value for all samples
# Extracting variable names with only 1 uniques level
noVar = unique[unique == 1].index  # No such variable found

#df.drop(columns = noVar, axis = 1, inplace = True)

# Removed duplicates
df.drop_duplicates(keep = 'first', inplace = True)

# There are no special characters present in columns with object data types
frequencies = df.apply(lambda x: x.value_counts()).T.stack()
print(frequencies)

# Percentage of frequencies of each level in all categorical variables. 
frequencies1 = df.apply(lambda x: x.value_counts(normalize=True)*100).T.stack()
print(frequencies1)

sns.pairplot(df, hue = 'Result', palette = 'hls')
plt.show()
columns = list(df.columns)
colhalf1 = columns[0:15]
colhalf1.append('Result')
dataframe = df[colhalf1]
sns.pairplot(dataframe, hue = 'Result', palette = 'hls')
plt.show()

"""
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


cols_list = df.columns
print(type(cols_list))
cols_list.delete(30) # Removing creditScore label as we dont want to make bar plot for it.

for i in range(0,29):
    plotting_percentages(df, cols_list[i], 'Result')

"""

# Function for classification report and confusion matrix
# Parameters: test data actual labels in y_test and predicted labels from model in y_pred_test
# Returns: Prints classification report and plot of confusion matrix
# Function for classification report and confusion matrix
def evaluate_classification_model(y_test, y_pred_test):
    
    target_names = ['Non-phishing', 'Phishing']

    print(classification_report(y_test, y_pred_test, target_names=target_names))


    conf_matrix = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(data=conf_matrix, annot=True, fmt='g', xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xtickslabels(rotation=90)
    plt.ytickslabels(rotation=90)
    plt.yticks(target_names)
    plt.show()


## Define function for obtaining predicted class and predicted probabilities for the classes when model is applied on test data
def fit_n_pred(model, X_tr, y_tr, X_te):
    
    """Takes in Classifier, training data (X,y), and test data(X). Will output 
    predictions based upon both the training and test data using the sklearn
    .predict method. MUST unpack into two variables (train, test)."""
    
    ## Fitting classifier to training data
    model.fit(X_tr, y_tr)

    ## Generate predictions for training + test data
    y_preds = model.predict(X_te)
    y_preds_prob = model.predict_proba(X_te)[:,1]
    
    ## Optional display check
    #display(model)
    #print(model.best_params_)
    
    return y_preds, y_preds_prob


#plot_curve(y_test, y_preds_prob, 'Random Forest with Important Features')
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
    
    
def feature_importance(model, X, label):
# Set importance features
    importance = model.feature_importances_

    d = {'Features':X.columns, 'Feature Importance': model.feature_importances_}
    dataframe = pd.DataFrame(d)

# Display the top 30 feature based on feature importance
    topFI = dataframe.sort_values(by='Feature Importance', ascending=False).head(30)

# Plot feature importance
    plt.rcParams["figure.figsize"] = (10, 10)

    y_axis = topFI['Features']
    x_axis = topFI['Feature Importance']
    plt.barh(y_axis, x_axis)
    plt.title('Plot FI: ' + label)
    plt.ylabel('Features')
    plt.xlabel('Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('E:/Anukriti/Avrutti Research/Task 9 Phishing/UciFiles/FI_RF_1l_new.png')
    plt.show()
    
def save_model(model, filename):
    import pickle
    from os import path
    pkl_filename = "E:/Anukriti/Avrutti Research/Task 9 Phishing/UciFiles" + "/" + filename + ".pkl"
    if (not path.isfile(pkl_filename)):
  # saving the trained model to disk 
      with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
      print("Saved model to disk")
    else:
      print("Previous Model exists on the disk! Please Remove")

df.columns

# Feature Importance: Choice of ML algorithms:
# 1. Random Forest
# 2. XG Boost


# Feature Importance: Random Forest

# Creating train test data (retaining original dataframe in df)
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
#X = df.drop(columns = ['Result','having_At_Symbol','Favicon','port','HTTPS_token','Submitting_to_email','on_mouseover','RightClick','popUpWidnow','Iframe','DNSRecord'], axis = 1)
X = df.drop(columns = ['Result'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)



# Checking the balance of dataset
from collections import Counter
Counter(df['Result']) # Data is nearly balanced
phishingClassRatio = df[df['Result'] == 1].shape[0]/df.shape[0]*100
print("Percentage of phishing data in dataset: ", round(phishingClassRatio,2))


#------------------------------------------------Random Forest Model--------------------------------------------

# Creating train test data (retaining original dataframe in df)
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
#X = df.drop(columns = ['Result','having_At_Symbol','Favicon','port','HTTPS_token','Submitting_to_email','on_mouseover','RightClick','popUpWidnow','Iframe','DNSRecord'], axis = 1)
X_cat = df.drop(columns = ['Result'], axis = 1)
X_cat = X_cat.astype(str)
X_cat.info()


# Dummy encoding
X = pd.get_dummies(X_cat, drop_first = True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)


# 1. Random Forest Hyperparameter Tuning
params = {'n_estimators': [50, 100, 200],
 'max_depth': [4, 6, 10, 12],
 'criterion': ['gini', 'entropy'],
 'random_state': [13]}


imba_pipeline = make_pipeline(RandomForestClassifier(random_state=13))

# Attaching name of classifier with name of each parameter for passing to Randomized Search CV
new_params = {'randomforestclassifier__' + key: params[key] for key in params} # Passing the parameters

rs_rf = RandomizedSearchCV(imba_pipeline, new_params, cv=10, scoring = 'accuracy')

y_pred, y_pred_prob = fit_n_pred(rs_rf, X_train, y_train, X_test)
rs_rf.best_params_

evaluate_classification_model(y_test, y_pred)
plotAUC(y_test, y_pred_prob, 'Random Forest')
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)



# Randomized Search object has no attribute feature_importance_ so creating new RF object with best parameters
# 2.  Random Forest Model with Best parameters

"""
# Preparing data
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
X = df.drop(columns = ['Result','having_At_Symbol','Favicon','port','HTTPS_token','Submitting_to_email','on_mouseover','RightClick','popUpWidnow','Iframe','DNSRecord'], axis = 1)
col = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = col)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)
"""

rf = RandomForestClassifier(n_estimators = 200, max_depth = 12, random_state = 13, criterion = 'gini')
rf.fit(X_train,y_train)
feature_importance(rf, X, 'FI: Random Forest')


# Fit model using each importance as a threshold
from sklearn.feature_selection import SelectFromModel

thresholds = sorted(rf.feature_importances_, reverse = True)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(rf, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = RandomForestClassifier(n_estimators = 200, max_depth = 12, random_state = 13, criterion = 'gini')
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	predictions = selection_model.predict(select_X_test)
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# Selecting threshold=0.009 with n=37 achieving an accuracy of 95.86%
# Random Forest classifier with important features
importance = rf.feature_importances_
d = {'Features':X.columns, 'Feature Importance': rf.feature_importances_}
dataframe = pd.DataFrame(d)

# Display the top 37 feature based on feature importance. The highest accuracy of 95.86% is achieved at n=37 and FI score = 0.09. Therefore selecting top 37 features
topFI = dataframe.sort_values(by='Feature Importance', ascending=False).head(37)

# Selecting the columns with feature importance score of more than 0.061 because accuracy is increased and number of features is also nominal otherwise if accuracy of 77.35% is chosen then n=3 is very less for model to identify pattern
selCols = topFI.loc[topFI['Feature Importance'] >= 0.002]
selCols = selCols['Features'].tolist()


X = X[selCols]
#y = df['phishing']
col = X.columns

"""
X.columns

sc = StandardScaler()
X = sc.fit_transform(X)
X = pd.DataFrame(X, columns = col)
"""
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)

rf = RandomForestClassifier(n_estimators = 200, max_depth = 12, random_state = 13, criterion = 'gini')
y_pred, y_pred_prob = fit_n_pred(rs_rf, X_train, y_train, X_test)

evaluate_classification_model(y_test, y_pred)
plotAUC(y_test, y_pred_prob, 'Random Forest')

save_model(rf, 'model_RfFi2')


#--------------------------------------XGB Classifier-----------------------------------------------------

# Creating train test data (retaining original dataframe in df)
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
#X = df.drop(columns = ['Result','having_At_Symbol','Favicon','port','HTTPS_token','Submitting_to_email','on_mouseover','RightClick','popUpWidnow','Iframe','DNSRecord'], axis = 1)
X_cat = df.drop(columns = ['Result'], axis = 1)
X_cat = X_cat.astype(str)
X_cat.info()
#X1 = pd.get_dummies(X_cat)

# One hot encoding
X = pd.get_dummies(X_cat, drop_first = True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)



params = {
"xgbclassifier__learning_rate" : [0.05, 0.1, 0.20],
#"subsample" : [0.6, 0.8, 1.0],
"xgbclassifier__max_depth" : [3,4,5, 6, 8, 10],
"xgbclassifier__min_child_weight" : [1,3,5],
"xgbclassifier__gamma" : [0.0, 0.5, 1.0, 1.5]
    }

#imba_pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state=13))
imba_pipeline = make_pipeline(XGBClassifier(random_state=13))

rs_xg = RandomizedSearchCV(imba_pipeline, params, cv=10, scoring = 'accuracy')

y_pred, y_pred_prob = fit_n_pred(rs_xg, X_train, y_train, X_test)
rs_xg.best_params_

evaluate_classification_model(y_test, y_pred)
plotAUC(y_test, y_pred_prob, 'XG Boost')
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)



# Randomized Search object has no attribute feature_importance_ so creating new XG Boost object with best parameters
# 2. XG Boost Model with Best parameters
"""
# Creating train test data (retaining original dataframe in df)
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
#X = df.drop(columns = ['Result','having_At_Symbol','Favicon','port','HTTPS_token','Submitting_to_email','on_mouseover','RightClick','popUpWidnow','Iframe','DNSRecord'], axis = 1)
col = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = col)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)
"""

# Creating instance of XG Boost and using best hyper-parameters same as obtained during hyper-parameter tuning.
# Need to fill values of hyper-parameters manually 
xgb = XGBClassifier(booster = 'gbtree', random_state=13, learning_rate=0.2, max_depth= 6, gamma = 1.0, min_child_weight = 1)
xgb.fit(X_train,y_train)
feature_importance(xgb, X, 'FI:XgBoost1')



# Fit model using each importance as a threshold

from sklearn.feature_selection import SelectFromModel

thresholds = sorted(xgb.feature_importances_, reverse = True)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(xgb, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier(booster = 'gbtree', random_state=13, learning_rate=0.2, max_depth= 6, gamma = 1.0, min_child_weight = 1)
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	predictions = selection_model.predict(select_X_test)
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

# The best accuracy of 96.01% can be obtained when threshold = 0.006 and no. of features = 34


# XG Boost classifier with important features
importance = xgb.feature_importances_
d = {'Features':X.columns, 'Feature Importance': xgb.feature_importances_}
dataframe = pd.DataFrame(d)

# Display the top 30 feature based on feature importance
topFI = dataframe.sort_values(by='Feature Importance', ascending=False).head(34) # Change to n=34(no. of columns) mentioned above

selCols = topFI.loc[topFI['Feature Importance'] > 0.006]
selCols = selCols['Features'].tolist()

X = X[selCols]
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
col = X.columns

for i in range(len(col)):
    sns.distplot(X[col[i]])
    plt.show()

"""
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = col)
"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)

###Change XGB classifier parameters###
xgb = XGBClassifier(booster = 'gbtree', random_state=13, learning_rate=0.2, max_depth= 6, gamma = 1.0, min_child_weight = 1)
y_pred, y_pred_prob = fit_n_pred(xgb, X_train, y_train, X_test)

evaluate_classification_model(y_test, y_pred)
plotAUC(y_test, y_pred_prob, 'XG Boost') # AUC = 99%
save_model(xgb, 'xg_model3_96')

print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)


# Plot between real and predicted data
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("XGBoost Binary Classification")
plt.savefig('plots/xgb_real_pred_bin.png')
plt.show()

#______________________________________________________________________________________________________


#---------------------------------Logistic Regression: Trial---------------------------------------

from sklearn.linear_model import LogisticRegression

# Creating train test data (retaining original dataframe in df)
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
#X = df.drop(columns = ['Result','having_At_Symbol','Favicon','port','HTTPS_token','Submitting_to_email','on_mouseover','RightClick','popUpWidnow','Iframe','DNSRecord'], axis = 1)
X_cat = df.drop(columns = ['Result'], axis = 1)
X_cat = X_cat.astype(str)
X_cat.info()
#X1 = pd.get_dummies(X_cat)

# One hot encoding
X = pd.get_dummies(X_cat, drop_first = True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)


# Create the parameter grid
param_grid_lr = [{
    'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
}]


pipelineLR = make_pipeline(LogisticRegression(random_state=1, penalty='l2', solver='lbfgs'))

# Create an instance of GridSearch Cross-validation estimator

gsLR = GridSearchCV(estimator=pipelineLR,
                     param_grid = param_grid_lr,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)

# Train the LogisticRegression Classifier
#gsLR = gsLR.fit(X_train, y_train)
y_pred, y_pred_prob = fit_n_pred(gsLR, X_train, y_train, X_test)



# Print the training score of the best model
print(gsLR.best_score_)

# Print the model parameters of the best model
print(gsLR.best_params_) # {'logisticregression__C': 0.001}

# Print the test score of the best model
clfLR = gsLR.best_estimator_

y_pred, y_pred_prob = fit_n_pred(clfLR, X_train, y_train, X_test)

evaluate_classification_model(y_test, y_pred)
plotAUC(y_test, y_pred_prob, 'Logistic Regression') # AUC = 99%
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)


# Plot between real and predicted data
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("XGBoost Binary Classification")
plt.savefig('plots/LogReg_real_pred_bin.png')
plt.show()


#------------------------------------------KNearest Neighbors-----------------------------------
# Elbow method for determining the value of K

misclassified_samples = [] # Creating empty list for misclassified samples


for i in range(1, 40, 2):
    KNN_model = KNeighborsClassifier(n_neighbors = i)
    KNN_model.fit(X_train, y_train)
    Y_preds_KNN = KNN_model.predict(X_test)
    misclassified_samples.append((y_test != Y_preds_KNN).sum())
    
print(misclassified_samples) 
# The number of misclassified samples for all values of K is more than number of misclassified samples in logistic regression which is 279.


# Elbow Plot
plt.figure(figsize = (10,6))
plt.plot(range(1,40,2), misclassified_samples, marker = 'o')
plt.xticks(range(1,40,2))
plt.title("Elbow Plot")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Number of misclassified samples")
plt.show() 
# The plot does not have an elbow so we could not choose best neihbour from elbow method.





# Creating train test data (retaining original dataframe in df)
y = df['Result']
# Label encoding of target variable
y=y.map({-1:0, 1:1}) 
#X = df.drop(columns = ['Result','having_At_Symbol','Favicon','port','HTTPS_token','Submitting_to_email','on_mouseover','RightClick','popUpWidnow','Iframe','DNSRecord'], axis = 1)
X_cat = df.drop(columns = ['Result'], axis = 1)
X_cat = X_cat.astype(str)
X_cat.info()
#X1 = pd.get_dummies(X_cat)

# One hot encoding
X = pd.get_dummies(X_cat, drop_first = True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)



param_grid = [{
    'kneighborsclassifier__n_neighbors': [ 3, 5, 7, 8, 9, 10],
    'kneighborsclassifier__p': [1, 2],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'kneighborsclassifier__metric': ['minkowski','euclidean','manhattan'],
    'kneighborsclassifier__leaf_size': [3,5,7,11,17,21,25]
}]

imba_pipeline1 = make_pipeline(KNeighborsClassifier())


# Create a grid search was taking a lot of time so we chose Randomized search as there are lot of parameters
# and no. of iterations will get 6 X 2 X 2X4X3X7 for each fold.
gs = RandomizedSearchCV(imba_pipeline1, param_distributions = param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10,
                  verbose=1,
                  n_jobs=2)

# Fit the most optimal model
gs.fit(X_train, y_train)

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
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, best_prediction_knn))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, best_prediction_knn))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, best_prediction_knn)))
print("R2 Score - " , metrics.explained_variance_score(y_test, best_prediction_knn)*100)


# Plot between real and predicted data
plt.figure(figsize=(20,8))
plt.plot(best_prediction_knn[:200], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("KNN Binary Classification")
plt.savefig('plots/xgb_real_pred_bin.png')
plt.show()


#________________________________________________________________________________________________________

#---------------------------------------SVC--------------------------------------------------

pipelineSVC = make_pipeline(SVC(random_state=1, probability=True))


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


imba_pipeline = make_pipeline(SVC(probability=True))

rs_svc = RandomizedSearchCV(imba_pipeline, param_grid_svc, cv=10, scoring = 'accuracy')

y_pred, y_pred_prob = fit_n_pred(rs_svc, X_train, y_train, X_test)
rs_svc.best_params_

evaluate_classification_model(y_test, y_pred)
plotAUC(y_test, y_pred_prob, 'SVM')
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)


# Plot between real and predicted data
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("SVM Binary Classification")
plt.savefig('plots/svm_real_pred_bin.png')
plt.show()

#________________________________________________________________________________________________


