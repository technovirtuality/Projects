# Importing necessary libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC



# Hyperparameter Tuning
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)


# Importing performance metrics
from sklearn import preprocessing,metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve, plot_confusion_matrix


import os

os.chdir('E:\Anukriti\Advance Data Science-IIT Madras\Hackathon\Housing Prices')

data_xlsx = pd.read_excel('housing_train.xlsx')
data = data_xlsx.copy()

# Dropping id column as it is of not any use
data.drop(columns = 'Id', axis = 1, inplace=True)
# Renaming column names
#data.rename(columns = {'Avg. Area Income':'income','Avg. Area House Age':'house_age','Avg. Area Number of Rooms':'no_of_rooms','Avg. Area Number of Bedrooms':'no_of_bedrooms','Area Population':'pop'}, inplace=True)


# Data Cleaning

# Checking the data types of all variables are correct or not 
info = data.info() # The data type of all variables are correct

# Checking null values present in dataset
null=data.isnull().sum() # No null values are present in data
null.sum() # Total null values present in data are 6965
# There are variables where null values means not applicable. Eg. there is no fireplace or ally for a house. So we cannot fill missing values in tose cases.
# Therefore we are considering all null values of data.

# Checking the number of variables in data
data.shape


#----------------------Check special characters and understanding spread of categorical variables----------------------------------------------

# Extracting categorical variables
categorical_data = data.select_dtypes(include = ['object'].copy())

# Extracting frequencies of each category/level in categorical variables to obtain frequencies of special characters also
frequencies = categorical_data.apply(lambda x: x.value_counts()).T.stack()
print(frequencies)
# No special characters are found in variables


#______________________________________________________________________________________________________

#-------------------------------######Numerical variables-------------------------------------------------

num_data = data.select_dtypes(include = ['int64','float64']) # 37 numerical variables. 37+43(categorical var) = 80
print("Number of numerical variables:", num_data.shape[1])
print("Number of categorical variables:", categorical_data.shape[1])


# Checking correlation among variables
corr = num_data.corr()

plt.figure(figsize = (10,6))
sns.heatmap(data= corr, annot = False)
plt.title('Correlation Matrix for Multi Labels',fontsize=16)
plt.savefig('E:/Anukriti/Advance Data Science-IIT Madras/Hackathon/Housing Prices/fig/correlation_matrix_multi.png')
plt.show()
# All variables have low correlation which indicates that all independent variables are uncorrelated to each other. The variable avg income in area has moderately positive correlation with target variable 'price'.


# Extracting correlation of variables with respect to target variable SalePrice
corr_price = corr['SalePrice'].sort_values(ascending = False).to_frame()
corr_price = abs(corr_price) # Considering absolute values of correlation


# Selecting those variables which have correlation w.r.t. SalePrice more than 0.5
selected = corr_price[corr_price['SalePrice'] > 0.5]
print("Variables with correlation greater than 0.5",selected.index) # Selected variables with more than 0.5 correlation


"""# Selecting variables having correlation greater than threshold
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(corr_price, 0.60)
len(set(corr_features))"""


cols = selected.index[1:] # Saving variable names with high correlation w.r.t SalePrice and dropping SalePrice from selection

#------------------------


#---------------------Relationship of selected numerical variables w.r.t SalePrice-----------------------
for i in range(len(cols)):
    plt.scatter(x = data[cols[i]], y = data['SalePrice'])
    plt.title("Relationship between Sale Price and "+cols[i])
    plt.ylabel("Sale Price of House")
    plt.xlabel(cols[i])
    plt.show()

""" Conclusions:

"""

# Five Point Summary statistics of variables
summary_num = num_data.describe()
summary_cat = categorical_data.describe()



# Outlier detection: Boxplot
for i in range(len(cols)):
    sns.boxplot(y = data[cols[i]])
    plt.title("Boxplot of "+cols[i])
    plt.show()
# No outliers are present in variables: FullBath and TearRemodAdd
# Outliers are present in variables OverallQual, GrLivArea,


# 1. Removing outliers: OverallQual

# Checking summary statistics: mean before removing outliers
data['OverallQual'].describe()

# IQR method for replacing outliers
# Calculating 25th and 75th quartile
q25, q75 = data['OverallQual'].quantile(0.25), data['OverallQual'].quantile(0.75)


# calculating interquartile range
iqr = q75-q25

# Calculating upper and lower whiskers of boxplot
lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr


# Replacing the outliers with median. We did not remove outliers instead replaced to prevent the loss of data.
# We replace by median because these are numerical variables so we can choose mean or median for replacement.
# We chose median because median is not affected by outliers.

data['OverallQual'] = np.where((data['OverallQual'] < lower) | (data['OverallQual'] > upper), data['OverallQual'].median(), data['OverallQual'])

sns.boxplot(y=data['OverallQual'])

# Checking summary statistics: mean after removing outliers
data['OverallQual'].describe() # There is not much change in mean value of variable after removing outliers so we can do replacement of outliers with mean and continue


# 2. Removing outliers: GrLivArea

# IQR method for replacing outliers
# Calculating 25th and 75th quartile
q25, q75 = data['GrLivArea'].quantile(0.25), data['GrLivArea'].quantile(0.75)


# calculating interquartile range
iqr = q75-q25

# Calculating upper and lower whiskers of boxplot
lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr


# Replacing the outliers with median( Median is considered because it is not affected by outliers) to prevent the loss of data
data['GrLivArea'] = np.where((data['GrLivArea'] < lower) | (data['GrLivArea'] > upper), data['GrLivArea'].median(), data['GrLivArea'])

sns.boxplot(y=data['GrLivArea'])
data['GrLivArea'].describe()



# 3. Removing outliers: GarageCars


data['GarageCars'].describe()

# IQR method for replacing outliers
# Calculating 25th and 75th quartile
q25, q75 = data['GarageCars'].quantile(0.25), data['GarageCars'].quantile(0.75)


# calculating interquartile range
iqr = q75-q25

# Calculating upper and lower whiskers of boxplot
lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr


# Replacing the outliers with median( Median is considered because it is not affected by outliers) to prevent the loss of data
data['GarageCars'] = np.where((data['GarageCars'] < lower) | (data['GarageCars'] > upper), data['GarageCars'].median(), data['GarageCars'])

sns.boxplot(y=data['GarageCars'])
data['GarageCars'].describe()




# 4. Removing outliers: GarageArea

# IQR method for replacing outliers
# Calculating 25th and 75th quartile
q25, q75 = data['GarageArea'].quantile(0.25), data['GarageArea'].quantile(0.75)


# calculating interquartile range
iqr = q75-q25

# Calculating upper and lower whiskers of boxplot
lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr


# Replacing the outliers with median( Median is considered because it is not affected by outliers) to prevent the loss of data
data['GarageArea'] = np.where((data['GarageArea'] < lower) | (data['GarageArea'] > upper), data['GarageArea'].median(), data['GarageArea'])

sns.boxplot(y=data['GarageArea'])
data['GarageArea'].describe()



#_________________________________________________________________________________________________________


#-------------------------------------------------SalePrice------------------------------------------------

sns.boxplot(y=data['SalePrice']) # There are lot of outliers in the variable



data['SalePrice'].describe() # Checking statistics before removing outliers

# IQR method for replacing outliers
# Calculating 25th and 75th quartile
q25, q75 = data['SalePrice'].quantile(0.25), data['SalePrice'].quantile(0.75)


# calculating interquartile range
iqr = q75-q25

# Calculating upper and lower whiskers of boxplot
lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr


# Replacing the outliers with median( Median is considered because it is not affected by outliers) to prevent the loss of data
data['SalePrice'] = np.where((data['SalePrice'] < lower) | (data['SalePrice'] > upper), data['SalePrice'].median(), data['SalePrice'])

sns.boxplot(y=data['SalePrice'])
data['SalePrice'].describe()


#__________________________________________________________________________________________________________

#-----------------------------------Removing Duplicates-----------------------------------------

# Checking duplicate samples/rows present in dataset
data.duplicated().sum() # No duplicate samples are present in dataset. 

#____________________________________________________________________________________________________________


# --------------------------------------------Feature Creation-----------------------------------------

# Total Are of house in square foot
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data.drop(columns=['1stFlrSF','2ndFlrSF','TotalBsmtSF'], axis =1, inplace = True)


# Total number of rooms
data['TotalNumberOfRooms'] = data['BedroomAbvGr'] + data['KitchenAbvGr'] + data['TotRmsAbvGrd']
data.drop(columns=['BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd'], axis =1, inplace = True)


# Total number of Bathrooms
data.columns
data['TotalNumberOfBathrooms'] = data['FullBath'] + 0.5*data['HalfBath'] + data['BsmtFullBath'] + 0.5*data['BsmtHalfBath']
data.drop(columns=['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath'], axis =1, inplace = True)


# Total Porche area in square foot
data['TotalPorchArea'] = (data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'])
data.drop(columns=['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch'], axis =1, inplace = True)

#_________________________________________________________________________________________________________

#----------------------------------------Missing values-----------------------------------------------------

null=data.isnull().sum()
null.sum()

# Percentage of missing values

indices = data.isnull().sum().index

percentages = []
for i in indices:
    percentages.append((data[i].isnull().sum() / data[i].shape[0]) * 100)


d = {'Columns':indices, 'Percentage of Null values':percentages}
null_frame = pd.DataFrame(data = d)
# We can see that PoolQC, Fence, MiscFeature, Alley has more than 80% missing values so we will remove them
data.drop(columns = ['PoolQC','Fence','MiscFeature','Alley'], axis=1, inplace=True)


# Filling missing values of street length with street length of neighborhood
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))


data['Electrical'].fillna(data['Electrical'].mode()[0], inplace = True)
data['BsmtQual'].fillna(data['BsmtQual'].mode()[0], inplace = True)

# Null values in following variables indicates non-existence of Basement/Fire Place/Garage not built so we replace it with None
data['BsmtCond'].fillna('None', inplace=True)
data['BsmtExposure'].fillna('None', inplace=True)
data['BsmtFinType1'].fillna('None', inplace=True)
data['BsmtFinType2'].fillna('None', inplace=True)
#data['BsmtFinType1'].fillna('None', inplace=True)
data['FireplaceQu'].fillna('None', inplace=True)
data['GarageType'].fillna('None', inplace=True)
data['GarageFinish'].fillna('None', inplace=True)
data['GarageQual'].fillna('None', inplace=True)
data['GarageYrBlt'].fillna('None', inplace=True)
data['GarageCond'].fillna('None', inplace=True)


# We fill the missing values with modal value because MasVnrType is a categorical variable( so mode statistic is used) and Masonry Veneer is external layer of masonry, typically made of brick, stone or manufactured stone 
# so it can't be empty
data['MasVnrType'].fillna(data['MasVnrType'].mode()[0], inplace = True)

#We fill the missing values with median value because MasVnrArea is a numerical variable( so median statistic is used)
data['MasVnrArea'].fillna(data['MasVnrArea'].median(), inplace = True)


# Checking count of null values after filling them
null=data.isnull().sum()
null.sum()

#___________________________________________________________________________________________________________

#-----------------------------------------EDA--------------------------------------------------------------

"""col = data.columns[:6] # Extracting variable names and selecting only numerical variables
col = list(col)# Converting col into list type


# Plotting boxplots of all numerical variables
for i in range(len(col)):
    sns.boxplot(y=data[col[i]])
    plt.show()
# We observed that except variable Average No. of bedrooms in the area, have lot of outliers.
# Now we analyze each variable separately."""



#---------------------------------------------PCA---------------------------------------------------------

data = num_data.drop(columns= 'SalePrice', axis=1)
y = data['SalePrice']
input_columns = data.columns # Extracting column names and saving in variable


# Standardizing the variables before applying PCA
scaler = StandardScaler()
input_data = scaler.fit_transform(data)
input_data = pd.DataFrame(input_data, columns = input_columns)


pca = PCA(n_components = 0.85) # 90% variance
pca_1 = pca.fit_transform(input_data)

print(pca.explained_variance_ratio_*100)



# Percentage of cumulative variance 
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
var


# Plot showing the amount of cumulative variation explained with corresponding number of components 
plt.ylabel('% Variance Explained')
plt.xlabel('# of Principal Components')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)
plt.show()


#__________________________________________________________________________________________________________


#------------------------------Standardization and Dummy encoding of categorical variables----------------

X_num = data.select_dtypes(include = ['int64','float64']).copy()
X_num.drop(columns = ['SalePrice'], axis=1, inplace = True)
X_cat = data.select_dtypes(include = ['object']).copy()
y = data['SalePrice']

# Standardization of numerical variables
scaler = StandardScaler()
num_data_scaled = scaler.fit_transform(X_num) # Standardizing the numerical variables
num_data_scaled = pd.DataFrame(num_data_scaled) # Converting to DataFrame
num_data_scaled.columns = X_num.columns # Assigned column names to dataframe


# One Hot Encoding of categorical variables and dropped first column (with first category) as remaining columns are sufficient to determine that removed category
data.info() 
X_categorical = pd.get_dummies(X_cat, drop_first = True)

# Concatenating the numerical and categorical variables into one data frame
#X = pd.concat([num_data_scaled.reset_index(drop=True), dummies_cat_data.reset_index(drop=True)], axis=1, ignore_index=True)
X = pd.concat([num_data_scaled, X_categorical], axis=1)



#_______________________________________________________________________________________________________


# Feature Importance using Random Forest Regressor


clf = RandomForestRegressor()

# Fit the model
clf.fit(X,y)

# Set importance features
importance = clf.feature_importances_



# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(topFI))], importance)
plt.show()


# Plot feature importance
plt.rcParams["figure.figsize"] = (10, 10)

y_axis = topFI['Features']
x_axis = topFI['Feature Importance']
plt.barh(y_axis, x_axis)
plt.title('Plot FI')
plt.ylabel('Features')
plt.xlabel('Feature Importance')
plt.gca().invert_yaxis()



d = {'Features':X.columns, 'Feature Importance': clf.feature_importances_}
df = pd.DataFrame(d)

# Display the top 30 feature based on feature importance
topFI = df.sort_values(by='Feature Importance', ascending=False).head(30)


#___________________________________________________________________________________________________________



#--------------------------Feature importance using Linear Regression----------------------------

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=3)


model = LinearRegression()

model.fit(X, y)


# Measure for multicollinearity

def calculateVIF(data):
    features = list(data.columns)
    num_features = len(features)
    
    model = LinearRegression()
    
    result = pd.DataFrame(index = ['VIF'], columns = features)
    
    result = result.fillna(0)
    
    for i in range(num_features):
        x_features = features[:]
        y_features = features[i]
        x_features.remove(y_features)
        model.fit(data[x_features], data[y_features])
        result[y_features] = 1/(1-model.score(data[x_features], data[y_features]))
    return result


vif_val = calculateVIF(X_train)
print(vif_val)

# As VIF value for all independent variables is less than 5. So no multicollinearity is present in independent variables.


#__________________________________________________________________________________________________________

"""
Conclusions:
    1. We did not use PCA for feature selection because PCA provides the principal components which are projection of original variables(in the direction of maximum variance)
       but we want to use original variables.
    2. PCA is typically used for numerical variables only and our data has majority of categorical variables.















