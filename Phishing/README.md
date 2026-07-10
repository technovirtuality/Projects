Aim: To develop an efficient, fast and reliable early Phishing detection model using machine learning which can identify the phishing websites. The purpose is to safeguard user exploitation with the help of an automated phishing website detection system based on machine learning model.

Problem Statement:
1.To identify the phishing URLs or websites
2.To perform binary classification for identifying malicious and legitimate websites using machine learning techniques
3.To evaluate the performance of phishing detection system
4.To select the best performing machine learning model among all models that are being used for phishing detection

Objectives: 
1.To create a classification model using feature extracted from websites and classify as phishing or legitimate websites.
2.To detect important features among all features useful for phishing website detection
3.To improve the accuracy of Phishing detection system using machine learning techniques.
4.To design an ensemble-based XG Boost learning model to avoid the problem of overfitting 
5.To compare and select the best performing model based on performance metric accuracy, precision and AUC-ROC curve
6.To minimize the false alarm rate of model to minimize the wastage of resources in attending the issue.


Dataset: The dataset contains the URLs of websites as features. The target variable has two labels: phishing website and normal represented by 1 and 0 respectively. The small variant dataset named, dataset_small.csv, is used in the system. The total number of samples are 58,645, the number of legitimate website instances (labelled as 0): 27,998, the number of phishing website samples (labelled as 1): 30,647, the total number of features: 111, among them URL based features can be used for analysis.


Design:

The procedure for phishing detection system is performed in two ways as explained below: 
Procedure I: The important features are selected from XG Boost model and selected features are used for building model for binary classification.

<img width="1037" height="675" alt="image" src="https://github.com/user-attachments/assets/979097db-b3e4-43ae-8cce-5e4bf0bb1804" />



Procedure II: Feature selection using principal component analysis approach
The difference between the two procedures is that feature extraction in procedure 1, is performed by the machine learning methods i.e., Random forest and XG Boost whereas in procedure 2, the features are extracted by converting the feature space into the direction of maximum variance and then principal components are selected by applying PCA approach. 

Pre-processing:

Check variable level: All variables are checked for unique levels to figure out those variables which has same value for all samples. The variable with constant value cannot be considered as variables and therefore removed from data. In this way 13 features are removed from data.
Duplicate removal: All duplicated samples are removed from data keeping only first occurrence of that sample. Thus, reducing the number of samples from 58,645 to 57,405.
Outlier removal: Dataset has all numerical variables and there are outliers present in variables. To improve the accuracy, removal of outliers is important so that each variable follow a normal distribution. The IQR method is used for outlier removal from dataset. The dataset has such variables whose Inter quartile range is 0. So, outliers are not removed from these variables because it will lead to deletion of this feature.  There are 28 outlier samples present in data which are removed providing 57,377 samples and 99 features.

Data Balance: The class ratio in data is 52.3% which represents almost balanced data. Therefore, no balancing technique is applied on dataset.

Standardization: There are a lot of deviations in data. To get rid of deviations, the mean of variable is subtracted from each value. We assume that data is normally distributed and converted the normal distribution into standard normal distribution using StandardScaler(). 

Train-test split: The data is split into train-test after applying PCA, in the ratio of 70:30. 

Hyper-parameter Tuning: The 5-fold Randomized search cross-validation is used for determining the optimal hyper-parameters. 

Performance Metric: Maximizing precision will minimize the number of false positives, whereas maximizing the recall will minimize the number of false negatives. Precision is appropriate when minimizing false positives is the focus. In order to reduce false alarm rate, false negative needs to be minimized. Therefore, precision is chosen as performance metric along with accuracy, F1-score and AUC ROC curve. 
