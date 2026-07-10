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

<img width="992" height="731" alt="image" src="https://github.com/user-attachments/assets/6c557a99-c077-4c6a-9e24-ab8a67bc2f12" />


The difference between the two procedures is that feature extraction in procedure 1, is performed by the machine learning methods i.e., Random forest and XG Boost whereas in procedure 2, the features are extracted by converting the feature space into the direction of maximum variance and then principal components are selected by applying PCA approach. 

Pre-processing:

Check variable level: All variables are checked for unique levels to figure out those variables which has same value for all samples. The variable with constant value cannot be considered as variables and therefore removed from data. In this way 13 features are removed from data.
Duplicate removal: All duplicated samples are removed from data keeping only first occurrence of that sample. Thus, reducing the number of samples from 58,645 to 57,405.
Outlier removal: Dataset has all numerical variables and there are outliers present in variables. To improve the accuracy, removal of outliers is important so that each variable follow a normal distribution. The IQR method is used for outlier removal from dataset. The dataset has such variables whose Inter quartile range is 0. So, outliers are not removed from these variables because it will lead to deletion of this feature.  There are 28 outlier samples present in data which are removed providing 57,377 samples and 99 features.

Data Balance: The class ratio in data is 52.3% which represents almost balanced data. Therefore, no balancing technique is applied on dataset.

Standardization: There are a lot of deviations in data. To get rid of deviations, the mean of variable is subtracted from each value. We assume that data is normally distributed and converted the normal distribution into standard normal distribution using StandardScaler(). 

Principal Component Analysis: 

<img width="855" height="430" alt="image" src="https://github.com/user-attachments/assets/4a22aca4-0df6-4c33-bfd4-0b23e938dc55" />

Scree plot: The number of principal components is chosen from the scree plot. Scree plot is a graph where explained variance is plotted for each principal component. We looked for ‘kink’ in Scree plot and flattening of the explained variance. The explained variance decreases as the number of principal components increases and after 10 principal components, explained variance does not vary (increase/decrease) further. Therefore 10 principal components are chosen.

PCA analysis plot: The graph is plotted for cumulative variance explained by principal components. The cumulative variance explained by first 10 principal components is 68%. The cumulative explained variance is 97% provided by first 43 principal components.

Train-test split: The data is split into train-test after applying PCA, in the ratio of 70:30. 

Hyper-parameter Tuning: The 5-fold Randomized search cross-validation is used for determining the optimal hyper-parameters. 

Performance Metric: Maximizing precision will minimize the number of false positives, whereas maximizing the recall will minimize the number of false negatives. Precision is appropriate when minimizing false positives is the focus. In order to reduce false alarm rate, false negative needs to be minimized. Therefore, precision is chosen as performance metric along with accuracy, F1-score and AUC ROC curve. 

Results:

1. Machine Learning Approach:

Feature selection by XGBoost algorithm:
<img width="866" height="452" alt="image" src="https://github.com/user-attachments/assets/bf1c3cc5-c148-462d-84dd-17f3c5d48116" />

Threshold for feature selection: The XGB model is built at each threshold of feature selection scores. The model is then fit to train data and accuracy is obtained at each threshold level. It can be seen from the output that with decreasing threshold feature importance score and increasing number of features, accuracy is initially increased till 95.08% and decreases or remains constant after that. The minimum number of features with maximum accuracy is at threshold score: 0.003 and minimum number of features: 28.

Interpretation: From confusion matrix it is visible that 9710 phishing samples are correctly classified whereas 369 phishing samples are wrongly classified as legitimate websites. Among legitimate samples, 8374 samples are correctly classified as legitimate websites and 491 samples are wrongly classified as phishing samples causing the false alarm.  The best hyperparameters are chosen for selecting the best features. The feature importance score of each variable is shown in graph above. It is visible that accuracy first starts increasing reaches the maximum value and then starts decreasing. The best threshold of feature importance is selected by applying model at each threshold. As XG Boost model provide better accuracy than Random Forest model, we did feature selection from XG Boost model by selecting a threshold of feature importance score.

Results with selected features: XGBoost model:

<img width="887" height="551" alt="image" src="https://github.com/user-attachments/assets/f5fc2d6b-c66c-44fe-896b-fe1dd4eac1b7" />


2. Principal Component Analysis Approach: 

XGBoost Model (n=10): The results obtained by XG Boost model on 10 principal components are shown below:

<img width="902" height="581" alt="image" src="https://github.com/user-attachments/assets/5e09c8e6-e7e2-4c96-9242-cc1aa18a5edd" />


Interpretation: The results shown in figure 8 displays that accuracy achieved by the XG Boost model on selected 10 principal components is 93%. It is evident from the confusion matrix that 8543 samples of phishing attack and 7453 samples of normal websites, are correctly identified by the XG Boost model whereas 567 phishing samples are wrongly classified as normal samples and 651 samples are wrongly classified as phishing attack. 

XG Boost Model(n=43): The selection of 43 PCs provides 97% of cumulative explained variance.

<img width="772" height="587" alt="image" src="https://github.com/user-attachments/assets/173638d3-a8e6-41a5-ba59-1568bb5523b8" />


Interpretation: It is visible from results as shown in figure 9, in classification report there is only 1% of increase in accuracy if number of components is increased from 10 to 43 components. Therefore, to select a time efficient model among them. XGB model with 10 components is a better model. It is evident from the confusion matrix that 8620 samples of phishing attack and 7632 samples of normal websites, are correctly identified by the XG Boost model whereas 490 phishing samples are wrongly classified as normal samples and 472 samples are wrongly classified as phishing attack. 

Conclusion:

The AUC score of XG Boost model without PCA is same as that of XG Boost model with 10 PCs but accuracy and precision of XG Boost model without PCA is highest. The accuracy is 95% and precision score for legitimate and malicious class is 96% and 95% respectively. As precision of proposed model is highest (96%), the model is effective in minimizing the false alarm rate which is a major concern due to the wastage of resources. In this context, XG Boost model with important feature selection is more accurate and time efficient. However, a smaller number of components are needed in PCA selected features as compared to 28 features needed in feature selection with XG Boost but other performance metrics achieved better performance and XG Boost is time efficient algorithm leading to the selection of XG Boost model with important features selected from threshold method of XG Boost algorithm.  
