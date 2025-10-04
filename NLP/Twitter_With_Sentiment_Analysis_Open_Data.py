import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



import os
os.chdir('E:/Anukriti/Avrutti Research/Task18 Twitter sentiment analysis/Dataset')

# Reading the dataset
data = pd.read_csv('TwitterDataPrepOpenDataAll.csv', encoding='utf-8') # Read the dataset

df = data.copy()
df.drop('Unnamed: 0', axis=1,inplace=True)

df.columns
len(df)
#df['user'][1]

# Removing duplicates based on text
#df.drop_duplicates(subset='text', inplace=True)

#df2.dropna(subset = ['favorite_count','user_followers_count','user_friends_count','user_favourites_count','retweet_count'], inplace = True)

# Percentage of null values
# The columns with more than 90% of null values are removed because they don't provide any information as most of the information is missing.
# Removing the columns tweet_coord, negativereason_gold and airline_sentiment_gold 
null_perc = df.isnull().sum()/df.shape[0]*100

#List of columns having more than 75% of null values
col_to_drop = null_perc[null_perc>75].keys()
#col_to_drop = col_to_drop.drop(['in_reply_to_screen_name'])


# Dropping columns 
df1 = df.drop(col_to_drop, axis=1)

#checking unique values 
unique = df1.nunique()
# Columns with same values are removed as they cannot be considered as a variable

cols_to_drop = unique[unique==1].keys() 
#cols_to_drop = cols_to_drop.drop([ 'retweet_count', 'favorite_count']) # Not removing retweet_count and favorite_count because these metrics are required for calculating the scores

df2 = df1.drop(cols_to_drop, axis=1)


#Dropping the rows with neutral sentiments as neutral sentiments doesn't display any meaningful attitude of customer towards airline
df2.drop(df2.loc[df['sentiment']=='neutral'].index, inplace=True)


# Finding if there is data for same user name multiple times 
df2['user_name'].nunique() # There are 1106 unique values and total dataframe rows are 1361. Therefore, 1361 - 1106 = 255 repetitive user names
df2.columns
data['user'][0]


#---------------------------------------Checking special characters-----------------------------------------
"""
import string

unwanted = string.ascii_letters + string.punctuation + string.whitespace
print(unwanted)
"""

# remove all string values from a column using regex
df2['user_friends_count'] = df2['user_friends_count'].str.replace('\D','', regex = True)
df2['user_friends_count'] = df2['user_friends_count'].str.replace('\W','', regex = True)
df2['user_favourites_count'] = df2['user_favourites_count'].str.replace('\D','', regex = True)
df2['user_favourites_count'] = df2['user_favourites_count'].str.replace('\W','', regex = True)
df2['user_followers_count'] = df2['user_followers_count'].str.replace('\D','', regex = True)
#df2['user_followers_count'] = df2['user_followers_count'].str.replace('\d.','', regex = True) # Using this will remove all dots present in values including values with decimal values present
df2['user_followers_count'] = df2['user_followers_count'].str.replace('\W','', regex = True)

df2['user_followers_count'].unique()


"""
# Removing unwaned characters
df2['user_followers_count'] = df2['user_followers_count'].str.strip(unwanted)
df2['user_friends_count'] = df2['user_friends_count'].str.strip(unwanted)
df2['user_favourites_count'] = df2['user_favourites_count'].str.strip(unwanted)
#df2['user_followers_count'] = df2['user_followers_count'].str.strip(unwanted)
#df2['user_followers_count'] = df2['user_followers_count'].str.strip(unwanted)
"""


# Removing null values from the dataframe
df2.isnull().sum()
df2.dropna(subset = ['user_followers_count', 'user_friends_count','retweet_count','user_statuses_count', 'user_favourites_count'], inplace = True)
df2.dtypes



# Converting object data types into float data types
#categorical_data = df2.select_dtypes(include = ['object']).copy()
#categorical_data.columns
df2['user_followers_count'] = df2['user_followers_count'].astype(float)
df2['user_friends_count'] = df2['user_friends_count'].astype(float)
df2['user_favourites_count'] = df2['user_favourites_count'].astype(float)



"""
# There are no special characters present in columns with object data types
frequencies = categorical_data.apply(lambda x: x.value_counts()).T.stack()
freqUserFollowerCount = df2['user_followers_count'].apply(lambda x: x.value_counts()).T.stack()
df2.dtypes
print(frequencies)
"""
#-------------------------------------Metrics Creation-----------------------------------------------

# Creating new metrics from existing parameters
#df2['retw_comments_count'] = df2['retweet_count'] + df2['status_count']

# Number of original tweets by all users in the topic. As retweet_count is not considered as variable in dataset so considering it as 0
df2['original_count'] = df2['user_statuses_count'] - df2['retweet_count'] # Count of original tweets by each user
N = df2['original_count'].sum() # Total number of original tweets by all users

# Finding unique users
df2['user_name'].nunique() # There are 1106 unique users and total data points are 1361 indicating same user is present multiple times in dataframe


# As in_reply_to_screen_name is removed due to 97% of null values, so reply_count is not calculated.
"""
# Metric: RP3: Number of users who have replied author’s tweets
reply = df2[df2['in_reply_to_screen_name'].notnull()] # 456 tweets are replies. contain the screen name of the original Tweet's author
#df2reply = reply.groupby('in_reply_to_screen_name', sort=False)['in_reply_to_screen_name'].count()
df2reply = reply.groupby('in_reply_to_screen_name', sort=False)['usr_name'].count() # usr_name are people who have replied. So we counted usr_name. 
reply['in_reply_to_screen_name'].str.count("pravngupta").sum() # Cross-Checking
df2reply = pd.DataFrame(df2reply) # Converting to dataframe
#df2reply.rename(columns = {'in_reply_to_screen_name':'reply_count'}, inplace=True)
df2reply.rename(columns = {'usr_name':'reply_count'}, inplace=True)
df2reply.columns
df3 = pd.merge(df2, df2reply, left_on='screen_name', right_on ='in_reply_to_screen_name', how = 'left') # in_reply_to_screen_name contain the screen name of the original Tweet's author
df3['reply_count'].value_counts()


df2 = df3
"""
#-----------------------------------------------------------------------------------------------------------
"""
df2.dtypes[df2.dtypes == 'int64']
df2['user_followers_count', 'user_friends_count','retweet_count','user_status_count', 'user_favorite_count'] = df2['user_followers_count', 'user_friends_count','retweet_count','user_statuses_count', 'user_favourites_count'].astype(float64)
df2['user_followers_count'] = df2['user_followers_count'].astype('float64')
"""

#-----------------------------------Calculating all scores----------------------------------------------------

# Calculating the top users based on popularity. 
df2['popularity'] = (df2['user_followers_count'] - df2['user_friends_count']) / (df2['user_followers_count'] + df2['user_friends_count'])

# Influence Score
df2['infl_score'] = (df2['retweet_count'] + df2['favorite_count']) / (df2['user_followers_count'] + df2['user_friends_count'])
# favorite_count: Number of users that have marked author’s tweets as favorite (likes)
#df2['infl_score'] = ((df2['retweet_count'] + df2['favorite_count'] + df2['reply_count'])/ (df2['foll_count'] + df2['friends_count'])) + ((df2['retweet_count'] + df2['favorite_count'] + df2['reply_count'])/N)

#df2['actvty_score'] = (df2['status_count'] + df2['favorite_count'])/N
df2['activity_score'] = (df2['user_statuses_count'] + df2['retweet_count'] + df2['user_favourites_count'])/N
# fav_count(Extracted from user variable during data preparation): Number of tweets of other users marked as favorite (liked) by the author

df2['imp_score'] = (df2['popularity'] + df2['infl_score'] + df2['activity_score']) / 3



#___________________________________________________________________________________________________________


# Filtering data with positive sentiments
pos = df2[df2['sentiment'] == 'positive']
neg = df2[df2['sentiment'] == 'negative']


# Working with only positive sentiments data
pos_pop = pos.groupby('user_screen_name', sort = False)['popularity'].agg('sum')
pos_infl = pos.groupby('user_screen_name', sort = False)['infl_score'].agg('sum')
pos_activity = pos.groupby('user_screen_name', sort = False)['activity_score'].agg('sum')
pos_imp = pos.groupby('user_screen_name', sort = False)['imp_score'].agg('sum')

# Converting the series into a dataframe
pos_pop = pd.DataFrame(pos_pop)
pos_infl = pd.DataFrame(pos_infl)
pos_activity = pd.DataFrame(pos_activity)
pos_imp = pd.DataFrame(pos_imp)


# Sorting the values based on popularity
pos_pop['popularity'].sort_values(ascending=False)[:10]
pos_infl['infl_score'].sort_values(ascending=False)[:10]
pos_activity['activity_score'].sort_values(ascending=False)[:10]
pos_imp['imp_score'].sort_values(ascending=False)[:10]


pop_top10 = pd.DataFrame(pos_pop['popularity'].sort_values(ascending=False)[:10])
infl_pop_top10 = pd.DataFrame(pos_infl['infl_score'].sort_values(ascending=False)[:10])
activity_pop_top10 = pd.DataFrame(pos_activity['activity_score'].sort_values(ascending=False)[:10])
imp_top10 = pd.DataFrame(pos_imp['imp_score'].sort_values(ascending=False)[:10])



#--------------------------------Bar Plot for top users based on all scores-----------------------------------------


# Graph to plot top 10 popular people
ax = pop_top10.plot(kind = 'bar', rot=0, colormap = 'Greens_r', figsize = (15,6), fontsize = 15)
ax.set_xlabel('Screen name of users')
ax.set_ylabel('Popularity Score')
plt.xticks(rotation=90)
plt.title("Top 10 users based on popularity score")
plt.savefig('Top10UsersPopularityWithSentimentAnalysisFinalOpenData.png', bbox_inches = 'tight')
plt.show()




# Graph to plot top 10 influencers
ax = infl_pop_top10.plot(kind = 'bar', rot=0, colormap = 'Greens_r', figsize = (15,6), fontsize = 15)
ax.set_xlabel('Screen name of users')
ax.set_ylabel('Influence Score')
plt.xticks(rotation=90)
plt.title("Top 10 influencers based on influence score")
plt.savefig('Top10UsersInfluenceWithSentimentAnalysisFinalOpenData.png', bbox_inches = 'tight')
plt.show()


# Graph to plot top 10 active users
ax = activity_pop_top10.plot(kind = 'bar', rot=0, colormap = 'Greens_r', figsize = (15,6), fontsize = 15)
ax.set_xlabel('Screen name of users')
ax.set_ylabel('Activity Score')
plt.xticks(rotation=90)
plt.title("Top 10 people based on activity score")
plt.savefig('Top10UsersActivityWithSentimentAnalysisFinalOpenData.png', bbox_inches = 'tight')
plt.show()



# Graph to plot top 10 important users
ax = imp_top10.plot(kind = 'bar', rot=0, colormap = 'Greens_r', figsize = (15,6), fontsize = 15)
ax.set_xlabel('Screen name of users')
ax.set_ylabel('CBPI Score')
plt.xticks(rotation=90)
plt.title("Top 10 users based on CBPI (Context Based Positive Influence) score")
plt.savefig('Top10UsersImportanceWithSentimentAnalysisFinalOpenData.png', bbox_inches = 'tight')
plt.show()


#----------------------------------------------------END-----------------------------------------------------------------







