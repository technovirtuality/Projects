import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



import os
os.chdir('...Twitter sentiment analysis/Dataset')

# Reading the dataset
#data = pd.read_csv('TwitterDataPrepBigTech2Data.csv', encoding='utf-8') # Read the dataset
data = pd.read_csv('TwitterDataPrep6.csv', encoding='utf-8') # Read the dataset

data.head(10)
df = data.copy()
df.drop('Unnamed: 0', axis=1,inplace=True)

df.columns
len(df)
df['user'][1]



# Percentage of null values
# The columns with more than 90% of null values are removed because they don't provide any information as most of the information is missing.
# Removing the columns tweet_coord, negativereason_gold and airline_sentiment_gold 
null_perc = df.isnull().sum()/df.shape[0]*100

#List of columns having more than 75% of null values
col_to_drop = null_perc[null_perc>75].keys()

# Dropping columns 
df1 = df.drop(col_to_drop, axis=1)

#checking unique values 
unique = df1.nunique()
# Columns with same values are removed as they cannot be considered as a variable

cols_to_drop = unique[unique==1].keys() 

df2 = df1.drop(cols_to_drop, axis=1)


#Dropping the rows with neutral sentiments as neutral sentiments doesn't display any meaningful attitude of customer towards airline
df2.drop(df2.loc[df['sentiment']=='neutral'].index, inplace=True)


# Finding if there is data for same user name multiple times 
df2['usr_name'].nunique() # There are 1106 unique values and total dataframe rows are 1361. Therefore, 1361 - 1106 = 255 repetitive user names


#-------------------------------------Metrics Creation-----------------------------------------------

# Creating new metrics from existing parameters
#df2['retw_comments_count'] = df2['retweet_count'] + df2['status_count']

# Number of original tweets by all users in the topic
df2['original_count'] = df2['status_count'] - df2['retweet_count'] # Count of original tweets by each user
N = df2['original_count'].sum() # Total number of original tweets by all users

# Finding unique users
df2['usr_name'].nunique() # There are 1106 unique users and total data points are 1361 indicating same user is present multiple times in dataframe


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

#---------------------------------------Overall Sentiment-----------------------------------------------------

df2['screen_name'].nunique()  # There are 266 unique samples where same user with multiple tweets are present in dataset

# Extracting columns from original dataframe tw_list
user_sentiment = df2[['screen_name','sentiment']] # There are total 1361 cases of twitter users in dataframe


# Grouping according to screen name so that multiple tweets by same users can be considered
user_final = user_sentiment.groupby(['screen_name','sentiment']).size().sort_values()   # There are 278 cases including duplicates in it
user_final.nunique()  # There are 266 unique users same as in original dataframe df2
user_final.to_csv('user_final.csv')



user_final = user_final.to_frame().reset_index()
user_final = user_final.rename(columns = {0: 'count'})
user_final.columns

# Resolving cases with tie values
# There are cases with multiple sentiments with tie up values. Considering cases with tie values
dup = user_final[user_final.duplicated(subset = 'screen_name', keep = False)] # There are 23 duplicated cases in the dataset
dup.nunique() # There are 11 unique cases of users with duplicated sentiments/tweets


#Considering neutral sentiment: Adding count of neutral sentiments with count of positive sentiments and  calculating overall sentiments
pos = dup.loc[(dup['sentiment'] == 'positive') | (dup['sentiment'] == 'neutral')] # There are 16 cases with positive and neutral sentiments


# Adding count of positive and neutral sentiments for each user/screen name
positive = pos.groupby(['screen_name'])['count'].sum() # There are total 11 cases with positive sentiments

# Converting the series data into dataframe
positive = pd.DataFrame(positive)

# Resolving cases with negative sentiments
neg = dup.loc[dup['sentiment'] == 'negative'] # There are 7 cases with negative sentiment in data


# Merging cases with positive (positive + neutral) and negative sentiments
df_common = positive.merge(neg, on = 'screen_name', how = 'outer')

# Replacing nan values with 0
df_common = df_common.fillna(0) 
#After merging there are 11 cases of user among them 7 cases have both positive and negative sentiments


#After adding neutral and positive sentiments, there are cases tie values i.e. equal count of positive and negative sentiment values. Resolving such cases by considering them as neutral sentiment (overall)
pos1 = df_common.loc[df_common['count_x'] > df_common['count_y'] ] # count_x are positive sentiments count; count_y is negative sentiments count
# There are 5 users with positive sentiments

# pos1 has all positive sentiment cases, so replacing other sentiments(negative or 0(for nan values)) to positive for these users
pos1['sentiment'] = pos1['sentiment'].replace('negative', 'positive')
pos1['sentiment'] = pos1['sentiment'].replace(0, 'positive')

pos1.drop(['count_x','count_y'], axis = 1, inplace = True)

# Resolving negative sentiment cases
neg1 = df_common.loc[df_common['count_y'] > df_common['count_x']] # count_x are positive sentiments count; count_y is negative sentiments count
neg1.drop(['count_x','count_y'], axis = 1, inplace = True) # there are 2 cases user with negative sentiments


#Handling Neutral sentiments
neu1 = df_common.loc[df_common['count_x'] == df_common['count_y']]

neu1['sentiment'] = neu1['sentiment'].replace('negative', 'neutral')

neu1.drop(['count_x','count_y'], axis = 1, inplace = True) # There are 4 users with neutral sentiments
# Total cases are pos:5 +negative:2 and neutral:4 = 11 users 

df_final = pd.concat([pos1,neg1, neu1], axis = 0)


df_final


# Replacing the tie cases (positive, negative and neutral) with their respective sentiments
postiecases = set(pos1['screen_name'].unique())
lspostie = list(postiecases)



# Extracting data for users in df_final
postiedf = df2.loc[df2['screen_name'].isin(postiecases)]

# Replacing sentiments of these cases with positive
postiedf.replace('negative','positive', inplace = True)
postiedf.replace('neutral','positive', inplace = True)


# Replacing the tie cases (positive, negative and neutral) with their respective sentiments
negtiecases = set(neg1['screen_name'].unique())
lsnegtie = list(negtiecases)

# Extracting data for users in df_final
negtiedf = df2.loc[df2['screen_name'].isin(lsnegtie)]

# Replacing sentiments of these cases with negative
negtiedf.replace('positive','negative', inplace = True)
negtiedf.replace('neutral','negative', inplace = True)


# Neutral: Replacing the tie cases (positive, negative and neutral) with their respective sentiments
neutiecases = set(neu1['screen_name'].unique())
lsneutie = list(neutiecases)

# Extracting data for users in df_final
neutiedf = df2.loc[df2['screen_name'].isin(lsneutie)]

# Replacing sentiments of these cases with negative
neutiedf.replace('positive','neutral', inplace = True)
neutiedf.replace('negative','neutral', inplace = True)



tiedf = pd.concat([postiedf,negtiedf, neutiedf], axis = 0) # There are 31 tie cases including duplicated users with multiple tweets


# Extracting non tie cases
names = df2['screen_name'].unique()
total = set(names)

tienames = dup['screen_name'].unique()
tiecases = set(tienames)

nontiecases = total - tiecases # i.e. 266-11 = 255

ls = list(nontiecases)

nontiedf = df2.loc[df2['screen_name'].isin(ls)]


df_overall_sentiment = pd.concat([tiedf, nontiedf], axis = 0) # There are 31 tie cases including duplicated users with multiple tweets

df_overall_sentiment.to_csv('df_overall_sentiment_Airlines.csv')


#_______________________________________________________________________________________________________________


#-----------------------------------Calculating all scores----------------------------------------------------

# Calculating the top users based on popularity. 
df_overall_sentiment['popularity'] = (df_overall_sentiment['foll_count'] - df_overall_sentiment['friends_count']) / (df_overall_sentiment['foll_count'] + df_overall_sentiment['friends_count'])

# Influence Score
df_overall_sentiment['infl_score'] = (df_overall_sentiment['retweet_count'] + df_overall_sentiment['favorite_count'] + df_overall_sentiment['reply_count'])/ (df_overall_sentiment['foll_count'] + df_overall_sentiment['friends_count'])
# favorite_count: Number of users that have marked author’s tweets as favorite (likes)
#df2['infl_score'] = ((df2['retweet_count'] + df2['favorite_count'] + df2['reply_count'])/ (df2['foll_count'] + df2['friends_count'])) + ((df2['retweet_count'] + df2['favorite_count'] + df2['reply_count'])/N)

#df2['actvty_score'] = (df2['status_count'] + df2['favorite_count'])/N
df_overall_sentiment['activity_score'] = (df_overall_sentiment['status_count'] + df_overall_sentiment['reply_count'] + df_overall_sentiment['retweet_count'] + df_overall_sentiment['fav_count'])/N
# fav_count(Extracted from user variable during data preparation): Number of tweets of other users marked as favorite (liked) by the author

df_overall_sentiment['imp_score'] = (df_overall_sentiment['popularity'] + df_overall_sentiment['infl_score'] + df_overall_sentiment['activity_score']) / 3



#___________________________________________________________________________________________________________


# Filtering data with positive sentiments
pos = df_overall_sentiment[df_overall_sentiment['sentiment'] == 'positive']
neg = df_overall_sentiment[df_overall_sentiment['sentiment'] == 'negative']


# Working with only positive sentiments data
pos_pop = pos.groupby('screen_name', sort = False)['popularity'].agg('sum')
pos_infl = pos.groupby('screen_name', sort = False)['infl_score'].agg('sum')
pos_activity = pos.groupby('screen_name', sort = False)['activity_score'].agg('sum')
pos_imp = pos.groupby('screen_name', sort = False)['imp_score'].agg('sum')

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
plt.savefig('Top10UsersPopularityWithSentimentAnalysisAirlines.png', bbox_inches = 'tight')
plt.show()




# Graph to plot top 10 influencers
ax = infl_pop_top10.plot(kind = 'bar', rot=0, colormap = 'Greens_r', figsize = (15,6), fontsize = 15)
ax.set_xlabel('Screen name of users')
ax.set_ylabel('Influence Score')
plt.xticks(rotation=90)
plt.title("Top 10 influencers based on influence score")
plt.savefig('Top10UsersInfluenceWithSentimentAnalysisAirlines.png', bbox_inches = 'tight')
plt.show()


# Graph to plot top 10 active users
ax = activity_pop_top10.plot(kind = 'bar', rot=0, colormap = 'Greens_r', figsize = (15,6), fontsize = 15)
ax.set_xlabel('Screen name of users')
ax.set_ylabel('Activity Score')
plt.xticks(rotation=90)
plt.title("Top 10 people based on activity score")
plt.savefig('Top10UsersActivityWithSentimentAnalysisAirlines.png', bbox_inches = 'tight')
plt.show()



# Graph to plot top 10 important users
ax = imp_top10.plot(kind = 'bar', rot=0, colormap = 'Greens_r', figsize = (15,6), fontsize = 15)
ax.set_xlabel('Screen name of users')
ax.set_ylabel('CBPI Score')
plt.xticks(rotation=90)
plt.title("Top 10 users based on CBPI (Context Based Positive Influence) score")
plt.savefig('Top10UsersImportanceWithSentimentAnalysisAirlines.png', bbox_inches = 'tight')
plt.show()


#----------------------------------------------------END-----------------------------------------------------------------























