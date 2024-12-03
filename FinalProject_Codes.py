#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:04:30 2024

@author: roland
"""


#pip install render


import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
import re
from PIL import Image
from textblob import TextBlob
import plotly.express as px
import plotly.offline as pyo
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import render
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                             roc_auc_score, precision_score, classification_report)


# Set the directory path (replace with your desired path)
directory = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 1/Final Project'

# Change the working directory
os.chdir(directory)

# Verify that the working directory has changed
print("Current working directory:", os.getcwd())


##============== DATA PREPARATION  ==============

data = pd.read_csv('DisneylandReviews.csv', encoding='latin-1')
data.head(10)
data.shape
data.isnull().sum()

data.drop_duplicates(subset='Review_Text', inplace=True, keep='first')

data['Branch'].value_counts(normalize=True) * 100
data.nunique()

duplicates = data.duplicated(subset='Review_Text', keep='first')
duplicates

# Remove duplicates and keep the first occurrence
data_cleaned = data[~duplicates]

# Reset index after removing duplicates
data_cleaned.reset_index(drop=True, inplace=True)
data_cleaned.shape


# Output the number of duplicates found and a preview of the cleaned data
print(f"Number of duplicates removed: {duplicates.sum()}")
print(data_cleaned.head(100))


# Ensure 'Year_month' is treated as a string before splitting
data_cleaned['Year_Month'] = data_cleaned['Year_Month'].astype(str)

# Split 'Year_month' to create 'Year' and 'Month' columns
data_cleaned[['Year', 'Month']] = data_cleaned['Year_Month'].str.split('-', expand=True)

# Convert 'Year' and 'Month' to integers for easier analysis
data_cleaned['Year'] = data_cleaned['Year']#astype(int)
data_cleaned['Month'] = data_cleaned['Month']#.astype(int)

data_cleaned.drop(columns = ['Year_Month'], inplace = True)

# Preview the updated data
print(data_cleaned[['Year', 'Month']].head())

data_cleaned.head(10)


# Function to determine the quarter based on the month
def get_quarter(Month):
    if Month in [1, 2, 3]:
        return 1
    elif Month in [4, 5, 6]:
        return 2
    elif Month in [7, 8, 9]:
        return 3
    else:
        return 4

# Apply the function to create a new 'Quarter' column
data_cleaned['Quarter'] = data_cleaned['Month'].apply(get_quarter)

# Preview the updated data
print(data_cleaned[['Month', 'Quarter']].head(10))



data_cleaned['Review_Text'].values[2:3]


# Create a new variable for review length using the 'Review_Text' column
# The length is calculated as the number of characters in each review
data_cleaned['Review_Length'] = data_cleaned['Review_Text'].apply(len)

# Preview the updated data
print(data_cleaned[['Review_Text', 'Review_Length']].head())



########### Plotting the number of Reviews by Branches
# Group by 'branch' column and count the number of reviews for each branch
branch_review_counts = data_cleaned['Branch'].value_counts()

# Calculate the percentage for each branch
total_reviews = branch_review_counts.sum()
branch_percentages = (branch_review_counts / total_reviews) * 100

# Create a bar chart
plt.figure(figsize=(10, 6))
ax = branch_review_counts.plot(kind='bar', color='green')

# Add labels and title
plt.title('Number of Reviews by Branch', fontsize=14)
plt.xlabel('Branch', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)

# Annotate the bar chart with percentages
for i, v in enumerate(branch_review_counts):
    ax.text(i, v + 5, f'{v} ({branch_percentages[i]:.1f}%)', ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=0)
# Show the plot
plt.tight_layout()
plt.show()



#=========== GETTING SENTIMENT ANALYSIS FEATURES =============
features = data_cleaned['Review_Text'].values

processed_features = []

# Loop through each review and clean the text
for sentence in range(len(features)):
    # Remove URLs
    processed_feature = re.sub(r'(https?://\S+)', '', str(features[sentence]))
    
    # Remove special characters
    processed_feature = re.sub(r'\W', ' ', processed_feature)
    
    # Remove single characters
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    
    # Remove single characters at the start
    processed_feature = re.sub(r'^\s*[a-zA-Z]\s+', ' ', processed_feature)
    
    # Replace multiple spaces with a single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    
    # Remove prefixed 'b' if present
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    
    # Convert text to lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

# Create a new DataFrame with the cleaned reviews
cleaned_review_data = pd.DataFrame()
cleaned_review_data['reviews'] = processed_features


# Define functions to calculate subjectivity and polarity using TextBlob
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Apply the functions to generate 'Subjectivity' and 'Polarity' columns
cleaned_review_data['Subjectivity'] = cleaned_review_data['reviews'].apply(get_subjectivity)
cleaned_review_data['Polarity'] = cleaned_review_data['reviews'].apply(get_polarity)

# Preview the DataFrame
print(cleaned_review_data.head())


##========= POLARITY CONDITIONS
# Define the conditions for classifying polarity
polarity_conditions = [
    (cleaned_review_data['Polarity'] < 0),
    (cleaned_review_data['Polarity'] == 0),
    (cleaned_review_data['Polarity'] > 0)
]

# Define the corresponding labels for each condition
labels = ['Negative', 'Neutral', 'Positive']

# Use numpy.select to apply the conditions and labels efficiently
cleaned_review_data['Analysis'] = np.select(polarity_conditions, labels)

# Preview the DataFrame
print(cleaned_review_data[['Polarity', 'Analysis']].head(10))



# Copy the 'Analysis' column from cleaned_review_data to a new column named 'sentiment' in cleaned_data
data_cleaned['sentiment'] = cleaned_review_data['Analysis']

# Create a new 'Sentiment' column in df based on the 'Rating' values:
# If 'Rating' is less than 3, classify it as 'Negative'; otherwise, set it to NaN
data_cleaned['Sentiment'] = data_cleaned['Rating'].apply(lambda x: 'Negative' if x < 3 else np.nan)

# Fill in the NaN values in 'Sentiment' with the values from the 'sentiment' column
data_cleaned['Sentiment'] = data_cleaned['Sentiment'].fillna(data_cleaned['sentiment'])


# Copy the cleaned review text from cleaned_review_data to a new column named 'Reviews_Text' in data_cleaned
data_cleaned['Reviews_Text'] = cleaned_review_data['reviews']


# Drop the 'sentiment' and 'Review_ID' columns from df
data_cleaned = data_cleaned.drop(['sentiment', 'Review_ID'], axis=1)

data_cleaned.columns.tolist()

data_cleaned[['Rating','Sentiment', 'Review_Length']].head(10)
data_cleaned.head()


### Making a copy of the cleaned data
data_cleaned_copy = data_cleaned.copy()



#============== EXPLORATORY DATA ANALYSIS ==============


##============= Reviews by years =======================
# Set the style for seaborn
sns.set_style("whitegrid")

# Aggregate the 'review length' by 'year' and plot
aggregated_data = data_cleaned.groupby('Year', as_index=False)['Review_Length'].sum()

# Remove the last row from df3
aggregated_data = aggregated_data.drop(aggregated_data.index[-1])

# Plot the data
plt.figure(figsize=(14, 7))
sns.lineplot(x='Year', y='Review_Length', data=aggregated_data, marker='o', 
             color='black', label='Review Length by Year', linewidth=2.0)
plt.title("Yearly Review Length Trend")
plt.xlabel("Year")
plt.ylabel("Total Review Length")
plt.legend()
plt.show()


# ================== Branches per year ===============
# Set the style for seaborn
sns.set(style="white")

# Filter out rows where the 'Year' column is missing
data_filtered = data_cleaned[data_cleaned['Year'] != 'missing']

# Plot the data
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', hue='Branch', data=data_filtered, palette='Set2')
plt.title("Review Counts of Branches Per Year")
plt.xlabel("Year")
plt.ylabel("Review Counts")
plt.legend()
plt.show()


data_cleaned[['Month']].value_counts()




# Set up the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot for "Quarter By Ratings"
sns.countplot(data=data_cleaned, x='Quarter', hue='Rating', palette='gnuplot', ax=axes[0])
axes[0].set_title('Quarter By Ratings')

# Plot for "Quarter per Reviews"
sns.countplot(data=data_cleaned, x='Quarter', palette='gist_heat', ax=axes[1])
axes[1].set_title('Quarter Per Reviews')

# Adjust layout for better appearance
plt.tight_layout()
plt.show()



# ================= Country per reviews =================

import plotly.io as pio
# Set the renderer to browser
pio.renderers.default = 'browser'
# Group the data by 'Reviewer_Location' and count the number of reviews per branch
fig_df = data_cleaned.groupby('Reviewer_Location', as_index=False)['Branch'].count().sort_values(by='Branch', ascending=False)
print(fig_df.head())  # Check grouped data output

# Create the choropleth map
fig = px.choropleth(
    fig_df,
    locations='Reviewer_Location',
    locationmode='country names',
    color='Branch',
    title='Country - Reviews',
    color_continuous_scale='portland',
    hover_data={'Branch': True}
)

# Hide the color axis scale
fig.update_coloraxes(showscale=False)

# Display the figure
fig.show()

fig.write_html("plot.html")



# ================= Low Ratings By Country =================
# Group by Reviewer_Location and calculate the mean of ratings, then sort and select top 10
top_locations = data_cleaned.groupby('Reviewer_Location', as_index=False)['Rating'].mean().sort_values(by='Rating').head(10)

# Apply background gradient styling to the Rating column
top_locations.style.background_gradient(cmap='autumn', subset=['Rating'])


#================= Rates-Year ==================
# Group by 'year' and 'Rating' to sum the ratings for each group

one_star_rating = data_cleaned.loc[data_cleaned['Rating']==1]
one_star_rating=one_star_rating.groupby('Year',as_index=False).agg({'Rating':'sum'})
one_star_rating = one_star_rating.drop(one_star_rating.index[-1])

#== 
two_star_rating = data_cleaned.loc[data_cleaned['Rating']==2]
two_star_rating = two_star_rating.groupby('Year',as_index=False).agg({'Rating':'sum'})
two_star_rating = two_star_rating.drop(two_star_rating.index[-1])

#== 
three_star_rating = data_cleaned.loc[data_cleaned['Rating']==3]
three_star_rating = three_star_rating.groupby('Year',as_index=False).agg({'Rating':'sum'})
three_star_rating = three_star_rating.drop(three_star_rating.index[-1])

#=== 
four_star_rating = data_cleaned.loc[data_cleaned['Rating']==4]
four_star_rating = four_star_rating.groupby('Year',as_index=False).agg({'Rating':'sum'})
four_star_rating = four_star_rating.drop(four_star_rating.index[-1])

#===
five_star_rating = data_cleaned.loc[data_cleaned['Rating']==5]
five_star_rating = five_star_rating.groupby('Year',as_index=False).agg({'Rating':'sum'})
five_star_rating = five_star_rating.drop(five_star_rating.index[-1])

#============== 
sns.set_style("darkgrid")
plt.figure(figsize=(14,7))
plt.plot(one_star_rating['Year'] ,one_star_rating['Rating'],color='red' , marker='o',label='1 Star') 
plt.plot(two_star_rating['Year'] , two_star_rating['Rating'],color='blue',marker='*',label='2 Star')  
plt.plot(three_star_rating['Year'] ,three_star_rating['Rating'],color='orange',marker='+',label='3 Star') 
plt.plot(four_star_rating['Year'] ,four_star_rating['Rating'],color='black',marker='+',label='4 Star') 
plt.plot(five_star_rating['Year'] ,five_star_rating['Rating'],color='green',marker='+',label='5 Star') 
plt.legend()


#================ Rates-Branch ====================
sns.set(style="white")
plt.figure(figsize=(10, 5))
sns.boxplot(x='Branch', y='Rating', data=data_cleaned, palette='Set2')
plt.title('Boxplot of Ratings by Branch')
plt.xlabel('Branch')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()


#================== Sentiments-Branches ===============
plt.figure(figsize=(10, 6))
sns.countplot(data=data_cleaned, x='Branch', hue='Sentiment', palette='magma')
plt.title('Count of Sentiment by Branch')
plt.xlabel('Branch')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.show()



# Calculate the total count for each branch
# Group the data by 'Branch' and 'Sentiment', then count occurrences
sentiment_counts = data_cleaned.groupby(['Branch', 'Sentiment']).size().reset_index(name='count')

# Group the data by 'Branch' and calculate the total count of reviews for each branch
branch_totals = data_cleaned.groupby('Branch').size().reset_index(name='total_count')

# Merge the sentiment counts with the branch totals
merged = pd.merge(sentiment_counts, branch_totals, on='Branch')

# Calculate the percentage of each sentiment for each branch
merged['percentage'] = (merged['count'] / merged['total_count']) * 100

# Display the result
print(merged[['Branch', 'Sentiment', 'percentage']])


# Define the percentages manually
percentages = [88.523909, 87.524477, 80.234776, 
               10.904366, 11.604658, 19.413059, 
               0.571726, 0.870865, 0.352164]

# Create a countplot
plt.figure(figsize=(10,6))
ax = sns.countplot(data=data_cleaned, x='Branch', hue='Sentiment', palette='magma')

# Label each bar with the corresponding percentage
counter = 0
for p in ax.patches:
    height = p.get_height()
    width = p.get_width()
    x_position = p.get_x() + width / 2
    y_position = height + 5  # Place label slightly above the bar
    
    # Use the predefined percentages list to label the bars
    percentage = percentages[counter]
    
    # Add the text label
    ax.text(x_position, y_position, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=10)
    counter += 1  # Move to the next percentage

# Remove y-axis labels by setting them to an empty list
ax.set_yticklabels([])

plt.title('Sentiment Percentages by Branch')
plt.xlabel('Branch')
plt.ylabel('Sentiment Counts')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



######### Sentiment by Year

data_filtered = data_cleaned[data_cleaned['Year'] != 'missing']
# Set plot size and style
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Create countplot for sentiments by year
sns.countplot(
    data=data_filtered,
    x='Year',
    hue='Sentiment',
    palette='inferno'
)

# Add title and labels
plt.title('Sentiments by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)

# Adjust legend and ticks
plt.legend(title='Sentiment', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()



#================== All Sentiment Wordcloud ===============

from PIL import Image
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt

# Combine all review text into a single string
Reviews_Text = " ".join(data_cleaned['Reviews_Text'].astype(str))

# Load and process the mask image
mask_image = np.array(Image.open("mickey2.jpg"))

# Ensure the mask is binary (black and white) for proper word cloud fitting
mask_image = np.where(mask_image > 128, 255, 1)

# Generate the word cloud with the Mickey mask
word_cloud = WordCloud(
    background_color='white',
    mask=mask_image,
    mode='RGB',
    max_words=1000,
    contour_width=1,
    contour_color='black',
    colormap='flag'
).generate(Reviews_Text)

# Display the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')  # Hide axes for better display
plt.tight_layout(pad=0)  # Remove padding
plt.show()


#================== Positive Sentiment Wordcloud ===============

# Filter the dataset for positive sentiments
data_like = data_cleaned[data_cleaned['Sentiment'] == 'Positive']

# Combine all positive reviews into a single string
reviews_text = " ".join(review for review in data_like['Reviews_Text'].astype(str))

# Load and process the mask image
mask_image = np.array(Image.open("thumbsup2.jpg"))

# Ensure the mask is binary (black and white) for proper word cloud fitting
mask_image = np.where(mask_image > 128, 255, 1)

# Generate the word cloud with the Mickey mask
word_cloud = WordCloud(
    background_color='white',
    mask=mask_image,
    mode='RGB',
    max_words=1000,
    contour_width=1,
    contour_color='black',
    colormap='flag'
).generate(Reviews_Text)

# Display the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')  # Hide axes for better display
plt.tight_layout(pad=0)  # Remove padding
plt.show()


#================== Negative Sentiment Wordcloud ===============

# Filter the dataset for positive sentiments
data_like = data_cleaned[data_cleaned['Sentiment'] == 'Negative']

# Combine all positive reviews into a single string
reviews_text = " ".join(review for review in data_like['Reviews_Text'].astype(str))

# Load and process the mask image
mask_image = np.array(Image.open("down5.jpg"))

# Ensure the mask is binary (black and white) for proper word cloud fitting
mask_image = np.where(mask_image > 128, 255, 1)

# Generate the word cloud with the Mickey mask
word_cloud = WordCloud(
    background_color='white',
    mask=mask_image,
    mode='RGB',
    max_words=1000,
    contour_width=0.5,
    contour_color='black',
    colormap='flag'
).generate(Reviews_Text)

# Display the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')  # Hide axes for better display
plt.tight_layout(pad=0)  # Remove padding
plt.show()

data_cleaned['Sentiment'].value_counts(normalize=True) * 100



################################################################################
#================================== ANALYSIS ===================================
################################################################################

# Removing the Neutral reviews
# Remove rows where Sentiment is 'Neutral'
final_data = data_cleaned.copy()

final_data = final_data[final_data['Sentiment'] != 'Neutral']

# Reset the index (optional, to maintain a clean index after dropping rows)
final_data.reset_index(drop=True, inplace=True)


# storing both Reviews and Labels in lists
review_data, labels = list(data_cleaned['Reviews_Text']), list(data_cleaned['Sentiment'])


final_data.head()
final_data[['Rating','Sentiment', 'Review_Length']].head(10)


# Encoding labels to o and 1
#labelencoder = LabelEncoder()
#final_data['response1'] = labelencoder.fit_transform(final_data['Sentiment'])


final_data.loc[:,'response'] = final_data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

#sentiment_mapping = {'Positive': 1, 'Negative': 0, 'Neutral': 2}
#final_data['response'] = final_data['Sentiment'].map(sentiment_mapping)



final_data[['Rating','Sentiment', 'response']].tail(10)
final_data[['Sentiment', 'response']].value_counts(normalize=True) * 100  #nunique()


# converting review texts to lower case
final_data['Review_Text'] = final_data['Review_Text'].str.lower()
final_data['Review_Text'].head()


# removing stopwords
nltk.download('stopwords')
nltk.download('punkt')
def RemoveStopWords(input_text):
    StopWordsList = stopwords.words('english')
    # Words that might indicate some sentiments are assigned to
    # WhiteList and are not removed
    WhiteList = ["n't", "not", "no"]
    words = input_text.split()
    CleanWords = [word for word in words if (word not in StopWordsList or
                                             word in WhiteList) and len(word) > 1]
    return " ".join(CleanWords)

final_data.Review_Text = final_data["Review_Text"].apply(RemoveStopWords)
final_data.Review_Text.head()


# removing punctuations
Punctuations = string.punctuation
print(Punctuations)
def RemovePunctuations(text):
    translator = str.maketrans('','', Punctuations)
    return text.translate(translator)

final_data["Review_Text"] = final_data["Review_Text"].apply(lambda x : RemovePunctuations(x))
final_data["Review_Text"].head()


# removing emojis
def RemoveEmoji(text):
    EmojiPattern = re.compile(pattern = "["
                              u"\U0001F600-\U0001F64F" # emoticons
                              u"\U0001F300-\U0001F5FF" # symbols & pictographs
                              u"\U0001F680-\U0001F6FF" # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                              "]+", flags = re.UNICODE)
    return EmojiPattern.sub(r'',text)

final_data["Review_Text"] = final_data["Review_Text"].apply(lambda x : RemoveEmoji(x))
final_data["Review_Text"].head()


# Tokenization of review texts
nltk.download('punkt_tab')
final_data.Review_Text = final_data.Review_Text.tolist()
TokenizeText = [word_tokenize(i) for i in final_data.Review_Text]
# for i in TokenizeText:
# print(i)
final_data.Review_Text = TokenizeText
print(final_data.Review_Text.head())

# Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def Lemmatization(text):
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

final_data["Review_Text"] = final_data["Review_Text"].apply(lambda x: Lemmatization(x))
print(final_data["Review_Text"].head())


# Joining all words with spaces
final_data["Review_Text"] = final_data["Review_Text"].apply(lambda x : " ".join(x))
final_data["Review_Text"]


#===============================================================================
# Splitting the data
X, y = final_data["Review_Text"], final_data["response"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 111)
print(X_train.shape)
print(X_test.shape)
y_train.nunique()
y_train.value_counts()


# Bag of Words
BoW = CountVectorizer(ngram_range= (1,1))
# Train data
BoW_X_train = BoW.fit_transform(X_train)
print(BoW_X_train.toarray())
print(BoW_X_train.toarray().shape)
# Test data
BoW_X_test = BoW.transform(X_test)
print(BoW_X_test.toarray())
print(BoW_X_test.toarray().shape)


#### Resampling due to data imbalance
# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit and resample the training data
BoW_X_train, BoW_y_train = smote.fit_resample(BoW_X_train, y_train)
print(BoW_y_train.value_counts(normalize=True))


#Check
#BoW_X_train.toarray()[100][21900:21950]

# Get and print the vocabulary
vocab = BoW.vocabulary_
print("Vocabulary:\n")
for word, index in sorted(vocab.items(), key=lambda x: x[1]):  # Sort by index
    print(f"Index {index}: {word}")

# TF-IDF
TF_IDF = TfidfVectorizer(ngram_range=(1,1), max_features= 200000)
#Train Data
TF_IDF_X_train = TF_IDF.fit_transform(X_train)
print(TF_IDF_X_train.toarray())
print(TF_IDF_X_train.toarray().shape)
# Test Data
TF_IDF_X_test = TF_IDF.transform(X_test)
print(TF_IDF_X_test.toarray())
print(TF_IDF_X_test.toarray().shape)


#### Resampling due to data imbalance
# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit and resample the training data
TF_IDF_X_train, TF_IDF_y_train = smote.fit_resample(TF_IDF_X_train, y_train)
print(TF_IDF_y_train.value_counts(normalize=True))

#Check
#TF_IDF_X_train.toarray()[100][21900:21950]


##### MODELLING
# Function for logistic regression to compare Bag of Words and TF-IDF
def LogisticRegressionFunction(X_train, X_test, y_train, y_test, description):
    # Initialize logistic regression for binary classification
    LogitClassifier = LogisticRegression(solver='lbfgs', random_state=111, n_jobs=-1)
    
    # Fit the model
    LogitClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = LogitClassifier.predict(X_test)
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description}")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(classification_report(y_test, y_prediction))



# Logistic regression with Bag of Words
LogisticRegressionFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'Logistic regression with BOW')


# Logistic regression with TF-IDF
LogisticRegressionFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'Logistic regression with TF-IDF')



# Function for logistic regression to compare Bag of Words and TF-IDF
def LogisticRegressionFunction(X_train, X_test, y_train, y_test, description):
    # Initialize logistic regression for binary classification
    LogitClassifier = LogisticRegression(solver='lbfgs', random_state=111, n_jobs=-1)
    
    # Fit the model
    LogitClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = LogitClassifier.predict(X_test)
    y_probabilities = LogitClassifier.predict_proba(X_test)[:, 1]  # For AUC calculation
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description}")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction)
    Precision = precision_score(y_test, y_prediction)
    AUC = roc_auc_score(y_test, y_probabilities)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(f'Precision for {description} is: {Precision:.2f}')
    print(f'AUC for {description} is: {AUC:.2f}')
    #print(classification_report(y_test, y_prediction))


# Logistic regression with Bag of Words
LogisticRegressionFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'Logistic regression with BOW')


# Logistic regression with TF-IDF
LogisticRegressionFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'Logistic regression with TF-IDF')



############ Random Forest


# Function for Random Forest to compare Bag of Words and TF-IDF
def RandomForestFunction(X_train, X_test, y_train, y_test, description):
    # Initialize Random Forest classifier
    RFClassifier = RandomForestClassifier(random_state=111, n_jobs=-1)
    
    # Fit the model
    RFClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = RFClassifier.predict(X_test)
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (Random Forest)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(classification_report(y_test, y_prediction))



# Logistic regression with Bag of Words
RandomForestFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'Random Forest with BOW')


# Logistic regression with TF-IDF
RandomForestFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'Random Forest with TF-IDF')



# Function for Random Forest to compare Bag of Words and TF-IDF
def RandomForestFunction(X_train, X_test, y_train, y_test, description):
    # Initialize Random Forest classifier
    RFClassifier = RandomForestClassifier(random_state=111, n_jobs=-1)
    
    # Fit the model
    RFClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = RFClassifier.predict(X_test)
    y_probabilities = RFClassifier.predict_proba(X_test)[:, 1]  # For AUC calculation
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (Random Forest)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction)
    Precision = precision_score(y_test, y_prediction)
    AUC = roc_auc_score(y_test, y_probabilities)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(f'Precision for {description} is: {Precision:.2f}')
    print(f'AUC for {description} is: {AUC:.2f}')
    #print(classification_report(y_test, y_prediction))


# Random Forest with Bag of Words
RandomForestFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'Random Forest with BOW')

# Random Forest with TF-IDF
RandomForestFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'Random Forest with TF-IDF')



# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees
    'max_depth': [None, 10, 20, 30],       # Maximum tree depth
    'min_samples_split': [2, 5, 10],       # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum samples at leaf node
    'bootstrap': [True, False]             # Bootstrap sampling
}

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=111)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

# Fit grid search on the training data
grid_search.fit(BoW_X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train a new Random Forest model with the best parameters
best_rf = grid_search.best_estimator_



# Function for Random Forest with tuned hyperparameters
def RandomForestFunction11(X_train, X_test, y_train, y_test, description):
    # Use the best model from grid search
    RFClassifier = best_rf
    
    # Fit the model
    RFClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = RFClassifier.predict(X_test)
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="viridis", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "white", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (Random Forest)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')




# Logistic regression with Bag of Words
RandomForestFunction11(BoW_X_train, BoW_X_test, y_train, y_test, 'Random Forest with BOW')


# Logistic regression with TF-IDF
RandomForestFunction(TF_IDF_X_train, TF_IDF_X_test, y_train, y_test, 'Random Forest with TF-IDF')



######### SVM 
# Function for SVM to compare Bag of Words and TF-IDF
def SVMFunction(X_train, X_test, y_train, y_test, description):
    # Initialize SVM classifier
    SVMClassifier = SVC(kernel='linear', probability=True, random_state=111)
    
    # Fit the model
    SVMClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = SVMClassifier.predict(X_test)
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (SVM)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='weighted')  # Use 'weighted' for multi-class
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(classification_report(y_test, y_prediction))



# Logistic regression with Bag of Words
SVMFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'SVM with BOW')


# Logistic regression with TF-IDF
SVMFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'SVM with TF-IDF')



# Function for SVM to compare Bag of Words and TF-IDF
def SVMFunction(X_train, X_test, y_train, y_test, description):
    # Initialize SVM classifier with linear kernel
    SVMClassifier = SVC(kernel='linear', probability=True, random_state=111)
    
    # Fit the model
    SVMClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = SVMClassifier.predict(X_test)
    y_probabilities = SVMClassifier.predict_proba(X_test)[:, 1]  # For AUC calculation
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (SVM)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction)
    Precision = precision_score(y_test, y_prediction)
    AUC = roc_auc_score(y_test, y_probabilities)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(f'Precision for {description} is: {Precision:.2f}')
    print(f'AUC for {description} is: {AUC:.2f}')
    #print(classification_report(y_test, y_prediction))


# SVM with Bag of Words
SVMFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'SVM with BOW')

# SVM with TF-IDF
SVMFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'SVM with TF-IDF')



######### XGBOOST
# Function for XGBoost to compare Bag of Words and TF-IDF
def XGBoostFunction(X_train, X_test, y_train, y_test, description):
    # Initialize XGBoost classifier
    XGBClassifierModel = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=111)
    
    # Fit the model
    XGBClassifierModel.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = XGBClassifierModel.predict(X_test)
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (XGBoost)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='weighted')  # Use 'weighted' for multi-class
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(classification_report(y_test, y_prediction))



# Logistic regression with Bag of Words
XGBoostFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'XGBOOST with BOW')


# Logistic regression with TF-IDF
XGBoostFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'XGBOOST with TF-IDF')



# Function for XGBoost to compare Bag of Words and TF-IDF
def XGBoostFunction(X_train, X_test, y_train, y_test, description):
    # Initialize XGBoost classifier
    XGBClassifierModel = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=111)
    
    # Fit the model
    XGBClassifierModel.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = XGBClassifierModel.predict(X_test)
    y_probabilities = XGBClassifierModel.predict_proba(X_test)[:, 1]  # For AUC calculation
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (XGBoost)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='binary')  # Use 'binary' for two-class
    Precision = precision_score(y_test, y_prediction)
    AUC = roc_auc_score(y_test, y_probabilities)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(f'Precision for {description} is: {Precision:.2f}')
    print(f'AUC for {description} is: {AUC:.2f}')
    #print(classification_report(y_test, y_prediction))


# XGBoost with Bag of Words
XGBoostFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'XGBoost with BOW')

# XGBoost with TF-IDF
XGBoostFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'XGBoost with TF-IDF')


######### K-Nearest Neighbors (KNN)
# Function for K-Nearest Neighbors to compare Bag of Words and TF-IDF
def KNNFunction(X_train, X_test, y_train, y_test, description):
    # Initialize K-Nearest Neighbors classifier
    KNNClassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    
    # Fit the model (not necessary for KNN since it's lazy learning)
    KNNClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = KNNClassifier.predict(X_test)
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (KNN)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='weighted')  # Use 'weighted' for multi-class
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(classification_report(y_test, y_prediction))



# Logistic regression with Bag of Words
KNNFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'KNN with BOW')


# Logistic regression with TF-IDF
KNNFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'KNN with TF-IDF')



# Function for K-Nearest Neighbors to compare Bag of Words and TF-IDF
def KNNFunction(X_train, X_test, y_train, y_test, description):
    # Initialize K-Nearest Neighbors classifier
    KNNClassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    
    # Fit the model (not necessary for KNN since it's lazy learning)
    KNNClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = KNNClassifier.predict(X_test)
    y_probabilities = KNNClassifier.predict_proba(X_test)[:, 1]  # For AUC calculation
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="viridis", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "white", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (KNN)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='binary')  # Use 'binary' for binary classification
    Precision = precision_score(y_test, y_prediction)
    AUC = roc_auc_score(y_test, y_probabilities)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(f'Precision for {description} is: {Precision:.2f}')
    print(f'AUC for {description} is: {AUC:.2f}')
    #print(classification_report(y_test, y_prediction))


# KNN with Bag of Words
KNNFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'KNN with BOW')

# KNN with TF-IDF
KNNFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'KNN with TF-IDF')


############## NEURAL NETWORK 
# Function for Neural Networks to compare Bag of Words and TF-IDF
def NeuralNetworkFunction(X_train, X_test, y_train, y_test, description):
    # Initialize the Neural Network classifier
    NNClassifier = MLPClassifier(hidden_layer_sizes=(100,), 
                                  activation='relu', 
                                  solver='adam', 
                                  max_iter=300, 
                                  random_state=111)
    
    # Fit the model
    NNClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = NNClassifier.predict(X_test)
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (Neural Network)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='weighted')  # Use 'weighted' for multi-class
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(classification_report(y_test, y_prediction))


# Logistic regression with Bag of Words
NeuralNetworkFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'NN with BOW')


# Logistic regression with TF-IDF
NeuralNetworkFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'NN with TF-IDF')



# Function for Neural Networks to compare Bag of Words and TF-IDF
def NeuralNetworkFunction(X_train, X_test, y_train, y_test, description):
    # Initialize the Neural Network classifier
    NNClassifier = MLPClassifier(hidden_layer_sizes=(100,), 
                                  activation='relu', 
                                  solver='adam', 
                                  max_iter=300, 
                                  random_state=111)
    
    # Fit the model
    NNClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = NNClassifier.predict(X_test)
    y_probabilities = NNClassifier.predict_proba(X_test)[:, 1]  # For AUC computation
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (Neural Network)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='binary')  # Use 'binary' for binary classification
    Precision = precision_score(y_test, y_prediction)
    AUC = roc_auc_score(y_test, y_probabilities)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(f'Precision for {description} is: {Precision:.2f}')
    print(f'AUC for {description} is: {AUC:.2f}')
    #print(classification_report(y_test, y_prediction))


# Neural Network with Bag of Words
NeuralNetworkFunction(BoW_X_train, BoW_X_test, BoW_y_train, y_test, 'NN with BOW')

# Neural Network with TF-IDF
NeuralNetworkFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'NN with TF-IDF')



# Function for Naive Bayes to compare Bag of Words and TF-IDF
def NaiveBayesFunction(X_train, X_test, y_train, y_test, description):
    # Initialize the Naive Bayes classifier
    NBClassifier = GaussianNB()
    
    # Fit the model
    NBClassifier.fit(X_train, y_train)
    
    # Make predictions
    y_prediction = NBClassifier.predict(X_test)
    y_probabilities = NBClassifier.predict_proba(X_test)[:, 1]  # For AUC computation
    
    # Generate the confusion matrix
    ConfMat = confusion_matrix(y_test, y_prediction)
    
    # Normalize the confusion matrix to percentages
    ConfMat_percentage = (ConfMat / ConfMat.sum(axis=1, keepdims=True)) * 100

    # Display the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ConfMat_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], cbar=False, ax=ax,
                annot_kws={"size": 12, "va": "center", "ha": "center",
                           "color": "black", "fontweight": "bold"})
    
    # Adding percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")
    
    ax.set_title(f"Confusion Matrix for {description} (Naive Bayes)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_prediction)
    F1 = f1_score(y_test, y_prediction, average='binary')  # Use 'binary' for binary classification
    Precision = precision_score(y_test, y_prediction)
    AUC = roc_auc_score(y_test, y_probabilities)
    
    # Print metrics
    print(f'Accuracy for {description} is: {Accuracy:.2f}')
    print(f'F1 score for {description} is: {F1:.2f}')
    print(f'Precision for {description} is: {Precision:.2f}')
    print(f'AUC for {description} is: {AUC:.2f}')
    #print(classification_report(y_test, y_prediction))


# Naive Bayes with Bag of Words
BoW_X_train1 = BoW_X_train.toarray()
BoW_X_test1 = BoW_X_test.toarray()
NaiveBayesFunction(BoW_X_train1, BoW_X_test1, BoW_y_train, y_test, 'Naive Bayes with BOW')

# Naive Bayes with TF-IDF
NaiveBayesFunction(TF_IDF_X_train, TF_IDF_X_test, TF_IDF_y_train, y_test, 'Naive Bayes with TF-IDF')



#################### Comparing the Accuracy Measures ###############

# Bar Plot
models = ['Logistic', 'Random Forest', 'SVM', 'XGBoost', 'Neural Network']
accuracy_bow = [0.89, 0.85, 0.89, 0.91, 0.90]  # Accuracy scores for Bag-of-Words
accuracy_tfidf = [0.89, 0.88, 0.90, 0.90, 0.90]  # Accuracy scores for TF-IDF
auc_bow = [0.90, 0.83, 0.88, 0.93, 0.88]  # AUC scores for Bag-of-Words
auc_tfidf = [0.95, 0.84, 0.94, 0.92, 0.91]  # AUC scores for TF-IDF


# Bar width for grouped bars
bar_width = 0.2
x = np.arange(len(models))  # Position of groups

# Create subplots for clarity
fig, ax = plt.subplots(figsize=(12, 6))

# Plot accuracy scores
ax.bar(x - bar_width, accuracy_bow, bar_width, label='Accuracy (BoW)', color='blue')
ax.bar(x, accuracy_tfidf, bar_width, label='Accuracy (TF-IDF)', color='dodgerblue')

# Plot AUC scores
ax.bar(x + bar_width, auc_bow, bar_width, label='AUC (BoW)', color='lightgreen')
ax.bar(x + 2 * bar_width, auc_tfidf, bar_width, label='AUC (TF-IDF)', color='green')

# Add titles and labels
ax.set_title('Comparison of Accuracy and AUC Scores Across Models', fontsize=16, fontweight='bold')
ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(models, rotation=45, fontsize=12)
ax.legend(loc='upper right', fontsize=12)
ax.set_ylim(0.75, 1)  # Adjust y-axis limits for better readability

# Add gridlines for better visualization
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display plot
plt.tight_layout()
plt.show()



# Line Plot
models = ['Logistic', 'Random Forest', 'SVM', 'XGBoost', 'Neural Network']
accuracy_bow = [0.89, 0.85, 0.89, 0.91, 0.90]  # Accuracy scores for Bag-of-Words
accuracy_tfidf = [0.89, 0.88, 0.90, 0.90, 0.90]  # Accuracy scores for TF-IDF
auc_bow = [0.90, 0.83, 0.88, 0.93, 0.88]  # AUC scores for Bag-of-Words
auc_tfidf = [0.95, 0.84, 0.94, 0.92, 0.91]  # AUC scores for TF-IDF

# Create the plot
plt.figure(figsize=(12, 6))

# Plot lines for each metric and feature type
plt.plot(models, accuracy_bow, label='Accuracy (BoW)', marker='o', linestyle='-', color='dodgerblue')
plt.plot(models, accuracy_tfidf, label='Accuracy (TF-IDF)', marker='o', linestyle='-', color='blue')
plt.plot(models, auc_bow, label='AUC (BoW)', marker='o', linestyle='--', color='lightgreen')
plt.plot(models, auc_tfidf, label='AUC (TF-IDF)', marker='o', linestyle='--', color='green')

# Add titles and labels
plt.title('Comparison of Accuracy and AUC Scores Across Models', fontsize=16, fontweight='bold')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust y-axis limits if needed
plt.ylim(0.75, 1)

# Display the plot
plt.tight_layout()
plt.show()



