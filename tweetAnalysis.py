# import dependencies
import csv
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# define function to analyze sentiment using VADER
sid = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(tweet):
    scores = sid.polarity_scores(tweet) # returns dictionary of sentiment scores
    compound_score = scores['compound'] # represents overall sentiment
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# initialize sentiment count variables
positive_count = 0
negative_count = 0
neutral_count = 0

# store sentiment scores
positive_scores = []
negative_scores = []
neutral_scores = []

# count total number of tweets
total_tweets = sum(1 for _ in open('tweet_10000.csv', 'r', encoding = 'utf-8'))

# read and analyze tweets
with open('tweet_10000.csv', 'r', encoding = 'utf-8') as tweets, tqdm(total=total_tweets, desc="Processing") as progress_bar:
    reader = csv.reader(tweets)
    for row in reader:
        tweet = row[0]
        sentiment = analyze_sentiment_vader(tweet)
        
        # total up positive, negative, and neutral tweets
        if sentiment == 'positive':
            positive_count += 1
            positive_scores.append(sid.polarity_scores(tweet)['compound'])
        elif sentiment == 'negative':
            negative_count += 1
            negative_scores.append(sid.polarity_scores(tweet)['compound'])
        else:
            neutral_count += 1
            neutral_scores.append(sid.polarity_scores(tweet)['compound'])

        progress_bar.update(1) # update the progress bar

# prepare data for visual representation
sentiments = ['Positive', 'Neutral', 'Negative']
counts = [positive_count, neutral_count, negative_count]

plt.figure(figsize=(12, 5))

# create bar graph
plt.subplot(1, 3, 1)
plt.bar(sentiments, counts)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis of 10,000 Tweets')

# create pie chart
plt.subplot(1, 3, 2)
plt.pie(counts, labels=sentiments, autopct='%1.1f%%')
plt.title('Sentiment Analysis of 10,000 Tweets')

# create histogram
plt.subplot(1, 3, 3)
plt.hist([positive_scores, neutral_scores, negative_scores], bins=10, label=sentiments)
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.title('Sentiment Scores Distribution')
plt.legend()

# adjust layout and spacing
plt.tight_layout()

# display visual data
plt.show()

# output sentiment count
# print(f"Positive tweets: {positive_count}")
# print(f"Negative tweets: {negative_count}")
# print(f"Neutral tweets: {neutral_count}")