import pandas as pd
from textblob import TextBlob

# Read the CSV file into a DataFrame
data = pd.read_csv('/htdocs/file.csv')

# Create a new column for sentiment polarity
data['Sentiment'] = data['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Define a function to label the sentiment
def label_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the sentiment labels to the DataFrame
data['Sentiment_Label'] = data['Sentiment'].apply(label_sentiment)

# Print the resulting DataFrame
print(data)
