from flask import Flask, render_template, request, jsonify
import pandas as pd
from textblob import TextBlob
import spacy
from gensim import corpora, models

app = Flask(__name__)
app.config['DEBUG'] = True

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Define the topics
topics = [
    "Product Quality",
    "Customer Service",
    "Shipping and Delivery",
    "Price and Value",
    "Product Features and Specifications",
    "Ease of Use",
    "Product Performance",
    "Compatibility and Interoperability",
    "User Experience",
    "Product Comparisons",
    "Product Recommendations",
    "Packaging and Presentation",
    "Returns and Refunds",
    "Product Complaints",
    "Product Praises and Positive Experiences"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        # Perform necessary operations on the uploaded file
        df = pd.read_csv(file)
        
        # Process the DataFrame and perform sentiment analysis
        sentiments = []
        for text in df['Review']:
            blob = TextBlob(str(text))  # Convert to string if necessary
            sentiment = blob.sentiment.polarity

            # Assign sentiment labels as positive or negative
            if sentiment > 0:
                sentiment_label = 'Positive'
            elif sentiment < 0:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'

            sentiments.append(sentiment_label)
        
        # Add the sentiments as a new column in the DataFrame
        df['Sentiment'] = sentiments
        
        # Perform named entity recognition on the selected documents
        entities = []
        categories = []
        for text in df['Review'].head(10):
            doc = nlp(text)
            for entity in doc.ents:
                entities.append(entity.text)
                categories.append(entity.label_)
        
        # Create a new DataFrame for the named entities
        num_entities = len(entities)  # Get the number of entities extracted
        df_subset = df.head(num_entities)  # Select a subset of the original DataFrame with matching length
        entities_df = pd.DataFrame({'Review': df_subset['Review'], 'Named Entity': entities, 'Category': categories})
        
        # Perform topic modeling
        texts = df['Review'].apply(lambda x: x.split())  # Tokenize the text into individual words
        dictionary = corpora.Dictionary(texts)  # Create a dictionary from the tokenized words
        corpus = [dictionary.doc2bow(text) for text in texts]  # Convert tokenized words to bag-of-words representation
        lda_model = models.LdaModel(corpus, num_topics=len(topics), id2word=dictionary, passes=10)  # Train the LDA model
        
        # Get the topic distribution for each document
        topic_distributions = [lda_model[doc] for doc in corpus]
        
        # Assign the most probable topic for each document
        topic_labels = [topics[max(doc, key=lambda x: x[1])[0]] for doc in topic_distributions]
        
        # Add the topics as a new column in the DataFrame
        df['Topic'] = topic_labels
        
        # Prepare the results as a dictionary
        results = {
            'sentiments': df.to_dict(orient='records'),
            'entities': entities_df.to_dict(orient='records'),
            'topics': df[['Review', 'Topic']].to_dict(orient='records')
        }

        # Render the template with the results
        return render_template('index.html', results=results)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(port=8000)
 