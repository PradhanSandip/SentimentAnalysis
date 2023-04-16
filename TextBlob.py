from textblob import TextBlob
import csv

# Open the file containing movie reviews
with open('reviews.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # Skip the header row
    for row in reader:
        review_id, movie_id, review_text = row
        #make sure review is not empty
        if(len(review_text) > 0)
            review_blob = TextBlob(review_text)
            sentence_sentiments = []
            for sentence in review_blob.sentences:
                sentence_sentiments.append(sentence.sentiment.polarity)
            overall_sentiment = sum(sentence_sentiments) / len(sentence_sentiments)
            if overall_sentiment < 0:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'positive'
            print(review_id, movie_id, sentiment_label)
