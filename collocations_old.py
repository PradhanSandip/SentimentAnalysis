import csv
from textblob import TextBlob
from collections import Counter
import nltk

nltk.download("stopwords")
# Load the dataset and extract the movie reviews
dataset = []
with open('reviews.csv', encoding="utf8") as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        dataset.append(row[2])

progress = 1
positive_collocations = []
negative_collocations = []

# Define a function to compute the overall sentiment score of a review and convert it into a Boolean label
def sentiment_analysis(review):
    global progress
    print(progress)
    progress += 1
    if(len(review) > 0):
        review = review.replace("&quot;", '"')
        # Compute the sentiment score of each sentence using TextBlob
        sentence_scores = []
        for sentence in TextBlob(review).sentences:
            sentence_scores.append(sentence.sentiment.polarity)
        # Compute the overall sentiment score of the review as the average of the sentence scores
        overall_score = sum(sentence_scores)/len(sentence_scores)
        # Convert the overall sentiment score into a Boolean label
        if overall_score < 0:
            negative_collocations.append(review)
            return 'negative'
        else:
            positive_collocations.append(review)
            return 'positive'
    
    
# Define a function to extract the most common collocations (i.e., sequences of words that often appear together) in a text
def extract_collocations(text, n=40):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Filter out stop words and punctuation
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in nltk.corpus.stopwords.words('english')]
    # Use the bigram function to extract all pairs of adjacent words in the text
    pairs = nltk.bigrams(words)
    # Filter out pairs where either word is too short (less than 3 characters)
    pairs = [pair for pair in pairs if len(pair[0])>=3 and len(pair[1])>=3]
    # Compute the frequency of each pair of words
    frequency = Counter(pairs)
    # Extract the n most common pairs of words
    top_pairs = frequency.most_common(n)
    # Convert each pair of words into a string and return the list of strings
    top_collocations = [' '.join(str(pair)) for pair in top_pairs]
    return top_collocations

for review in dataset[:10]:
    sentiment_analysis(review)
# # Compute the most common collocations in positive and negative reviews separately, with and without part-of-speech filtering
positive_collocations_with_pos = extract_collocations(' '.join(positive_collocations), n=40)
# positive_collocations_without_pos = extract_collocations(' '.join(positive_collocations), n=40)
# negative_collocations_with_pos = extract_collocations(' '.join(negative_collocations), n=40)
# negative_collocations_without_pos = extract_collocations(' '.join(negative_collocations), n=40)

print(positive_collocations_with_pos)