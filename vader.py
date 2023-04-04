import nltk
from ProcessData import ProcessData
from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
import threading
data = ProcessData("reviews.csv")
reviews = data.get_reviews()
#nltk.download(["maxent_ne_chunker","words"])
'''Hutto, C.J. & Gilbert, Eric. (2015). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Proceedings of the 8th International Conference on Weblogs and Social Media, ICWSM 2014. '''
################################################################################################### VADER APPROACH #######################################################################################################
#SIMPLE RULE BASED APPROACH, THAT SCORES EACH WORD FROM NEGATIVE, NEUTRAL AND POSITIVE
def vader(review):
    #replace quote unicode with actual quote marks
    review = review.replace("&quot;",'"')
    # tokenize = word_tokenize(review)
    # speach_tag = pos_tag(tokenize)
    # chunked = ne_chunk(speach_tag)
    intensity_analyzer = SentimentIntensityAnalyzer()
    result = intensity_analyzer.polarity_scores(review)
    return result



def process(pos1,pos2):
    for review in data.get_reviews()[pos1:pos2]:
        print(vader(review), review)



#use thread to process faster
'''Using 10 threads to process 10000 reviews, each thread will handle 1000 reviews'''
threads = []
workload = [[0,999], [1000,1999], [2000, 2999], [3000, 3999], [4000, 4999], [5000, 5999], [6000,6999], [7000, 7999], [8000, 8999], [9000, 9999]]
for i in range(10):
    thread = threading.Thread(target=process, args=workload[i])
    thread.start()

for thread in threads:
    thread.join()




