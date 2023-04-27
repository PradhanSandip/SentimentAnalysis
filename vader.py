import nltk
from ProcessData import ProcessData
from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import threading
###########################
# Author: Sandip Pradhan  #
###########################
class Vader:
    def __init__(self):
        self.data = ProcessData("reviews.csv",True)
        self.reviews = self.data.get_reviews()
        nltk.download(["maxent_ne_chunker","words", "punkt", "averaged_perceptron_tagger","vader_lexicon"])
        #store all sentiment score for each movie.
        self.temp = {}
        #keep track of sentiment analysis progress, where progress is the review number out of 10000.
        self.progress = 0
        self.show_progress = True    

    '''Hutto, C.J. & Gilbert, Eric. (2015). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Proceedings of the 8th International Conference on Weblogs and Social Media, ICWSM 2014. '''
################################################################################################### VADER APPROACH #######################################################################################################
    #SIMPLE RULE BASED APPROACH, THAT SCORES EACH WORD FROM NEGATIVE, NEUTRAL AND POSITIVE
        
    def vader(self,review):
        self.progress
        self.progress += 1
        if(not type(review) is type(float('nan'))):
            #replace quote unicode with actual quote marks
            review = review.replace("&quot;",'"')
            # tokenize = word_tokenize(review)
            # speach_tag = pos_tag(tokenize)
            # chunked = ne_chunk(speach_tag)
            intensity_analyzer = SentimentIntensityAnalyzer()
            result = intensity_analyzer.polarity_scores(review)
            #get the highest score result between positive, neutral and negative
            highest = None
            tag = None
    
            # for x in list(result)[:3]:
            #     if(highest == None):
            #         highest = result[x]
            #         tag = x
            #     elif(highest < result[x]):
            #         highest = result[x]
            #         tag = x
            for tag in list(result)[3:]:
                if(result[tag] > 0):
                    return ("pos", result[tag])
                elif(result[tag] < 0):
                    return ("neg", result[tag])
                elif(result[tag] == 0):
                    return ("neu", result[tag])


    def process(self,pos1,pos2):
        ids = self.data.get_movie_ids()
        for index, review in enumerate(self.data.get_reviews()[pos1:pos2]):
            if(not type(review) is type(float('nan'))):
                vader_sentiment = self.vader(review)
                if(ids[index] not in self.temp):
                    if(vader_sentiment[0] == "pos"):
                        self.temp[ids[index]] = [1,0,0,1]
                    elif(vader_sentiment[0] == "neu"):
                        self.temp[ids[index]] = [0,1,0,1]
                    elif(vader_sentiment[0] == "neg"):
                        self.temp[ids[index]] = [0,0,1,1]
                else:
                    if(vader_sentiment[0] == "pos"):
                        self.temp[ids[index]][0] = self.temp[ids[index]][0] + 1
                        self.temp[ids[index]][3] = self.temp[ids[index]][3] + 1
                    elif(vader_sentiment[0] == "neu"):
                        self.temp[ids[index]][1] = self.temp[ids[index]][1] + 1
                        self.temp[ids[index]][3] = self.temp[ids[index]][3] + 1
                    elif(vader_sentiment[0] == "neg"):
                        self.temp[ids[index]][2] = self.temp[ids[index]][2] + 1
                        self.temp[ids[index]][3] = self.temp[ids[index]][3] + 1
                # print(pos1+index)
            else:
                print(type(review))
    

    def print_p(self):
        while self.show_progress:
            print("Progress: "+str(self.progress/100)+"%")

    def start(self):
        p_thread = threading.Thread(target=self.print_p)
        p_thread.start()

        #use thread to process faster
        '''Using 10 threads to process 10000 reviews, each thread will handle 1000 reviews'''
        threads = []
        workload = [[0,999], [1000,1999], [2000, 2999], [3000, 3999], [4000, 4999], [5000, 5999], [6000,6999], [7000, 7999], [8000, 8999], [9000, 9999]]
        for i in range(10):
            thread = threading.Thread(target=self.process, args=workload[i])
            threads.append(thread)
            thread.start()
            
        c = 0
        for index, thread in enumerate(threads):
            thread.join()
            c += 1
            if(c == 10):
                self.show_progress = False
                p_thread.join()
                print("Processed all reviews")
                print("Compiling data......")
                self.temp = pd.DataFrame(self.temp).T
                self.temp.to_csv("generated_vader.csv", sep="\t", header=["positive", "neutral", "negative", "total"])
                print("Generated csv generated_vader.csv")


if __name__ == "__main__":
    #create class object
    vader = Vader()
    #display progress 
    vader.start()
    




