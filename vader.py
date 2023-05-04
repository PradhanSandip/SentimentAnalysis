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
        #required nltk downloads
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
        if(type(review) != type(float('nan'))):
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
            for tag in list(result)[3:]:
                if(result[tag] > 0):
                    return ("pos", result[tag])
                elif(result[tag] < 0):
                    return ("neg", result[tag])
                elif(result[tag] == 0):
                    return ("neu", result[tag])

    ''' This function is passed to a thread to run, main aim of this function to run vader function defined above in a batch'''
    def process(self,pos1,pos2):
        #get movie ids
        ids = self.data.get_movie_ids()
        #for each review in a range
        for index, review in enumerate(self.data.get_reviews()[pos1:pos2]):
            #if review is not empty
            if(not type(review) is type(float('nan'))):
                #calculate vader compund score
                vader_sentiment = self.vader(review)
                '''The temp dicitonary contains key as the movie id, and each key has 5 values [columns] 
                    [movie_id, positive, neutral, negative, total]
                    movie_id:       id of the movie which the review on
                    poitive:        total number of positives review of a movie
                    neutral:        total number of neutral review of a movie
                    negative:       total number of negative review of a movie
                    total:          total number of review of a movie
                '''
                #if the movie id does not exist in the temp dictionary
                if(ids[index] not in self.temp):
                    #if review is positive
                    if(vader_sentiment[0] == "pos"):
                        #add the review as positive
                        self.temp[ids[index]] = [1,0,0,1]
                    #if review is neutral
                    elif(vader_sentiment[0] == "neu"):
                        #add the review as neutral
                        self.temp[ids[index]] = [0,1,0,1]
                    #if the review is negative
                    elif(vader_sentiment[0] == "neg"):
                        #add the movie as negative
                        self.temp[ids[index]] = [0,0,1,1]
                #if the movie id is in temp dictionary
                else:
                    #if the review is positive
                    if(vader_sentiment[0] == "pos"):
                        #increment positive column
                        self.temp[ids[index]][0] = self.temp[ids[index]][0] + 1
                        #increment total by 1
                        self.temp[ids[index]][3] = self.temp[ids[index]][3] + 1
                    #if the review is neutral
                    elif(vader_sentiment[0] == "neu"):
                         #increment neutral column
                        self.temp[ids[index]][1] = self.temp[ids[index]][1] + 1
                        #increment total by 1
                        self.temp[ids[index]][3] = self.temp[ids[index]][3] + 1
                    #if the review is negative
                    elif(vader_sentiment[0] == "neg"):
                         #increment negative column
                        self.temp[ids[index]][2] = self.temp[ids[index]][2] + 1
                        #increment total by 1
                        self.temp[ids[index]][3] = self.temp[ids[index]][3] + 1
                # print(pos1+index)
            else:
                print(type(review))
    
    '''This function displays progress'''
    def print_p(self):
        #while show progress is true
        while self.show_progress:
            #print progress
            print("Progress: "+str(self.progress/100)+"%")

    '''This function creates 10 threads each threads handles 1000 reviews'''
    def start(self):
        #initialize progress thread to show progress
        p_thread = threading.Thread(target=self.print_p)
        #start progress thread
        p_thread.start()

        #use thread to process faster
        '''Using 10 threads to process 10000 reviews, each thread will handle 1000 reviews'''
        threads = []
        #workload range
        workload = [[0,999], [1000,1999], [2000, 2999], [3000, 3999], [4000, 4999], [5000, 5999], [6000,6999], [7000, 7999], [8000, 8999], [9000, 9999]]
        #for rach workload
        for i in range(10):
            #create a thread and run the process function
            thread = threading.Thread(target=self.process, args=workload[i])
            #store the thread
            threads.append(thread)
            #start the thread
            thread.start()

        #cthread counter    
        c = 0
        #for index and thread in threads list
        for index, thread in enumerate(threads):
            #join thread once completed
            thread.join()
            #increment the thread counter
            c += 1
            #if the thread counter is 10. means the last thread has finished
            if(c == 10):
                #stop displaying progress
                self.show_progress = False
                #join the progress thread
                p_thread.join()
                #diplay info
                print("Processed all reviews")
                print("Compiling data......")
                #create data frame to save as csv
                self.temp = pd.DataFrame(self.temp).T
                #saving data frame
                self.temp.to_csv("generated_vader.csv", sep="\t", header=["positive", "neutral", "negative", "total"])
                #update user
                print("Generated csv generated_vader.csv")


if __name__ == "__main__":
    #create class object
    vader = Vader()
    #display progress 
    vader.start()
    




