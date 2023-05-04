from word_cloud_generator import *
from ProcessData import ProcessData
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from accuracy import Accuracy
#######################################
# Author: Sandip Pradhan              #
#######################################


'''#################################### This file is used to generate plots/graphs for analysis ##########################################'''




#load the review csv file
reviews = ProcessData("reviews.csv",True)
#load the roberta_individual.csv
roberta_reviews = pd.read_csv("generated_roberta.csv", sep="\t", lineterminator="\n")
#set plot style
plt.style.use('fivethirtyeight')
#function that displays total number of reviews and total number of unique movie in the reviews
def review_info():
    #creating subplot with 2 axis
    fig, (ax1, ax2) = plt.subplots(1, 2)
    #bar chart of total number of reviews
    ax1.bar(["Total number of reviews"], len(reviews.get_reviews()), width=1)
    #set x axis view to modify the width and position of the bar
    ax1.set_xlim(-4, 3)
    #bar chart of unique movies
    ax2.bar(["Total number of unique movies"], len(set(reviews.get_movie_ids())), width=1, color="#00e56e")
    #set x axis view to modify the width and position of the bar
    ax2.set_xlim(-4, 3)
    #setting the y axis interval (starts from 0, goes to 700 with intervals of 40)
    ax2.yaxis.set_ticks(np.arange(0, 700, 40))
    #draw the chart
    plt.show()

# review_info()

def review_per_movie():
    total = roberta_reviews["total"]
    plt.plot(total)
    plt.yticks(np.arange(0,980,20))
    plt.show()    

# review_per_movie()

#plot average review per movie
def average_review():
    total = roberta_reviews["total"]
    average = (sum(total)/len(total))
    plt.bar("Average",average)
    plt.gca().set_xlim(-3,3)
    plt.gca().set_yticks(np.arange(0,20,1))
    plt.show()

# average_review()

#draw word cloud of negative review of top reviewed movie
def negative_word_cloud(pos):
    _reviews = roberta_reviews.sort_values(by=["total"],ascending=False,ignore_index=True)
    row = _reviews.loc[pos]
    negative_list = literal_eval(row["negative_list\r"])
    negative_reviews = []
    for index in negative_list:
        negative_reviews.append(reviews.get(index)["review"])
        
    generate_from_review("".join(negative_reviews), f'Negative word cloud of {row["Unnamed: 0"]}')
    
#draw word cloud of negative review of top reviewed movie
def positive_word_cloud(pos):
    _reviews = roberta_reviews.sort_values(by=["total"],ascending=False,ignore_index=True)
    row = _reviews.loc[pos]
    positive_list = literal_eval(row["positive_list"])
    positive_reviews = []
    for index in positive_list:
        positive_reviews.append(reviews.get(index)["review"])
        
    generate_from_review("".join(positive_reviews), f'Positive word cloud of {row["Unnamed: 0"]}')

# positive_word_cloud(0)

'''This function plots the accuracy compairing two sentiment analysis approach'''
def plot_sentiment_accuracy():
    a = Accuracy()
    a.plot_result()

plot_sentiment_accuracy()

'''This function draws a pie chart of sentiment score of all reviews, there are two charts one for vader and one for roberta'''
def sentiment_pie_chart():
        #load vader csv
        vader_data = ProcessData("generated_vader.csv", False).get_data_frame()
        #load roberta csv
        roberta_data = ProcessData("generated_roberta.csv", False).get_data_frame()
        #calculate vader sentement score
        vader_pos = pd.to_numeric(vader_data[1][1:]).sum()
        vader_neu = pd.to_numeric(vader_data[2][1:]).sum()
        vader_neg = pd.to_numeric(vader_data[3][1:]).sum()
        #calculate roberta sentiment score
        roberta_pos = pd.to_numeric(roberta_data[1][1:]).sum()
        roberta_neu = pd.to_numeric(roberta_data[2][1:]).sum()
        roberta_neg = pd.to_numeric(roberta_data[3][1:]).sum()
        #creating 2 axis
        fig, (axis1,axis2) = plt.subplots(1,2)
        #pie chart label
        labels = ["positive", "negative", "neutral"]
        #vader pie chart
        axis1.pie([vader_pos, vader_neg, vader_neu], labels = labels)
        axis1.set_title("Vader sentiment")
        #roberta pie chart
        axis2.pie([roberta_pos, roberta_neg, roberta_neu], labels = labels)
        axis2.set_title("Roberta sentiment")
        #figure title
        fig.suptitle("Sentiment score of all reviews", fontsize=20)
        #show plot
        plt.show() 


sentiment_pie_chart()