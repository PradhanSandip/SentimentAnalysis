from ProcessData import ProcessData
import roberta
import vader
import pandas as pd
from nltk import word_tokenize
import matplotlib.pyplot as plt
'''------------------------This class is used to measure accuracy of various sentiment analysis approach-----------------------'''
class Accuracy:
    #constructor
    def __init__(self):
        #machine learning model score
        self.model_accuracy_score = 0
        #vader score
        self.vader_accuracy_score = 0
        self.vader_plot = []
        self.model_plot = []
        self.model_accuracy("review_labeled.csv")
        self.vader_accuracy("review_labeled.csv")
        
    #function calculates accuracy of machine learning model, with review data
    def model_accuracy(self, csv_file):
        r = roberta.Roberta()
        #loading the labeled reviews file
        data = ProcessData(csv_file, False)
        #list of reviews
        reviews = data.get_data_frame()[1]
        #list of labels
        labels = data.get_data_frame()[2]
        #for each review
        for index, review in enumerate(reviews):
            #perform a inference and store the result
            output = r.roberta_sentiment(review)
            #if the output tag matches the label
            if(output[0] == labels[index]):
                
                #add 1 score
                self.model_accuracy_score += 1
                self.model_plot.append(self.model_accuracy_score)
            else:
                self.model_plot.append(self.model_accuracy_score)

        #return the result in %
        return (self.model_accuracy_score/len(reviews))*100
  
    #function calculates accuracy of vader sentiment
    def vader_accuracy(self, csv_file):
        v = vader.Vader()
        #loading the labeled reviews file
        data = ProcessData(csv_file,False)
        #list of reviews
        reviews = data.get_data_frame()[1]
        #list of labels
        labels = data.get_data_frame()[2]
        #for each review
        for index, review in enumerate(reviews):
            #perform a inference and store the result
            output = v.vader(review)
            #if the output tag matches the label
            if(output[0] == labels[index]):
                
                #add 1 score
                self.vader_accuracy_score += 1
                self.vader_plot.append(self.vader_accuracy_score)
            else:
                self.vader_plot.append(self.vader_accuracy_score)
        #return the result in %
        return (self.vader_accuracy_score/len(reviews))*100

    def plot_result(self):
        plt.plot(self.model_plot,label=f"Roberta accuracy {self.model_accuracy_score}%")
        plt.plot(self.vader_plot,label=f"Vader accuracy {self.vader_accuracy_score}%")
        plt.legend()
        plt.show()
        










        
