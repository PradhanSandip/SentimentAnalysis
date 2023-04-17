from ProcessData import ProcessData
import roberta
import vader
import pandas as pd
'''------------------------This class is used to measure accuracy of various sentiment analysis approach-----------------------'''
class Accuracy:
    #constructor
    def __init__(self):
        #machine learning model score
        self.model_accuracy_score = 0
        #vader score
        self.vader_accuracy_score = 0
    
    #function calculates accuracy of machine learning model, with review data
    def model_accuracy(self, csv_file):
        #loading the labeled reviews file
        data = ProcessData(csv_file)
        #list of reviews
        reviews = data.get_reviews()
        #list of labels
        labels = data.get_data_frame()["label"]
        #for each review
        for index, review in enumerate(reviews):
            #perform a inference and store the result
            output = roberta_sentiment(review)
            #if the output tag matches the label
            if(output[0] == labels[index]):
                #add 1 score
                self.model_accuracy_score += 1

        #return the result in %
        return (self.model_accuracy_score/len(reviews))*100

    #function calculates accuracy of vader sentiment
    def vader_accuracy(self, csv_file):
        #loading the labeled reviews file
        data = ProcessData(csv_file)
        #list of reviews
        reviews = data.get_reviews()
        #list of labels
        labels = data.get_data_frame()["label"]
        #for each review
        for index, review in enumerate(reviews):
            #perform a inference and store the result
            output = vader(review)
            #if the output tag matches the label
            if(output[0] == labels[index]):
                #add 1 score
                self.vader_accuracy_score += 1

        #return the result in %
        return (self.vader_accuracy_score/len(reviews))*100


    def create(self):
        #loading the labeled reviews file
        data = ProcessData("reviews.csv")
        #list of reviews
        reviews = data.get_reviews()
        frame = {"review":[],"label":[]}
        for review in reviews:
            frame["review"].append(review)

        df = pd.DataFrame(temp).T
        df.to_csv("review_labeled.csv", sep="\t", header=["review", "label"])

a = Accuracy()
a.create()


        
