from ProcessData import ProcessData
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import threading
from time import sleep
import pandas as pd
'''------------------------This part of the code is responsible for loading the model and setting up the model for inference-------------------------------------------'''
class Roberta:
    def __init__(self):
        #pre trained model, https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
        self.MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        #custom tokenizer from the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        #loading model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL, from_tf=True)
        #loading the reviews csv
        self.data = ProcessData("reviews.csv")
        #getting all the reviews
        self.reviews = self.data.get_reviews()
        #defining labels and its corrosponding menaing
        self.LABELS = {"LABEL_0":"neg", "LABEL_1":"neu", "LABEL_2":"pos"}
        #store all the sentiment score for each movie
        self.temp = {}
        #keep track of sentiment analysis progress, where progress is the review number out of 10000.
        self.progress = 0
        #boolean that controls whether to show progress or not
        self.show_progress = True
        #progress thread
        self.progress_thread = None

# '''--------------------------------------------------This part of the code is responsible for performing inference on the model------------------------------------------------'''
#     #function that generates sentiment score of given review and return the scores.
    def roberta_sentiment(self,review):
        #increment progress each time this function runs
        self.progress += 1
        #creating tensor of review text using tokenizer, this rturns a pytorch tensor, with max length of 512
        encoded_input = self.tokenizer(review, padding=True, truncation=True, max_length=512, return_tensors="pt")
        #performing inference using the tensor encoded_input, ** unpacks the tensor
        output = self.model(**encoded_input)
        #obtaining the result from output and creating a new list, this return list with 3 probablilty values [negative, neutral, positive] 
        scores = output[0][0].detach().numpy()
        #applying softmax to the scores in order to normalise the result and create probability scores
        scores = softmax(scores)
        #initialise the higest score as the first result in the list
        heighest = scores[0]
        #settin the tag to negative as first result in the scores is corrospondent to negative
        tag = "neg"
        #if second value in the scores is higher than current value 
        if(scores[1] > heighest):
            #set the heigest as second value with position 1
            heighest = scores[1]
            #set the tag as neutral, as second value is corrospondent to neutral
            tag = "neu"
        #if the third value in the scores is higher than current value     
        if(scores[2] > heighest):
            # set the heigest as second value with position 2
            heighest = scores[2]
            #set the tag as positive, as third value is corrospondent to positive
            tag ="pos"
        #return the result as tuple with the tag and score
        return (tag, heighest)

    #function that process review from given range [pos1,pos2]. pos1 is the starting position of the review in the csv file and pos2 is the ending position of the review. 
    def process(self, pos1,pos2):
        #get all the movie ids
        ids = self.data.get_movie_ids()
        #for each review within the range.
        for index, review in enumerate(self.data.get_reviews()[pos1:pos2]):
            #if the review is not empty, ie if the review is not float. [pandas return nan (float) is the column is empty]
            if(not type(review) is type(float('nan'))):
                #perform the inference and store the result
                roberta = self.roberta_sentiment(review)
                '''The temp dicitonary contains key as the movie id, and each key has 7 values [columns] 
                    [positive, neutral, negative, total, positive_list, neutral_list, negative_list]
                    poitive:        total number of positives review of a movie
                    neutral:        total number of neutral review of a movie
                    negative:       total number of negative review of a movie
                    total:          total number of review of a movie
                    positive_list:  list of row numbers that are positive
                    neutral_list:   list of row numbers that are neutral
                    negative_list:  list of row numbers that are negative
                '''
                #if the movie is not in the dictionary 
                if(ids[pos1+index] not in self.temp):
                    #if the review is positive
                    if(roberta[0] == "pos"):
                        #add the movie to the dictionary as positive. setting positive to 1 and total to 1 and storing the positive row number
                        self.temp[ids[pos1+index]] = [1,0,0,1,[pos1+index],[],[]]
                    #if the review is neutral
                    elif(roberta[0] == "neu"):
                        #add the movie to the dictionary as neutral. setting neutral to 1 and total to 1 and storing the neutral row number
                        self.temp[ids[pos1+index]] = [0,1,0,1,[],[pos1+index],[]]
                    #if the review is negative
                    elif(roberta[0] == "neg"):
                        #add the movie to the dictionary as negative. setting negative to 1 and total to 1 and storing the negative row number
                        self.temp[ids[pos1+index]] = [0,0,1,1,[],[],[pos1+index]]
                #if the movie is already in the dictionary
                else:
                    #if the review is positive
                    if(roberta[0] == "pos"):
                        #increment the positive counter by 1
                        self.temp[ids[pos1+index]][0] = self.temp[ids[pos1+index]][0] + 1
                        #increment the total by 1
                        self.temp[ids[pos1+index]][3] = self.temp[ids[pos1+index]][3] + 1
                        #add the positive row number to the positive list
                        self.temp[ids[pos1+index]][4].append(pos1+index)
                    #if the review is neutral
                    elif(roberta[0] == "neu"):
                        #increment the neutral counter by 1
                        self.temp[ids[pos1+index]][1] = self.temp[ids[pos1+index]][1] + 1
                        #increment the total by 1
                        self.temp[ids[pos1+index]][3] = self.temp[ids[pos1+index]][3] + 1
                        #add the neutral row number to the neutral_list
                        self.temp[ids[pos1+index]][5].append(pos1+index)
                    #if the review is negative
                    elif(roberta[0] == "neg"):
                        #increment the negative counter by 1
                        self.temp[ids[pos1+index]][2] = self.temp[ids[pos1+index]][2] + 1
                        #increment the total by 1
                        self.temp[ids[pos1+index]][3] = self.temp[ids[pos1+index]][3] + 1
                        #add the negative row number to the negative list
                        self.temp[ids[pos1+index]][6].append(pos1+index)
            #if the review is empty print the row number
            else:
                print(index+pos1)



# '''--------------------------------------This part of the code is responsible for showing progress to the screen-----------------------------------------------------------------'''
    #function that displays progress    
    def print_progress(self):
        #while the show progress flag is true
        while self.show_progress:
            #print progress to the screen, where progress is number of reviews processed / 100
            print("Progress: "+str(self.progress/100)+"%")
            #sleep for 2 seconds
            sleep(2)

    def display_progress(self):
        #using thread to show progress as this will allow other part of the code to run
        self.progress_thread = threading.Thread(target=self.print_progress)
        #start the progress thread
        self.progress_thread.start()



# '''------------------------------------This part of the code is responsible for using multithreads to speed up the process----------------------------------------'''
# '''Using 10 threads to process 10000 reviews, each thread will handle 1000 reviews'''
    def multi_threading(self):
        #list of threads
        threads = []
        #workload defines which portion of the review each thread will handle, as there are 10000 reviewas workload is between 0 - 10000
        workload = [[0,999], [1000,1999], [2000, 2999], [3000, 3999], [4000, 4999], [5000, 5999], [6000,6999], [7000, 7999], [8000, 8999], [9000, 9999]]
        #looping 100 time to create 10 threads
        for i in range(10):
            #create thread and run the process function, with parameter workload[i] ( for example workload[0] is [0,999])
            thread = threading.Thread(target=self.process, args=workload[i])
            #store the thread in the list
            threads.append(thread)
            #start the thread
            thread.start()

        #keep track of the interation of for loop
        count = 0
        #for each thread
        for index, thread in enumerate(threads):
            #join the result
            thread.join()
            #increase the count by 1
            count += 1
            #if the count is 10 ie the last thread has finished 
            if(count == 10):
                #stop displaying the progress
                self.show_progress = False
                #terminate the progress thread
                if(self.progress_thread != None):
                    self.progress_thread.join()
                #update the progress to the user
                print("Processed all reviews")
                print("Compiling data......")
                #generate dataframe using temp variable which holds all the generated results, and flip the axis
                self.temp = pd.DataFrame(self.temp).T
                #save the datafrave to csv file, with columns ["positive", "neutral", "negative", "total", "positive_list", "neutral_list", "negative_list"]
                self.temp.to_csv("roberta_individual.csv", sep="\t", header=["positive", "neutral", "negative", "total", "positive_list", "neutral_list", "negative_list"])
                #update user
                print("Generated csv roberta_individual.csv")

if __name__ == "__main__":
    #create class object
    roberta = Roberta()
    #display progress 
    roberta.display_progress()
    #run inference and save the result in roberta_individual.csv
    roberta.multi_threading()