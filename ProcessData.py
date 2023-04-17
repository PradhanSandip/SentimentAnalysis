import pandas as pd
'''############################ This class is responsible for reading csv file and process the data it contains ###############################################'''
class ProcessData:
    #constructor, takes the csv file location
    def __init__(self, csv_file, set_col):
        #storing the file location in a global variable
        self.csv_file = csv_file
        #reading the csv file and storing the content in a variable. 
        #parameters
        #[sep is short for seperator which tells how the columns in csv file is sperated]
        #[lineterminator tells the csv parser how each line is separated]
        #[hearder is used to inform csv parser if the file contains column names]
        self.data = pd.read_csv(csv_file,sep='\t', lineterminator='\n', header=None)
        #defining column names as the csv file does not have any column names.
        if(set_col):
            columns = ["id","movie_id","review"]
            #setting the column names
            self.data.columns = columns

    #function that returns all the review 
    def get_reviews(self):
        return self.data["review"]
    #function that returns all the ids
    def get_ids(self):
        return self.data["id"]
    #function that returns all the movie id
    def get_movie_ids(self):
        return self.data["movie_id"]
    #function that returns a row of given index
    def get(self, index):
        return self.data.loc[index]
    #function that returns data frame of the csv file
    def get_data_frame(self):
        return self.data
