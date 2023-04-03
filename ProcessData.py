'''This class is responsible for reading csv file and process the data it contains'''
import pandas as pd

class ProcessData:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file,sep='\t', lineterminator='\n', header=None)
        columns = ["id","movie_id","review"]
        self.data.columns = columns


    def get_reviews(self):
        return self.data["review"]

    def get_ids(self):
        return self.data["id"]

    def get_movie_ids(self):
        return self.data["movie_id"]

    def get(self, index):
        return self.data.loc[index]
