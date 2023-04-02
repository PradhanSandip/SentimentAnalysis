import pandas as pd

movie_data = pd.read_csv("reviews.csv",sep='\t', lineterminator='\n')
print(movie_data)