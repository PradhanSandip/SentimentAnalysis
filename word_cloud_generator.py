import pandas as pd
from ProcessData import ProcessData
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#################################
# Author: Miriam Carvlho        #
#################################


#read csv file, with tab spaced columns. add columns names to the data ["id","movie_id","review"]
data = ProcessData("reviews.csv",True).get_data_frame()
# Group data by movie ID
reviews_by_movie = data.groupby('movie_id')['review'].apply(list).to_dict()




'''function generates wordcloud of given movie id or reviews such as negativc or positve reviews'''
def generate_from_id(movie_id):
    sentence_list = reviews_by_movie[movie_id]
    print(sentence_list)
    text = ""
    #for each review
    for x in sentence_list:
        #if review exist, string = exist, float = does not exist, as pandas return float when column is empty.
        if(type(x) == str):
            text += x
        text += " " 
    # Create word cloud
    wordcloud = WordCloud(width=1200, height=900, background_color='white').generate(text)
    # Plot word cloud
    plt.figure(figsize=(8, 8), facecolor=None)
    #plot the word cloud
    plt.imshow(wordcloud)
    #disable axis
    plt.axis('off')
    #no padding
    plt.tight_layout(pad=0)
    #set title
    plt.title(f'Word cloud for movie {movie_id}')
    #show plot
    plt.show()

'''function that generates word cloud from review(s)'''
def generate_from_review(review, title):
    # Create word cloud
    wordcloud = WordCloud(width=1200, height=900, background_color='white').generate(review)
    # Plot word cloud
    plt.figure(figsize=(8, 8), facecolor=None)
    #display word cloud
    plt.imshow(wordcloud)
    #disable axis
    plt.axis('off')
    #no padding
    plt.tight_layout(pad=0)
    #set tile to movie id
    plt.title(title)
    #show the plot
    plt.show()


    
