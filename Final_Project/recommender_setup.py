import pandas as pd
import numpy as np
from IPython.display import display # Allows the use of display() for DataFrames
import time
import pickle #To save the objects that were created using webscraping
import pprint
from lxml import html
import requests
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from urllib.request import urlopen
from bs4 import BeautifulSoup
from IPython.display import HTML
import re
import urllib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("processed_data.csv")
#print("Initial records of processed_data.csv file")
#print(df.head())



#Build the TFIDF scores
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Plot"])
#print("The TF-IDF matrix has {} rows and {} columns".format(tfidf_matrix.shape[0],tfidf_matrix.shape[1]))

#Get the cosine similarity between each movie
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cos_sim_df = pd.DataFrame(cos_sim,columns=df["Movie_ID"].tolist(),index=df["Movie_ID"].tolist())

URL = pd.read_csv("Movie_Details.csv")

#Get the mapping between available Movie plots and movie IDs
Movie_Map=pd.merge(URL[["Movie","Movie_ID","URL"]],df,how='inner',on=["Movie_ID"])[["Movie","Movie_ID","Plot","URL"]]


def Get_Recommendations(Movie_ID):
    #Get the indices (movie IDs) with highest cosine sim scores
    recommended_idx=np.argpartition(np.array(cos_sim_df[Movie_ID].tolist()), -6)[-6:]
    
    #Convert to a list
    Recommended_Movie_IDs = cos_sim_df.columns[recommended_idx].tolist()
    
    #Prepare a dict and return the recommended movies list
    return dict(zip(Recommended_Movie_IDs,np.array(cos_sim_df[Movie_ID].tolist())[recommended_idx]))


def Get_Available_Images():
    #Get all the available image names (movie IDs which have images)    
    image_files = os.listdir("./images")
    
    #Make sure that we are dealing with movie data files only
    image_files = [i for i in image_files if re.search('[1-9]*\.jpg',i)]
    
    #Define a list to collect the movie IDs
    y = list()
    for i in image_files:
        y.append(int(i.split(".")[0]))
    #Return the list    
    return y


def Display_Recommendations(Recommended_Movies_Dict,Movie_Map,Source_Movie_ID):
    #The following statement will make sure that we sort the movies in the descending order of similarity
    Recommended_Movies = pd.DataFrame(sorted(Recommended_Movies_Dict.items(), key=lambda x: -x[1]))[0].tolist()
    
    #Delete the liked movie from the list (since cosine sim with itself is 1)
    Recommended_Movies = Recommended_Movies[1:]
    
    Recommended_Movies_Plot = dict()
    Recommended_Movies_URL = dict()
    
    for i in Recommended_Movies:
        Recommended_Movies_Plot[i] = Movie_Map[Movie_Map["Movie_ID"] == i]["Plot"].tolist()[0]
        Recommended_Movies_URL[i] = Movie_Map[Movie_Map["Movie_ID"] == i]["URL"].tolist()[0]

    #Get the available movies with images    
    Available_Images_List = Get_Available_Images()
    
    Source_Movie_Name = Movie_Map[Movie_Map["Movie_ID"] == Source_Movie_ID]["Movie"].tolist()[0]
    Source_Plot = Movie_Map[Movie_Map["Movie_ID"] == Source_Movie_ID]["Plot"].tolist()[0]
    Source_URL = Movie_Map[Movie_Map["Movie_ID"] == Source_Movie_ID]["URL"].tolist()[0]
    print("Assuming that the user liked {}:".format(Source_Movie_Name))
    
    #Prepare HTML for display:    
    if Source_Movie_ID in Available_Images_List:
        display(HTML("<table><tr><td><a href='"+str(Source_URL)+\
                     "' target='_blank'><img src='./images/"+str(Source_Movie_ID)+".jpg' title='"+\
                     str(Source_Plot)+"'></a></td></tr></table>" \
            ))        
        
    display_html = ""
    display_values = ""
    for i in Recommended_Movies:
        if i in Available_Images_List:
            display_html = display_html + "<td><a href='"+str(Recommended_Movies_URL[i])+\
            "' target='_blank'><img src='./images/"+str(i)+".jpg' title='"+\
            str(Recommended_Movies_Plot[i])+"'></a></td>"
            display_values = display_values + "<td> Similarity:"+\
            str(Recommended_Movies_Dict[i])+"</td>"
    print("The following movies are recommended:")        
    display(HTML("<table><tr>"+display_html+"</tr><tr>"+display_values+"</tr></table>" \
            ))        


from IPython.display import display, HTML

def Make_Selection(movie_name = ""):
    pd.set_option('display.max_colwidth', -1)
    while movie_name == "":
        if movie_name == "":
            movie_name = input("Enter the movie watched by the user:")
        
    URL_temp = URL[URL.Movie.apply(lambda x: x.lower()).str.contains(movie_name.lower())][["Movie_ID", "Movie", "Year", "Cast"]]
    if len(URL_temp) == 0:
        print("No movies qualified for the enteres search criteria")
    else:    
        display(HTML(URL_temp.to_html(index=False).replace("\\n","<br>")))
        Movie_ID = input("Enter the movie ID:")
        try:
            Movie_ID = int(Movie_ID)
            if Movie_ID not in list(URL_temp.Movie_ID):
                print("You entered a movie ID which is NOT in the displayed movies list.\n Cannot get recommendations!!")
                return
            Recommended_Movies_Dict = Get_Recommendations(Movie_ID)
            Display_Recommendations(Recommended_Movies_Dict,Movie_Map,Movie_ID)
        except:
           print("Movie ID must be an integer and must be one of the displayed IDs")        
                      
        

            
            
#print(Movie_Map.head())            



