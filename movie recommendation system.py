import numpy as np 
import pandas as pd

movies = pd.read_csv('C:\\Users\\pc\\OneDrive\\Desktop\\movie recomndation system\\tmdb_5000_movies.csv')
credits = pd.read_csv('C:\\Users\\pc\\OneDrive\\Desktop\\movie recomndation system\\tmdb_5000_credits.csv')

print(movies.head())
print(movies.shape)

print(credits.head())
print(credits.shape)

movies = movies.merge(credits,on='title')
# feature selection
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head())
print(movies.shape)
print("the NULL values in their each respective column in the dataset are : ",movies.shape)

# removing all the columns which have null values 
movies.dropna(inplace=True)
print("the count of null values in the columns after droping them : \n",movies.isnull().sum())
print("the shape of the dataset after dropping the null values : ",movies.shape)

# checking if there are duplicate values
print("the number of duplicate values in the dataset are : ",movies.duplicated().sum())


import ast # ast is used to 
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])        # obtainin only the name from given dictionary
    return L

# processing  genres column
movies.iloc[0]['genres']
movies['genres'] = movies['genres'].apply(convert)
print("after processing genres column  : \n\n")
print(movies.head())

# processing kerwords column
print(movies.iloc[0]['keywords'])
movies['keywords'] = movies['keywords'].apply(convert)
print("after processing keywords column : \n\n")
print(movies.head())

# processing cast column
print(movies.iloc[0]['cast'])
# obtaining top 3 cast members since we have many
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 5:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)
movies.head()

# processing crew column
print(movies.loc[0]['crew'])

# fetching only director from the crew column
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)


# processing overview column   (converting string to to list)
movies.iloc[0]['overview']
movies['overview'] = movies['overview'].apply(lambda x:x.split())
print(movies.iloc[0]['overview'])
print(movies.sample(4))


# now removing space like the below sample 
'yahswanth valavala'
'yashwanthvalavala'

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Concatinate all
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# droping those extra columns
new_data = movies[['movie_id','title','tags']]
# Converting list to str
new_data['tags'] = new_data['tags'].apply(lambda x: " ".join(x))
print(new_data.head())
print(new_data.iloc[0]['tags'])

# Converting to lower case
new_data['tags'] = new_data['tags'].apply(lambda x:x.lower())
print(new_data.head())
print(new_data.iloc[0]['tags'])


import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()
def stems(text):
    T = []
    for i in text.split():
        T.append(ps.stem(i))
    return " ".join(T)


new_data['tags'] = new_data['tags'].apply(stems)
print(new_data.iloc[0]['tags'])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_data['tags']).toarray()
print(vector[0])
print("the shape of vectores is : ",vector.shape)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
# calculating similar movies for the title "th Lego Movie"
new_data[new_data['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new_data[new_data['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_data.iloc[i[0]].title)

print(recommend('Spider-Man 2'))

import pickle
pickle.dump(new_data,open('list_of_movies.pkl','wb'))
pickle.dump(similarity,open('similarities.pkl','wb'))
