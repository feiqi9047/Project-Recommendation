import seaborn as sns
import pandas as pd
import numpy as np
import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

df_w_dummies = pd.read_csv('Project_Files/df_w_dummies.csv')
df_cont = df_w_dummies[['title', 'podcast_id','genre_tags', 'desc', 'url', 'tf-idf']]
df_cont.drop_duplicates(inplace=True)

df_stars = pd.DataFrame(df_w_dummies.groupby('podcast_id')['stars'].mean())

md = df_cont.merge(df_stars, left_on='podcast_id', right_on='podcast_id')
md['genre_tags'] = md['genre_tags'].map(lambda i: i.split(','))

# transpose the dataframe to allow each genre its own column

s = md.apply(lambda x: pd.Series(x['genre_tags']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genre_tags', axis=1).join(s)
gen_md.head(3).transpose()

# provide top recommendations for each genre

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]

    qualified = df[(df['stars'].notnull())][['title', 'desc', 'url','stars','genre']]
    qualified = qualified.sort_values('stars', ascending=False).head(250)

    return qualified

# build a recommender using the podcast descriptions.

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(md['desc'])

# used the TF-IDF Vectorizer, calculating the Dot Product will directly give the Cosine Similarity Score.
# use sklearn's linear_kernel instead of cosine_similarities since it is much faster

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

md = md.reset_index(drop=True)
titles = md[['title', 'genre_tags', 'stars']]
indices = pd.Series(md.index, index=md['title'])

# We now have a pairwise cosine similarity matrix for all the movies in our dataset.
# The next step is to write a function that returns the 30 most similar movies based on the cosine similarity score.

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    podcast_indices = [i[0] for i in sim_scores]
    return titles.iloc[podcast_indices]


md['desc_keywords'] = md['desc'].apply(lambda x: x.split(' '))

# giving the genre tags 3x the weight
md['genre_tags_x_3'] = md['genre_tags'].apply(lambda x: x*3)

md['desc_genre'] = md['desc_keywords'] + md['genre_tags_x_3']
md['desc_genre'] = md['desc_genre'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(md['desc_genre'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

md = md.reset_index(drop=True)
titles =  md[['title', 'genre_tags', 'stars']]
indices = pd.Series(md.index, index=md['title'])

def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    podcast_indices = [i[0] for i in sim_scores]

    podcast = md.iloc[podcast_indices][['title', 'desc', 'url','stars','genre_tags']]
    podcast = podcast.sort_values('stars', ascending=False).head(10)
    return podcast

from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split

# for Surprise, we only need three columns from the dataset
data = df_w_dummies[['user_id', 'podcast_id', 'stars']]
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_df(data, reader=reader)

# train-test-split
trainset, testset = train_test_split(data, test_size=.2)

# instantiate SVD and fit the trainset
svd = SVD()
svd.fit(trainset)

predictions = svd.test(testset)

def hybrid(user_id, title):
    idx = indices[title]
    podcast_id = md.loc[md.title==title]['podcast_id']
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    podcast_indices = [i[0] for i in sim_scores]
    podcast = md.iloc[podcast_indices][['title', 'desc', 'url','stars','genre_tags']]
    podcast['est'] = podcast.index.map(lambda x: svd.predict(user_id, md.loc[x]['podcast_id']).est)
    podcast = podcast.sort_values('est', ascending=False)
    return podcast.head(10)
