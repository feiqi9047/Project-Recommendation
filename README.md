# Project-Recommendation
Hybrid Recommendation System for Podcasts using Natural Language Processing and Singular Value Decomposition

## Data Collection

300 podcasts were webscraped from Chartable.com along with the 50 most recent user ratings and reviews for each podcast, compiling a total of 15,000 datapoints. The data are stored in two dataframes:

Podcast Content Dataframe:

- Podcast ID
- Title
- URL
- Genres / Tags
- Description

<img width="898" alt="Screen Shot 2019-06-13 at 1 47 07 PM" src="https://user-images.githubusercontent.com/44821660/59455264-d1e34000-8de1-11e9-8b00-e67fc9d50429.png">

User Rating Dataframe:

- User 
- User Rating (1-5 stars)
- User Review
- URL

<img width="742" alt="Screen Shot 2019-06-13 at 1 48 49 PM" src="https://user-images.githubusercontent.com/44821660/59455351-ff2fee00-8de1-11e9-9897-a15f5117db4b.png">


## Data Cleaning and Feature Engineering

Using LabelEcoder, each podcast and user was assigned their own unique ID that could be references across the dataframes. Additional feature was added using TF-IDF on the podcast descriptions to map the top unique keywords.
All rows where genre and descriptions that were not available were dropped, and lists of genre tags for each podcast were cleaned into category objects. 
Grouping the star ratings for each podcast, an additional column was added to show the average star rating.
The resulting Podcast Content Dataframe is as following:

<img width="895" alt="Screen Shot 2019-06-13 at 2 02 22 PM" src="https://user-images.githubusercontent.com/44821660/59456257-f0e2d180-8de3-11e9-8db0-a1ac21106ae8.png">


## Exploratory Data Analysis

Preliminary overview of the dataset:

<img width="445" alt="Screen Shot 2019-06-13 at 1 57 07 PM" src="https://user-images.githubusercontent.com/44821660/59455912-2804b300-8de3-11e9-9c80-29b4d6ed6604.png">

Per rating distribution, the majority of ratings are five stars (~13,000 ratings), followed by ~1,500 one-star ratings:

<img width="376" alt="Screen Shot 2019-06-13 at 1 55 32 PM" src="https://user-images.githubusercontent.com/44821660/59455806-eecc4300-8de2-11e9-971c-c48321aed488.png">

Most of the users left one review, with the most reviews by one user of up to 9:

<img width="354" alt="Screen Shot 2019-06-13 at 1 57 52 PM" src="https://user-images.githubusercontent.com/44821660/59455968-479bdb80-8de3-11e9-9c5b-6fc9ce1e6ad2.png">
<img width="375" alt="Screen Shot 2019-06-13 at 1 57 59 PM" src="https://user-images.githubusercontent.com/44821660/59455969-479bdb80-8de3-11e9-99d1-ab0c8b15113f.png">

To explore Natural Language Processing algorithms on podcast descriptions, I used Gensim's Word2Vec and TF-IDF to extract the most 'significant' keywords from each text. To illustrate, I used the below podcast as an example:

<img width="892" alt="Screen Shot 2019-06-13 at 2 05 16 PM" src="https://user-images.githubusercontent.com/44821660/59456544-8ed69c00-8de4-11e9-97b4-bcb15218912d.png">

To visualize a 2-dimensional PCA model of the word vectors for 'Armchair Expert with Dax Shepard':

<img width="890" alt="Screen Shot 2019-06-13 at 2 05 23 PM" src="https://user-images.githubusercontent.com/44821660/59456545-8ed69c00-8de4-11e9-8d35-288805aef946.png">

To visualize the tf-idf matrix for 'Armchair Expert with Dax Shepard':

<img width="904" alt="Screen Shot 2019-06-13 at 2 05 30 PM" src="https://user-images.githubusercontent.com/44821660/59456546-8f6f3280-8de4-11e9-81fd-f9ed7c76150b.png">

## Baseline Singular Value Decomposition Model

Formatting the dataframe to parse into the Surprise algorithm as 3 columns: User ID, Podcast ID, and Star Rating.
Splitting the dataset into 80% training data and 20% testing data, fitting the model by training on the training set, and predicting on the test set provided the following accuracy metric:

<img width="120" alt="Screen Shot 2019-06-13 at 2 13 12 PM" src="https://user-images.githubusercontent.com/44821660/59456926-6bf8b780-8de5-11e9-8967-618600d568af.png">

An RMSE of 1.1381 indicates that the predicted rating could be off by 1.1 star.

Testing out the SVD model by predictions for the first 10 users:

<img width="767" alt="Screen Shot 2019-06-13 at 2 13 18 PM" src="https://user-images.githubusercontent.com/44821660/59456927-6bf8b780-8de5-11e9-8b18-2bae67009562.png">


## Baseline LightFM Model

To parse my data into the LighFM model, I needed to create the following interaction matrix for User ID and Podcast ID:

<img width="645" alt="Screen Shot 2019-06-13 at 2 17 42 PM" src="https://user-images.githubusercontent.com/44821660/59457235-0953eb80-8de6-11e9-92f4-c8b7fbe13c48.png">

As evident, my matrix seems very sparse.

After fitting the model on the training data, I obtained a high AUC but a very low precision score, as well as a problem of overfitting on the test set:

<img width="359" alt="Screen Shot 2019-06-13 at 2 21 42 PM" src="https://user-images.githubusercontent.com/44821660/59457534-97c86d00-8de6-11e9-9471-b339aed02ce4.png">

Using the LightFM model to make predictions:

<img width="404" alt="Screen Shot 2019-06-13 at 2 19 33 PM" src="https://user-images.githubusercontent.com/44821660/59457421-59cb4900-8de6-11e9-8dd7-b99ef0382641.png">

For an item that a user has already rated, the recommendations are provided above.

<img width="370" alt="Screen Shot 2019-06-13 at 2 19 46 PM" src="https://user-images.githubusercontent.com/44821660/59457422-59cb4900-8de6-11e9-8b25-1f12db4d2d86.png">

For an item that a user is interested in but has not rating, the recommendations are show above.

## Hybrid Recommendation Model

I decided to use a layering approach to construct a hybrid recommender, using the base layer of genre tags and description TF-IDF keyword Cosine Similarity, and then adding the SVD model on top. 

The first layer of using genre tags provided the following results for "Comedy" genres, sorted by avg rating:

<img width="890" alt="Screen Shot 2019-06-13 at 2 25 32 PM" src="https://user-images.githubusercontent.com/44821660/59457846-39e85500-8de7-11e9-936d-7e36b3e58614.png">


Then, I layered in the content-based component by calculating the dot product using TF-IDF vectorizer to get the Cosine Similarity scores between podcast descriptions. Since keywords vary greatly between each podcast, I decided to give genre tags 3x more weight than individual keywords. If I searched for 'The Shrink Next Door', I would received the following recommendations:

<img width="889" alt="Screen Shot 2019-06-13 at 2 28 47 PM" src="https://user-images.githubusercontent.com/44821660/59458016-95b2de00-8de7-11e9-86eb-e14831cf3752.png">

As evident, the recommendations are much more driven by genre tags than description keywords. However, this content based engine suffers from some severe limitations. First, it is only capable of suggesting podcasts that are very similar, without taking into consideration a user's personal preference. Second, it is highly dependent on what genre tags the podcast is labelled with. 

To take into account user preferences, I layered the SVD component into the hybrid recommender by adding a new column into the dataframe of what the estimated rating that a particular user would give for each podcast, given that he/she has taken interesting in at least one other podcast. 
For example, if a new user has taken interest in "The Joe Rogan Experience", my model would recommend the following based on what he/she is estimated to rate podcasts with similar genre/descriptions:

<img width="897" alt="Screen Shot 2019-06-13 at 2 39 16 PM" src="https://user-images.githubusercontent.com/44821660/59458725-0c041000-8de9-11e9-8edf-2e7a2cd23f97.png">

Another example would be if a new user has taken interest in "TED Talks Science and Medicine", the recommendations would be as follows:

<img width="892" alt="Screen Shot 2019-06-13 at 2 40 59 PM" src="https://user-images.githubusercontent.com/44821660/59458824-4c638e00-8de9-11e9-9b0a-6412f22b2519.png">
 
