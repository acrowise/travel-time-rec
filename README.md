# Travel Time: Travel Destination Recommender

![frontpage](https://github.com/kammybdeng/travel-time-rec/blob/master/images/web_frontpage.png)

[Overview](#Overview)
[Data](#Data)
[Hybrid Recommender](#Hybrid-Recommender)
[Run the model yourself!](#Testing)
[Technical explanations](#Technical-explanations)


# Overview
There are tons of great cities around the world, but it's quite hard for most people to go to all of them. I always wonder if there are ways to narrow down the choices to a few places. Thus, I decided to build a travel destination recommender, TravelTime, that will compare new traveler's personality, age, and travel style to other traveler and select the most favorable travel destination for travelers!

# Data

The data used in this project is collected from [Alexander Roshchina](https://www.researchgate.net/publication/301543515_TripAdvisor_dataset_with_personality_scores)

They are:
1) Tripadvisor users' Big 5 personality scores
2) Tripadvisor users' profiles (age, gender, travel style, Tripadvisor points, Tripadvisor traveler status, and etc.)
3) Tripadvisor users' city text reviews and star ratings


# Hybrid Recommender

![workflow.png](https://github.com/kammybdeng/travel-time-rec/blob/master/images/workflow.png)

## Jaccard Similarity
Matches your personality similarity with other travelers

## Collaborative Filtering
Matches your traveling and rating preferences and with other travelers


# Testing
To run the model and the website in your local computer, you can:

  - **1) clone the repository to your folder**
  then run these commends in your terminal:

  - **2) python src/spark_model.py
  - 3) python src/pickle_model.py
  - 3) python webapp/server.py**


## Technical explanations
For people who want to learn more about how ALS Collaborative Filtering does.

### 1. ALS Collaborative Filtering Recommender
The PySpark's ALS model starts with a sparse utility matrix with items listed on the x axis and users listed on the y axis. The utility matrix will then break into two smaller U, V matrixes. These two small, dense matrixes (often with reduced rank) will then combined to reform a new, dense utility matrix with filled in predicted ratings.

###  2. Hybrid Recommender System
In order to improve the accuracy of the ALS model. I decided to add in another user-user jaccard similarity matrix. The Jaccard similarity are calculated based on the number of common features between two users. A customized Jaccard similarity function is created to return predicted ratings for all the cities based on user similarities.

The final predicted rating of my model is an addition of the alpha times the ALS model and the beta times the user-user Jaccard similarity matrix.

### 3. Clustering
Besides accuracy, I also want to personalize the recommendation. Therefore, I used k-Means clustering on the vectorized TFIDF city reviews to create distinct city groups.

# Travel Class Model
Finally, a class model is generated to combine the clusters and the city rating predictions together. The model is represented by the diagram below.
