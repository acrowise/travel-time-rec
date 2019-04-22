# Travel Time: Travel Destination Recommender

- [Overview](#Overview)
- [Data](#Data)
- [Model](#Model)
- [Get your own recommendation!](#Get-your-own-recommendation)
- [Technical details](#Technical-explanations)

![frontpage](https://github.com/kammybdeng/travel-time-rec/blob/master/images/web_frontpage.png)

# Overview
There are tons of great cities around the world, but it's quite difficult for most people to go to all of these well known places. So, I always wonder if there are ways to narrow down the choices. Thus, I decided to build a travel destination recommender, TravelTime, **that will compare new traveler's personality, age, and travel style to other travelers and recommend the most favorable travel destinations for travelers**!

# Data
The data used in this project is collected from [Alexander Roshchina](https://www.researchgate.net/publication/301543515_TripAdvisor_dataset_with_personality_scores)

They are:
1) Tripadvisor users' **Big 5 personality scores**
2) Tripadvisor **users' profiles** (age, gender, travel style, Tripadvisor points, Tripadvisor traveler status, and etc.)
3) Tripadvisor users' city **text reviews and star ratings**

# Model
## Hybrid Recommender
![workflow.png](https://github.com/kammybdeng/travel-time-rec/blob/master/images/model-structure.png)

Like most of the recommenders, this model was trained with the dataset and will analyze new traveler's personality and preferences and find the most matching old users's reviews and ratings, and recommend the matching cities.
In order to provide a more specific recommendation, I also clustered the cities into four different groups.

By answering the two questions on the webpage, **your preferred city type** and **your traveling style**, the model in the backend will generate two matching cities for the traveler.

### Jaccard Similarity
Matches your personality similarity with other travelers

### Collaborative Filtering
Matches your traveling and rating preferences and with other travelers

### Clustering
Categorize the cities into four different groups

# Get your own recommendation!
To run the model and the website in your local computer, you can:

  1. prepare your environment with requirements.txt
  ```
  pip install -r requirements.txt
  ```

  2. run the model
  ```
  python src/spark_model.py
  ```

  3. save up the model
  ```
  python src/pickle_model.py
  ```
  4. run the website
  ```
  python python webapp/server.py
  ```

  5. copy and paste the local website link to your browser


## Technical details
For people who want to learn more about ALS Collaborative Filtering.

### 1. ALS Collaborative Filtering Recommender
The PySpark's ALS model starts with a sparse utility matrix with items listed on the x axis and users listed on the y axis. The utility matrix will then break into two smaller U, V matrixes. These two small, dense matrixes (often with reduced rank) will then combined to reform a new, dense utility matrix with filled in predicted ratings.

###  2. Hybrid Recommender System
In order to improve the accuracy of the ALS model. I decided to add in another user-user jaccard similarity matrix. The Jaccard similarity are calculated based on the number of common features between two users. A customized Jaccard similarity function is created to return predicted ratings for all the cities based on user similarities.

The final predicted rating of my model is an addition of the alpha times the ALS model and the beta times the user-user Jaccard similarity matrix.

### 3. Kmeans clustering
Besides accuracy, I also want to personalize the recommendation. Therefore, I used k-Means clustering on the vectorized TFIDF city reviews to create distinct city groups.

### 4. Travel Class Model
Finally, a class model is generated to combine the clusters and the city rating predictions together. The model is represented by the diagram above.
