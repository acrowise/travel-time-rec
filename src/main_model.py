# model main temp

#general
import pandas as pd
import numpy as np

# spark
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
import pyspark
from pyspark.sql import SQLContext, Row

# cluster

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

# similarity
from sklearn.metrics import jaccard_similarity_score

# private modules
import loader as load
import filtering as filter
import general as gn
import als_model_prep as prep


class TravelModelMain(object):


    def __init__(self, n_clusters=5):
        # fit --------
        # spark
        # self.util_matrix = util_matrix
        # self.invert_feature = invert_feature
        # self.city_temp = city_temp
            #self.spark_df = spark_df

        # cluster
        # self.texts = content
        # self.reviews = reviews
        self.n_clusters = n_clusters


        #selected matrix

            #self.df_predictions = df_predictions  #spark -> pandas prediction df
            #self.selected_df = selected_df

        # predict --------
        #input
            #self.k_recommendation = k_recommendation




    def spark_rdd(self, util_matrix):
        # Build our Spark Session and Context
        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        spark, sc
        sqlContext = SQLContext(sc)


        schema = StructType( [
        StructField('user', IntegerType(), True),
        StructField('city', IntegerType(), True),
        StructField('rating', FloatType(), True)]
        )

        spark_df = sqlContext.createDataFrame(util_matrix, schema)

        return spark_df

    def ALS_model(self, spark_df):

        #train, test = spark_df.randomSplit([0.85, 0.15], seed=427471138)

        als_model = ALS(userCol='user',
                        itemCol='city',
                        ratingCol='rating',
                        nonnegative=True,
                        regParam=0.1,
                        rank=15
                       )

        #als_recommender = als_model.fit(train)
        self.als_recommender = als_model.fit(spark_df)
        #predictions = als_recommender.transform(spark_df)


        #return predictions

    #
    # def prediction_pd_df(self, predictions):
    #     df_predictions = predictions.toPandas()
    #     df_predictions.fillna(0,inplace=True)
    #
    #     return df_predictions



    def cluster_texts(self, corpus):
        """
        Transform texts to Tf-Idf coordinates and cluster texts using K-Means
        """
        my_additional_stop_words = ['acute', 'good', 'great', 'really', 'just', 'nice', 'like', 'day']
        stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


        vectorizer = TfidfVectorizer(stop_words= stop_words,
                                     max_features = 400,
                                     lowercase=True)

        tfidf_model = vectorizer.fit_transform(corpus)
        vectors = tfidf_model.toarray()
        cols = vectorizer.get_feature_names()

        return (vectors, cols)


    def cluster(self, vectors, cols, reviews):
        """
        Cluster vecotirzed reviews and create k books a data frame relating the
        k label to the book id
        """
        kmeans = KMeans(self.n_clusters, random_state=2929292).fit(vectors)
        k_books = pd.DataFrame(list(zip(list(kmeans.labels_),
                                    list(reviews.index))),
                                    columns=['cluster_k', 'city_index'])

        ''' added code to print centriod vocab - Print the top n words from all centroids vocab
        '''
        n = 10
        centroids = kmeans.cluster_centers_
        for ind, c in enumerate(centroids):
            print(ind)
            indices = c.argsort()[-1:-n-1:-1]
            print([cols[i] for i in indices])
            print("=="*20)

        return k_books


    def fit(self, util_matrix, invert_feature, city_temp, content, reviews):

        self.util_matrix = util_matrix
        self.invert_feature = invert_feature
        self.city_temp = city_temp


        spark_rdd = self.spark_rdd(self.util_matrix)
        self.ALS_model(spark_rdd)
        #predictions = self.ALS_model(spark_rdd)
        #self.df_predictions = self.prediction_pd_df(predictions)

        vector, cols = self.cluster_texts(content)
        self.cluster_df = self.cluster(vector, cols, reviews)


################################################################################################



    def predict(self, cluster_id, user_id, k_recommendation=3):


        up_cluster_df = filter.selected_cities_in_cluster(self.cluster_df, cluster_id)
        selected_df = filter.selected_city_df(up_cluster_df, self.city_temp)

        final_rating_lst = []

        for city in selected_df.city_id:
            user_i = user_id
            item = city

            user_matrix = self.invert_feature
            utility_matrix = self.util_matrix

        #    als_score = self.df_predictions[(self.df_predictions.user == user_i) & (self.df_predictions.city == item)] \
        #                .prediction

            user_factor_df = self.als_recommender.userFactors.filter('id = ' + str(user_i))
            item_factor_df = self.als_recommender.itemFactors.filter('id = ' + str(item))


            user_factors = user_factor_df.collect()[0]['features']
            item_factors = item_factor_df.collect()[0]['features']

            als_score = np.dot(user_factors, item_factors)

            final_sim_score = self.jaccard_sim_score(user_i, item, user_matrix, utility_matrix)
            final_rating = self.overall_rating(als_score, final_sim_score)
            final_pair = (final_rating, item)
            final_rating_lst.append(final_pair)


        rec_cities = self.top_list(final_rating_lst, selected_df, k_recommendation)

        return rec_cities



    def jaccard_sim_score(self, udi, cid, user_matrix, util_matrix):
        '''
        takes in user(index) and item
        returns jaccard similarity score
        '''
        overall_rating = 0
        overall_sim = 0
        final_score = 0

        filtered_user = util_matrix[util_matrix.city_id == cid]
        #print(filtered_user)
        for user in filtered_user.user_id.values:
            sim_score = jaccard_similarity_score(user_matrix[udi], user_matrix[user])
            rating = filtered_user[(filtered_user.user_id == user)].rating.values[0]
            overall_rating += sim_score * rating
            overall_sim +=sim_score


        final_score = overall_rating / overall_sim

        return final_score


    def overall_rating(self, als_score ,jacc_sim_score):
        alpha = 0.5
        beta = 0.5
        if als_score == 0:
            final_score = jacc_sim_score
        else:
            final_score = 0.5 * jacc_sim_score + 0.5 * als_score

        return final_score



    def top_list(self, final_rating_lst, selected_df, k_recommendation):

        print("length of final_rating_list",len(final_rating_lst))
        print(selected_df)

        top_lst = sorted(final_rating_lst, reverse = True)[:k_recommendation]


        rec_items = []
        for rating, city in top_lst:
            row = selected_df.loc[selected_df['city_id'] == city]
            rec_city = row.taObjectCity.values
            rec_items.append(rec_city[0])

        #return rec_items[0],
        return rec_items





if __name__ == '__main__':
    # loading dataframe
    reviews_file = '../data/reviews_32618_for_1098_users_with_location.xlsx'
    user_file = '../data/users_full_7034.xlsx'
    personality_file = '../data/pers_scores_1098.xlsx'
    article_file = '../data/articles_159.xlsx'

    u_df = load.load_user_profile(user_file)
    #df2 = load.load_articles(article_file)
    r_df = load.load_reviews(reviews_file)
    #df4 = load.load_personality_scores(personality_file)


    # filtering dataframe
    u_filtered = filter.filter_user(u_df)
    r_filtered = filter.filter_review(r_df)
    merge_filtered = filter.merge_review_and_user(u_filtered, r_filtered)
    merge_filtered = filter.foreign_review_filter(merge_filtered)
    pop_city_lst = gn.popular_city_list(merge_filtered)
    final_df = filter.filter_final(merge_filtered, pop_city_lst)


    # prep df for spark rdd
    als_temp_df = prep.prep_als_df(final_df)
    user_temp = prep.unique_user_id(als_temp_df)
    city_temp = prep.unique_city_id(als_temp_df)
    # with username, cityname, rating, user_id, city_id
    result_df = prep.merging_unique_user_city(als_temp_df, user_temp, city_temp)
    # with user_id, city_id, rating - aggregated by cityid and userid
    util_matrix = prep.utility_matrix(result_df)


    # user-feature matrix
    feature_temp_0 = filter.user_feature_filter(final_df)
    feature_temp = filter.travel_style(feature_temp_0)
    style_df = filter.travel_matrix(feature_temp)
    feature_temp_1 = filter.age_gender_dummie(feature_temp_0)
    invert_feature = filter.combine_all_dummies(feature_temp_1, style_df).T



    # clustering
    #prep
    cluster_input_df = filter.cluster_prep_filter(final_df)
    df_title_comb = filter.grouping_city_title(cluster_input_df)
    df_text_comb = filter.grouping_city_text(cluster_input_df)
    cluster_final = filter.merging_content(df_title_comb, df_text_comb)


    reviews = cluster_final.title + ' ' + cluster_final.text
    content = [i for i in reviews]
    desire_clusters = 5


    # ------model--------

    travel_m = TravelModelMain(desire_clusters)
    # ------fit--------

    travel_m.fit(util_matrix, invert_feature, city_temp, content, reviews)

    # --------predict-------
    cluster_id = 1
    user_id = 12
    k_recommendation = 3

    # 1) male, ageRange_50-64,Family Hoilday Maker, History Buff, Thrifty Traveller

    # cluster_id = 0, user_id = 151, ['Kyoto', 'Reykjavik', 'St. Petersburg']
    # cluster_id = 1, user_id = 151, ['Monterey', 'Lancaster', 'Sydney']
    # cluster_id = 2, user_id = 151 ['Yellowstone National Park', 'Bergen', 'Lucerne']
    # cluster_id = 3, user_id = 151 ['Vienna', 'Chattanooga', 'Munich']

    # 2) female, ageRange_25-34, Beach Goer, Nightlife Seeker, Urban Explorer

    # cluster_id = 1, user_id = 183 ['Helsinki', 'Sydney', 'Bridgetown']

    # cluster_id = 3, user_id = 183 ['Vienna', 'Munich', 'Stockholm']


    # 3) #female, ageRange_50-64, Beach Goer, History Buff, Nature Lover,Vegetarian

    # cluster_id = 0, user_id = 30 ['Reykjavik', 'Kyoto', 'Amsterdam']
    # cluster_id = 1, user_id = 30,['Lancaster', 'Salzburg', 'Bridgetown']


    # cluster_id = 1, user_id = 12, ['Lancaster', 'Salzburg', 'Helsinki']


    rec_cities = travel_m.predict(cluster_id, user_id, k_recommendation)
    print(rec_cities)
