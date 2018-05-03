
#general
import pandas as pd
import numpy as np
import pickle
import filtering as filter
# cluster

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

# similarity
from sklearn.metrics import jaccard_similarity_score


class TravelModelMain():

    def __init__(self, user_df, item_df):

        self.user_df = user_df
        self.item_df = item_df
        # predict --------
        #input
        #self.k_recommendation = k_recommendation

    def cluster_texts(self, corpus):
        """
        Transform texts to Tf-Idf coordinates and cluster texts using K-Means
        """
        #my_additional_stop_words = ['acute', 'good', 'great', 'really', 'just', 'nice', 'like', 'day']
        my_additional_stop_words = ['acute', 'good', 'great', 'really', 'just', 'nice',
                                    'like', 'day', 'beautiful', 'visit', 'time', 'don',
                                    'did', 'place', 'didn', 'did', 'tour', 'sydney','pm',
                                    'lot', '00', 'inside', 'istanbul', 'doesn','going',
                                    'right', '15']
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
        kmeans = KMeans(5, random_state=2929292).fit(vectors)
        k_books = pd.DataFrame(list(zip(list(kmeans.labels_),
                                    list(reviews.index))),
                                    columns=['cluster_k', 'city_index'])

        ''' added code to print centriod vocab - Print the top n words from all centroids vocab
        '''
        n = 10
        centroids = kmeans.cluster_centers_
        for ind, c in enumerate(centroids):
            #print(ind)
            indices = c.argsort()[-1:-n-1:-1]
            #print([cols[i] for i in indices])
            #print("=="*20)

        return k_books


    def fit(self, utility_matrix, invert_feature, city_temp, content, reviews):


        self.utility_matrix = utility_matrix
        self.invert_feature = invert_feature
        self.city_temp = city_temp

        vector, cols = self.cluster_texts(content)
        self.cluster_df = self.cluster(vector, cols, reviews)


################################################################################################



    def predict(self, cluster_id, user_id):


        up_cluster_df = filter.selected_cities_in_cluster(self.cluster_df, cluster_id)
        selected_df = filter.selected_city_df(up_cluster_df, self.city_temp)

        final_rating_lst = []

        for city in selected_df.city_id:
            user_i = user_id
            item = city

            als_score = np.dot(self.user_df.features[user_i], self.item_df.features[item])

            final_sim_score = self.jaccard_sim_score(user_i, item, self.invert_feature, self.utility_matrix)
            final_rating = self.overall_rating(als_score, final_sim_score)
            final_pair = (final_rating, item)
            final_rating_lst.append(final_pair)


        rec_cities = self.top_list(final_rating_lst, selected_df)

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



    def top_list(self, final_rating_lst, selected_df):

        print("length of final_rating_list",len(final_rating_lst))
        print(selected_df)

        top_lst = sorted(final_rating_lst, reverse = True)[:3]


        rec_items = []
        for rating, city in top_lst:
            row = selected_df.loc[selected_df['city_id'] == city]
            rec_city = row.taObjectCity.values
            rec_items.append(rec_city[0])

        #return rec_items[0],
        return rec_items
