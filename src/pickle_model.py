import sys
sys.path.insert(0, '../src')
import main_model_no_spark

import pandas as pd
import numpy as np
import pickle


# private modules
import loader as load
import filtering as filter
import general as gn
import als_model_prep as prep



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
#desire_clusters = 5


# ------model--------


user_df = pd.read_pickle('user_factor_df.pkl')
item_df = pd.read_pickle('item_factor_df.pkl')
#travel_m = TravelModelMain(desire_clusters)

travel_m = main_model_no_spark.TravelModelMain(user_df, item_df)
# ------fit--------

travel_m.fit(util_matrix, invert_feature, city_temp, content, reviews)

pred = travel_m.predict(4, 30)

print(pred)


#pickle.dump(travel_m, open('samp.p', 'wb'))
