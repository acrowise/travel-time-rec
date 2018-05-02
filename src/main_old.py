import loader as load
import filtering as filter
import general as gn
import als_model_prep as prep
import model as m
import clustering as cls
import final_recom as final


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


    # building ALS ALS

    spark_rdd = m.spark_rdd(util_matrix)
    predictions = m.ALS_model(spark_rdd)
    df_predictions = m.prediction_pd_df(predictions)

    # travel = TravelModelMain()
    # travel.fit(util_matrix)
    # travel.predict(cluster_id, user_id)




    # user-feature matrix
    feature_temp_0 = filter.user_feature_filter(final_df)
    feature_temp = filter.travel_style(feature_temp_0)
    style_df = filter.travel_matrix(feature_temp)
    feature_temp_1 = filter.age_gender_dummie(feature_temp_0)
    invert_feature = filter.combine_all_dummies(feature_temp_1, style_df).T
    #print(invert_feature.head())

    # hybrid model

    # user_i = 200
    # item = 22
    # user_matrix = invert_feature
    # utility_matrix = util_matrix
    #
    # als_score = predictions[(predictions.user == str(user_i)) & (predictions.city == str(item))] \
    #         .select("prediction").collect()[0][0]
    # final_sim_score = m.jaccard_sim_score(user_i, item, user_matrix, util_matrix)
    #
    # final_rating = m.overall_rating(als_score, final_sim_score)

    #print(final_rating)

    # clustering
    #prep
    cluster_input_df = filter.cluster_prep_filter(final_df)
    df_title_comb = filter.grouping_city_title(cluster_input_df)
    df_text_comb = filter.grouping_city_text(cluster_input_df)
    cluster_final = filter.merging_content(df_title_comb, df_text_comb)


    reviews = cluster_final.title + ' ' + cluster_final.text
    content = [i for i in reviews]
    n = 5
    corpus = content

    # clusters
    vector, cols = cls.cluster_texts(corpus, n)
    cluster_df = cls.cluster(vector, cols, reviews)


    # selected cities in clusters
    # ranges 0-4
    city_type = 4

    up_cluster_df = filter.selected_cities_in_cluster(cluster_df, city_type)
    selected_df = filter.selected_city_df(up_cluster_df, city_temp)



    # rates for these selected cities
    final_rating_lst = []

    #pickle.dumps(model, 'model.1')


    for city in selected_df.city_id:
        user_i = 50 #error when user =  123
        item = city
        user_matrix = invert_feature
        utility_matrix = util_matrix

        als_score = df_predictions[(df_predictions.user == user_i) & (df_predictions.city == item)] \
                    .prediction
        final_sim_score = m.jaccard_sim_score(user_i, item, user_matrix, util_matrix)

        final_rating = m.overall_rating(als_score, final_sim_score)

        final_pair = (final_rating, item)

        final_rating_lst.append(final_pair)
    print(selected_df)
    #selected_df['final_rating'] = final_rating_lst
    #print('last of final_rating_lst:', final_rating_lst[-1])
    print("=" *10)
    #print(final_rating_lst)
    #print(selected_df.head())


    rec_cities = final.top_three(final_rating_lst, selected_df)

    print(rec_cities)
