import loader as load
import filtering as filter
import general as gn
import model

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

    print(final_df.shape)



    # clustering 


    # prep df for spark rdd
    asl_temp_df_1 = filter.prep_als_df(final_df)
    asl_temp_df_2 = gn.unique_user_id(asl_temp_df_1)
    print(asl_temp_df_2.head())
    # asl_temp_df_3 = gn.unique_city_id(asl_temp_df_2)
    # asl_temp_df_4 = gn.update_rating_type(asl_temp_df_3)
    #
    #
    # als_final_df = gn.prepped_df_spark(asl_temp_df_4)
    #
    # print(als_final_df.head())


    # creating ALS model with spark
    #model.ALS_model(asl_temp_df)
