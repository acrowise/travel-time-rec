# prep for spark ALS model

import pandas as pd


def prep_als_df(final_df):

    asl_temp_df = final_df[['username', 'taObjectCity', 'rating']]

    return asl_temp_df


def unique_user_id(input_df):

    user_dict_df = pd.DataFrame(input_df.username.unique(), columns = ['username'])
    user_temp = user_dict_df.reset_index()
    user_temp = user_temp.rename(columns = {'index':'user_id'})

    return user_temp


def unique_city_id(input_df):
    city_dict_df = pd.DataFrame(input_df.taObjectCity.unique(), columns = ['taObjectCity'])
    city_temp = city_dict_df.reset_index()
    city_temp = city_temp.rename(columns = {'index':'city_id'})
    return city_temp


def merging_unique_user_city(left_df, right_df_1, right_df_2):

    up_temp_df = pd.merge(left_df, right_df_1, on = 'username')
    result_df = pd.merge(up_temp_df, right_df_2, on = 'taObjectCity')

    return result_df


def utility_matrix(result_df):

    agg_dict = {'rating':'median'}
    util_matrix = result_df.groupby(['user_id', 'city_id']).agg(agg_dict).reset_index()

    return util_matrix
