import pandas as pd

def filter_user(user_df):
    null_age = user_df.ageRange.isnull()
    null_gender = user_df.gender.isnull()
    null_style = user_df.travelStyle.isnull()

    user_filtered = user_df[user_df.totalPoints > 1000][~null_age][~null_gender][~null_style]

    user_filtered = user_filtered[['username', 'ageRange', 'gender', 'travelStyle']]
    return user_filtered


def filter_review(review_df):

    attraction_only = review_df.type == 'Attractions'
    filtered_review_df = review_df[['id', 'username', 'type', 'text', 'rating', 'taObjectCity']]
    filtered_review_df = filtered_review_df[attraction_only]

    return filtered_review_df

def merge_review_and_user(user_df, review_df):

    merged_df = pd.merge(review_df, user_df, on=['username'])

    return merged_df

def foreign_review_filter(merged_df):

    span_mask1 = (merged_df.username == 'AnaS1')
    span_mask2 = (merged_df.username == 'DaniLK')
    span_mask3 = (merged_df.username == 'Aprile_24')
    non_city_mask = (merged_df.taObjectCity == 'California')

    merged_df = merged_df[~span_mask1][~span_mask2][~span_mask3][~non_city_mask]

    return merged_df

def filter_final(merged_df, popular_city_list):

    final_df = merged_df[merged_df.taObjectCity.isin(popular_city_list)]

    return final_df


def prep_als_df(final_df):

    asl_temp_df = final_df[['username', 'taObjectCity', 'rating']]

    return asl_temp_df
