import pandas as pd

# general filtering
def filter_user(user_df):
    null_age = user_df.ageRange.isnull()
    null_gender = user_df.gender.isnull()
    null_style = user_df.travelStyle.isnull()
    one_thou_pt = (user_df.totalPoints > 1000)

    user_filtered = user_df[one_thou_pt][~null_age][~null_gender][~null_style]

    user_filtered = user_filtered[['username', 'ageRange', 'gender', 'travelStyle']]
    return user_filtered


def filter_review(review_df):

    attraction_only = review_df.type == 'Attractions'
    filtered_review_df = review_df[['id', 'username', 'type', 'title', 'text', 'rating', 'taObjectCity']]
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



# for user feature matrix


def user_feature_filter(final_df):

    feature_temp = final_df[['username', 'ageRange', 'gender', 'travelStyle']]
    feature_temp = feature_temp.drop_duplicates()

    return feature_temp

def travel_style(feature_temp):

    style_lst = [item.split(', ') for item in feature_temp.travelStyle]
    style_serie = pd.Series(style_lst)

    feature_temp['new_travel'] = style_serie.values

    return feature_temp

def travel_matrix(feature_temp):


    style_matrix = feature_temp['new_travel'].apply(pd.Series)
    style_df = pd.get_dummies(style_matrix.apply(pd.Series). \
                  stack()).sum(level=0). \
                  rename(columns = lambda x : x)

    return style_df

def age_gender_dummie(feature_temp):

    feature_temp = pd.get_dummies(feature_temp, \
                                  columns = ['ageRange', 'gender'])

    return feature_temp

def combine_all_dummies(left, right):

    feature_temp = left.join(right)
    feature_final = feature_temp.drop(['travelStyle', \
                                       'new_travel', \
                                       'gender_male', \
                                       '60+ Traveler', \
                                       'username'], axis =1)

    feature_final.reset_index(drop=True, inplace=True)


    return feature_final


# prep for clustering


def cluster_prep_filter(final_df):

    cluster_input_df = final_df.copy()
    cluster_input_df = cluster_input_df[['title', 'text', 'taObjectCity']]

    return cluster_input_df


def grouping_city_title(cluster_input_df):
    df_title_comb = cluster_input_df.groupby(['taObjectCity']). \
                            apply(lambda x: ' '. \
                            join(x.title)). \
                            reset_index()

    return df_title_comb



def grouping_city_text(cluster_input_df):
    df_text_comb = cluster_input_df.groupby(['taObjectCity']). \
                                    apply(lambda x: ' '. \
                                    join(x.text)). \
                                    reset_index()
    return df_text_comb


def merging_content(left, right):

    cluster_input_df = left.merge(right, on= 'taObjectCity')
    cluster_input_df.columns = ['taObjectCity','title','text']
    cluster_input_df.set_index(['taObjectCity'], drop=True, inplace=True)

    return cluster_input_df


# selected cities from cluster



def selected_cities_in_cluster(cluster_df, city_cluster_idx):


    cluster2_mask = (cluster_df['cluster_k'] == city_cluster_idx)
    up_cluster_df = cluster_df[cluster2_mask]
    up_cluster_df.columns = ['cluster_k', 'taObjectCity']

    return up_cluster_df

def selected_city_df(up_cluster_df, city_df):

    selected_df = pd.merge(up_cluster_df, city_df, how = 'left', on = 'taObjectCity')

    return selected_df
