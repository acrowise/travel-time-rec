from collections import Counter
import pandas as pd

def popular_city_list(merged_df):

    popular_city = []
    for item, value in Counter(merged_df.taObjectCity).items():
        if value > 7:
            popular_city.append(item)
    return popular_city

# prep for spark ALS model

def unique_user_id(als_temp_df):

    als_temp_df_updated = als_temp_df.copy()

    user_dict = {}
    for idx, user in enumerate(als_temp_df.username.unique()):
        user_dict[user] = idx
        print(user_dict)
        break


    user_id_list = [user_dict[item]
                  for user in als_temp_df.username
                  for item, key in user_dict.items()
                  if item == user]
    print(user_id_list[:10])
    als_temp_df_updated['user_id'] = user_id_list

    return  als_temp_df_updated


def unique_city_id(als_temp_df):

    city_dict = {}
    for idx, city in enumerate(als_temp_df.taObjectCity.unique()):
        city_dict[city] = idx

    city_id_list = [city_dict[item]
                    for city in als_temp_df.taObjectCity
                    for item, key in city_dict.items()
                    if item == city]

    als_temp_df['city_id'] = city_id_list

    return als_temp_df


def update_rating_type(als_temp_df):
    als_temp_df = als_temp_df['rating_float'] = pd.to_numeric(als_temp_df.rating, downcast='float')

    return als_temp_df



def prepped_df_spark(als_temp_df):
    als_final_df = als_temp_df[['user_id', 'city_id', 'rating_float']]
    return als_final_df
