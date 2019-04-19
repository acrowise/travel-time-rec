from collections import Counter
import pandas as pd

def popular_city_list(merged_df):

    popular_city = []
    for item, value in Counter(merged_df.taObjectCity).items():
        if value > 7:
            popular_city.append(item)
    return popular_city
