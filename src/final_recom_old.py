# final recommendation

def top_three(final_rating_lst, selected_df):

    #print(len(final_rating_lst))
    print("=" *10)
    top_three_lst = sorted(final_rating_lst, reverse = True)[:3]
    print("+"*10)
    rec_list = []
    for rating, city in top_three_lst:
        row = selected_df.loc[selected_df['city_id'] == city]
        rec_city = row.taObjectCity.values
        print(rec_city)
        rec_list.append(rec_city[0])
    return rec_list
