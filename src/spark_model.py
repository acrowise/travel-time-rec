#general
import pandas as pd
import numpy as np


# spark
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
import pyspark
from pyspark.sql import SQLContext, Row

# private modules
import loader as load
import filtering2 as filter
#import general as gn
import als_model_prep as prep


def spark_rdd(util_matrix):
    # Build our Spark Session and Context
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    spark, sc
    sqlContext = SQLContext(sc)


    schema = StructType( [
    StructField('user', IntegerType(), True),
    StructField('city', IntegerType(), True),
    StructField('rating', FloatType(), True)]
    )

    spark_df = sqlContext.createDataFrame(util_matrix, schema)

    return spark_df

def ALS_model(spark_df):

    #train, test = spark_df.randomSplit([0.85, 0.15], seed=427471138)

    als_model = ALS(userCol='user',
                    itemCol='city',
                    ratingCol='rating',
                    nonnegative=True,
                    regParam=0.1,
                    rank=15
                   )

    #als_recommender = als_model.fit(train)
    als_recommender = als_model.fit(spark_df)

    user_factor_df = als_recommender.userFactors.toPandas()
    item_factor_df = als_recommender.itemFactors.toPandas()

    user_factor_df.to_pickle('user_factor_df.pkl')
    item_factor_df.to_pickle('item_factor_df.pkl')



if __name__ == '__main__':
    # loading dataframe
    reviews_file = 'data/reviews_32618_for_1098_users_with_location.xlsx'
    user_path = 'data/users_full_7034.xlsx'

    u_df = load.load_user_profile(user_path)
    r_df = load.load_reviews(reviews_file)


    # filtering dataframe
    u_filtered = filter.filter_user(u_df)
    r_filtered = filter.filter_review(r_df)
    merge_filtered = filter.merge_review_and_user(u_filtered, r_filtered)
    merge_filtered = filter.foreign_review_filter(merge_filtered)
    pop_city_lst = filter.popular_city_list(merge_filtered)
    final_df = filter.filter_final(merge_filtered, pop_city_lst)


    # prep df for spark rdd
    als_temp_df = prep.prep_als_df(final_df)
    user_temp = prep.unique_user_id(als_temp_df)
    city_temp = prep.unique_city_id(als_temp_df)
    # with username, cityname, rating, user_id, city_id
    result_df = prep.merging_unique_user_city(als_temp_df, user_temp, city_temp)
    # with user_id, city_id, rating - aggregated by cityid and userid
    util_matrix = prep.utility_matrix(result_df)


    spark_df = spark_rdd(util_matrix)
    ALS_model(spark_df)

    print("DONE MODELING! "*5)
