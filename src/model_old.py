# spark
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
import pyspark
from pyspark.sql import SQLContext, Row

# similarity
from sklearn.metrics import jaccard_similarity_score



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

    train, test = spark_df.randomSplit([0.85, 0.15], seed=427471138)

    als_model = ALS(userCol='user',
                    itemCol='city',
                    ratingCol='rating',
                    nonnegative=True,
                    regParam=0.1,
                    rank=15
                   )

    als_recommender = als_model.fit(train)
    predictions = als_recommender.transform(test)


    return predictions


def prediction_pd_df(predictions):
    df_predictions = predictions.toPandas()
    df_predictions.fillna(0,inplace=True)

    return df_predictions



def jaccard_sim_score(udi, cid, user_matrix, util_matrix):
    '''
    takes in user(index) and item
    returns jaccard similarity score
    '''
    overall_rating = 0
    overall_sim = 0
    final_score = 0

    filtered_user = util_matrix[util_matrix.city_id == cid]
    #print(filtered_user)
    for user in filtered_user.user_id.values:
        sim_score = jaccard_similarity_score(user_matrix[udi], user_matrix[user])
        rating = filtered_user[(filtered_user.user_id == user)].rating.values[0]
        overall_rating += sim_score * rating
        overall_sim +=sim_score


    final_score = overall_rating / overall_sim

    return final_score


def overall_rating(als_score ,jacc_sim_score):
    alpha = 0.5
    beta = 0.5
    if als_score.shape[0] == 0:
        final_score = jacc_sim_score
    else:
        final_score = 0.5 * jacc_sim_score + 0.5 * als_score

    return final_score
