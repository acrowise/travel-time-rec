import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text



def cluster_texts(texts):
    """
    Transform texts to Tf-Idf coordinates and cluster texts using K-Means
    """
    my_additional_stop_words = ['acute', 'good', 'great', 'really',
                                'just', 'nice', 'like', 'day', 'ok',
                                'visit', 'did', 'don', 'place', 'london',
                                'paris','san', 'sydney', 'dubai','diego',
                                'didn', 'fun', 'venice','boston', 'chicago',
                                'tour', 'went', 'time', 'vegas', 'museum',
                                'disney', 'barcelona', 'st', 'pm', 'sf',
                                'worth', 'beautiful', 'la', 'interesting',
                                'inside', 'outside', 'experience', 'singapore',
                                'lot', 'free', 'istanbul', 'food', 'people',
                                'way']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


    vectorizer = TfidfVectorizer(stop_words= stop_words,
                                 max_features = 500,
                                 lowercase=True)

    tfidf_model = vectorizer.fit_transform(texts)
    vectors = tfidf_model.toarray()
    cols = vectorizer.get_feature_names()

    return (vectors, cols)


def cluster(vectors, cols, texts):
    """
    Cluster vecotirzed reviews and create k books a data frame relating the
    k label to the book id
    """
    #10000 is not bad
    # 10000000
    kmeans = KMeans(n_clusters=4, random_state=10000000).fit(vectors)
    k_books = pd.DataFrame(list(zip(list(kmeans.labels_),
                                list(texts.index))),
                                columns=['cluster_k', 'city_index'])

    ''' added code to print centriod vocab - Print the top n words from all centroids vocab
    '''
    n = 20
    centroids = kmeans.cluster_centers_
    for ind, c in enumerate(centroids):
        print(ind)
        indices = c.argsort()[-1:-n-1:-1]
        print([cols[i] for i in indices])
        print("=="*20)

    return k_books
