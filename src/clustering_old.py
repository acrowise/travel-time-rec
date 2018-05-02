import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text



def cluster_texts(texts, clusters=3):
    """
    Transform texts to Tf-Idf coordinates and cluster texts using K-Means
    """
    my_additional_stop_words = ['acute', 'good', 'great', 'really', 'just', 'nice', 'like', 'day']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


    vectorizer = TfidfVectorizer(stop_words= stop_words,
                                 max_features = 800,
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
    kmeans = KMeans(n_clusters=5, random_state=347837898).fit(vectors)
    k_books = pd.DataFrame(list(zip(list(kmeans.labels_),
                                list(texts.index))),
                                columns=['cluster_k', 'city_index'])

    ''' added code to print centriod vocab - Print the top n words from all centroids vocab
    '''
    n = 10
    centroids = kmeans.cluster_centers_
    for ind, c in enumerate(centroids):
        print(ind)
        indices = c.argsort()[-1:-n-1:-1]
        print([cols[i] for i in indices])
        print("=="*20)

    return k_books
