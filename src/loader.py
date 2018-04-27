import pandas as pd


# traveler profile
def load_user_profile(path):

    df = pd.read_excel(path)

    return df

# traveler articles
def load_articles(path):

    df = pd.read_excel(path)

    return df
# traveler and reviews
def load_reviews(path):

    df = pd.read_excel(path)

    return df
# articles by some traveler
def load_personality_scores(path):

    df = pd.read_excel(path)

    return df
