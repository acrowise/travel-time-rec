import pandas as pd
import os


# traveler profile
def load_user_profile(file_path):
    df = pd.read_excel(file_path)
    return df

# traveler articles
def load_articles(file_path):
    df = pd.read_excel(file_path)
    return df

# traveler and reviews
def load_reviews(file_path):
    df = pd.read_excel(file_path)
    return df

# articles by some traveler
def load_personality_scores(file_path):
    df = pd.read_excel(file_path)
    return df
