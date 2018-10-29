import ast
from datetime import datetime

from sklearn.preprocessing import LabelEncoder

import pandas as pd

import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
import string
from collections import Counter

def process_movies_data(path_movie):
    """
    Clean the movie file and apply different processings to get usable data
    :param path_movie: path of the movie data
    :return: dataframe of processed movie data
    """

    df_movie = pd.read_csv(path_movie, quotechar='"')
    df_movie.rename(columns={"id": "tmdbId"}, inplace=True)
    nltk.download('stopwords')

    df_movie = df_movie.drop_duplicates("tmdbId")

    print("process genre.")
    df_movie = process_genres(df_movie)

    print("process date.")
    # Determine the year with a precision of 10 years
    df_movie['release_date'] = pd.to_datetime(df_movie['release_date'])

    # fill missing values in release date
    mean_year_dataset = df_movie['release_date'].dropna().apply(lambda x: x.year).mean()

    # apply the function estimate year to find or estimate the year of release
    df_movie.loc[df_movie['release_date'] != df_movie['release_date'], 'release_date'] = df_movie[
        df_movie['release_date'] != df_movie['release_date']].apply(
        lambda x: estimate_year(x, df_movie.copy(), mean_year_dataset), axis=1)

    df_movie['year'] = df_movie['release_date'].apply(lambda x: round(x.year, -1))

    print("process language.")
    ## language category encoded
    df_movie = process_language(df_movie)

    print("process status.")
    ## encode status
    df_movie = process_status(df_movie)

    print("process overview.")
    ## find keywords inside the overview
    df_movie = process_overview(df_movie)

    cols_drop = ['adult', 'belongs_to_collection', 'budget', 'revenue',
                 'genres', 'homepage', 'genres', 'runtime', 'original_language', 'original_title',
                 'overview', 'poster_path', 'video',  'production_companies',
       'production_countries','tagline', 'title', 'spoken_languages', 'released_date']

    df_movie.drop(cols_drop, axis=1, inplace=True)

    df_movie = df_movie.fillna(0)

    df_movie.to_csv('processed_movie_data.csv', index = None)

    return df_movie


def estimate_year(x, df_movie, mean_year_dataset):
    """
    Fill the missing release dates by estimating the year
    :param x: row of df_movie with filling release-date
    :param df_movies:dataframe of movies metadata
    :param mean_year_dataset: mean year of release-date in the dataset
    :return:
    """

    if x['status'] == "Planned":
        # if the movie is not release, use the max year
        return datetime(int(max(df_movie['release_date'].apply(lambda x : x.year))), 1, 1)

    else:
        # look at a date in the overview or the title
        for year in range(1930, 2018):
            if str(year) in str(x['title']):
                return datetime(year, 1, 1)
            if str(year) in str(x['overview']):
                return datetime(year, 1, 1)

        # fill with the production company mean of release years
        if (len(ast.literal_eval(x['production_companies'])) >= 1):
            list_same_company = []

            for i in ast.literal_eval(x['production_companies']):
                list_same_company.append(
                    df_movie[df_movie['production_companies'].apply(lambda x: i['name'] in x)])

            df_company = pd.concat(list_same_company)
            mean_year = df_company['release_date'].dropna().apply(lambda x: x.year).mean()
            return datetime(int(round(mean_year, 0)), 1, 1)

        # if nothing else has been found, fill with the dataset mean of years
        return datetime(int(round(mean_year_dataset, 0)), 1, 1)


def process_overview(df_movie):
    """
    Find most common bi-grams in movies overview and make it boolean categories
    :param df_movie:
    :param df_movie: dataframe of movies metadata
    :return: df_movie dataframe of movies metadata with new categories columns
    """
    overview_corpus = ' '.join(df_movie['overview'].astype("str"))
    # delete ponctuation
    translator = str.maketrans('', '', string.punctuation)
    corpus = overview_corpus.lower().translate(translator)
    text = [i for i in corpus.split() if i not in stopwords.words('english')]
    text_bigrams = [i for i in ngrams(text, 2)]
    for bigram in Counter(text_bigrams).most_common(13):
        for i, x in df_movie.iterrows():
            words = bigram[0][0] + " " + bigram[0][1]
            if words in str(x['overview']).lower().translate(translator):
                df_movie.loc[i, bigram[0][0] + "_" + bigram[0][1]] = 1

    return df_movie


def process_language(df_movie):
    """
    For the 20 most represented languages, the category is numerically encoded
    :param df_movie: dataframe of movies metadata
    :return: df_movie dataframe of movies metadata with encoded language
    """
    languages = df_movie.groupby("original_language").size().sort_values(ascending=False)[:20].reset_index()[
        'original_language'].tolist()
    df_movie.loc[df_movie['original_language'].isin(languages) == False, 'original_language'] = "other"
    le_lang = LabelEncoder()
    df_movie['original_language_enc'] = le_lang.fit_transform(df_movie.original_language)
    return df_movie


def process_status(df_movie):
    """
    Fill missing status with the most common one and encode as numerical value
    :param df_movie:
    :return:
    """
    le_status = LabelEncoder()
    most_common_status = df_movie.groupby('status').size().sort_values(ascending=False).reset_index().iloc[0].status
    df_movie.loc[df_movie['status'].isnull(), 'status'] = most_common_status
    df_movie['status'] = le_status.fit_transform(df_movie.status)
    return df_movie


def process_genres(df_movie):
    """
    Determine the 10 most represented genres in the dataset
    Create new columns with the genre names that are boolean, equal to 1 if the movie has this genre
    :param df_movie: dataframe of movies metadata
    :return: dataframe of movies metadata with new genres
    """
    list_all = []
    for i in df_movie["genres"]:
        split_genre = [x['name'] for x in ast.literal_eval(i)]
        for j in split_genre:
            list_all.append(j)
    counter = Counter(list_all)
    dict_count_genre = dict(counter)
    top_genres = sorted(dict_count_genre, key=dict_count_genre.get, reverse=True)[:10]
    for word in top_genres:
        df_movie[word] = df_movie['genres'].map(lambda s: 1 if word in str(s) else 0)

    return df_movie

if __name__ == "__main__":
    path_movie = "../data/movies_metadata.csv"
    path_rating = "../data/ratings.csv"
    links = "../data/links.csv"

    df_movie = process_movies_data(path_movie)
    df_links = pd.read_csv(links)
    df_links = df_links[df_links["tmdbId"].notnull()]

    df_rating = pd.read_csv(path_rating)

    df_movie["tmdbId"] = df_movie["tmdbId"].apply(lambda x: int(x))
    df_links["tmdbId"] = df_links["tmdbId"].apply(lambda x: int(x))

    df_movie = pd.merge(df_movie, df_links, on="tmdbId", how="left")
    df_rating = pd.merge(df_rating, df_movie, on='movieId', how='left')

    df_rating.to_csv("rating_with_movie_data.csv", index = None)




