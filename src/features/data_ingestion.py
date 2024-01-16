import pickle
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    books_data_uri: str = "/Users/shuchi/Documents/work/personal/Book-Recommender/dataset/Books.csv"
    user_data_uri: str = "/Users/shuchi/Documents/work/personal/Book-Recommender/dataset/Users.csv"
    rating_data_uri: str = "/Users/shuchi/Documents/work/personal/Book-Recommender/dataset/Ratings.csv"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        books = pd.read_csv(self.ingestion_config.books_data_uri)
        users = pd.read_csv(self.ingestion_config.user_data_uri)
        ratings = pd.read_csv(self.ingestion_config.rating_data_uri)

        ratings_with_name = ratings.merge(books,on='ISBN')

        num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
        num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

        avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
        avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

        popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
        popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
        popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
            ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]
        x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
        active_users = x[x].index
        filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]
        y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
        famous_books = y[y].index
        final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
        books.drop_duplicates('Book-Title')
        pickle.dump(popular_df, open('popular.pkl', 'wb'))
        pickle.dump(books, open('books.pkl', 'wb'))
        return final_ratings
