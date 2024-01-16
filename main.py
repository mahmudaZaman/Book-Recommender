import os
import numpy as np
import streamlit as st
from pandas import read_pickle
from src.models.train_model import run_train_pipeline

def streamlit_run():
    # Load pickled data
    pt = read_pickle(open('/Users/shuchi/Documents/work/personal/Book-Recommender/src/models/pt.pkl', 'rb'))
    books = read_pickle(open('/Users/shuchi/Documents/work/personal/Book-Recommender/src/models/books.pkl', 'rb'))
    similarity_scores = read_pickle(open('/Users/shuchi/Documents/work/personal/Book-Recommender/src/models/similarity_scores.pkl', 'rb'))

    # Create Streamlit UI
    st.title("Book Recommendation System")
    user_input = st.text_input("Enter a book title:")
    if user_input:
        # Find the index of the user input in the pt DataFrame
        index = np.where(pt.index == user_input)[0]
        if index:
            index = index[0]  # Take the first index if multiple matches
            # Find similar items based on similarity_scores
            similar_items = sorted(enumerate(similarity_scores[index]), key=lambda x: x[1], reverse=True)[1:6]
            data = []
            for i in similar_items:
                item = []
                temp_df = books[books['Book-Title'] == pt.index[i[0]]]
                item.extend(temp_df.drop_duplicates('Book-Title')['Book-Title'].to_list())
                item.extend(temp_df.drop_duplicates('Book-Title')['Book-Author'].to_list())
                item.extend(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].to_list())
                data.append(item)
            st.write("5 books are recommended:")
            for i, book in enumerate(data, 1):
                st.write(f"{i}. {book}")
                # st.image(book[2], caption=f"Image of {book[2]}", use_column_width=True)
                print("book===========", book[0])


def model_run():
    run_train_pipeline()


if __name__ == '__main__':
    mode = os.getenv("mode", "streamlit")
    print("mode", mode)
    if mode == "model":
        model_run()
    else:
        streamlit_run()
