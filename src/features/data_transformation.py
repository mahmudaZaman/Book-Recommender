import pickle


class DataTransformation:
    def initiate_data_transformation(self, final_ratings):
        pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
        pt.fillna(0, inplace=True)
        pickle.dump(pt, open('pt.pkl', 'wb'))
        return pt
