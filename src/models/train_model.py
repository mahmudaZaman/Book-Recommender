from sklearn.metrics.pairwise import cosine_similarity
import pickle

from src.features.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation


class ModelTrainer:
    def initiate_model_trainer(self, pt):
        similarity_scores = cosine_similarity(pt)
        pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))

def run_train_pipeline():
    obj = DataIngestion()
    final_ratings = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    pt = data_transformation.initiate_data_transformation(final_ratings)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(pt)


if __name__ == '__main__':
    run_train_pipeline()
    print("model run successfully")