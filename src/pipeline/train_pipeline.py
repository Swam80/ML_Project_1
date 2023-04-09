import sys

from src.exception import CustomException

from  src.components.data_injestion import DataInjestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import   ModelTrainer



class TrainPipeline:

    try:

        def raw_data_split(self):

            injestion_obj = DataInjestion()

            train_path,test_path = injestion_obj.initiate_injestion()

            return train_path,test_path


        def pre_process(self):

            train_path,test_path = self.raw_data_split()

            preproc_obj = DataTransformation()

            train_arr,test_arr,_ = preproc_obj.initiate_data_transformation(train_path,test_path)

            return train_arr,test_arr

        def model_train(self):

            train_arr,test_arr = self.pre_process()

            trainer_obj = ModelTrainer()
            r2_sc,best_model_name,best_parameters_best_model = trainer_obj.initiate_model_trainer(train_arr,test_arr)

            return r2_sc,best_model_name,best_parameters_best_model
        
    
    except Exception as e:
        raise CustomException(e,sys)
        


if __name__ == '__main__':
    train_pipeline = TrainPipeline()
    r2_sc = train_pipeline.model_train()
    print(r2_sc)



