import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split




from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation,DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig,ModelTrainer


@dataclass
class DataInjestionConfig:
    train_data_path :str = os.path.join('artifact','train.csv')
    test_data_path :str = os.path.join('artifact','test.csv')
    raw_data_path :str = os.path.join('artifact','raw.csv')

class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig()

    def initiate_injestion(self):
        logging.info(" Entered the Data Injestion Method or Component")

        try:
            df = pd.read_csv('notebook/data/stud.csv')
            
            logging.info("Read the dataset as DataFrame")

           

            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.injestion_config.raw_data_path,index=False)

            logging.info( " Train Test Split initialized")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.injestion_config.train_data_path,index=False)

            test_set.to_csv(self.injestion_config.test_data_path,index=False)

            logging.info('Injestion Completed')

            return(

                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

# if __name__ =='__main__' :
#     obj = DataInjestion()
#     train_path,test_path = obj.initiate_injestion()

#     data_transformation = DataTransformation()
#     train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_path,test_path)

#     model_trainer = ModelTrainer()
#     print(model_trainer.initiate_model_trainer(train_arr,test_arr))
