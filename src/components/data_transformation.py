import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


##  TO GIVE INPUT TO DATA_TRANS CLASS AHEAD, CREATED A PATH TO SAVE THE MODEL AS PICKEL FILE.
@dataclass
class DataTransformationConfig :
    preprocessor_obj_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # To have above class's variable in the instance



    def get_data_transformer_obj(self,train_path): # To create pickle files responsible for data transformation

        """
        This function creates an object that is responsible for data transformation
        """
        try:

            dataset = pd.read_csv(train_path).drop(columns=['math_score'],axis=1) # Since math score is our target variable

            num_features = dataset.select_dtypes(exclude="object").columns.tolist()
            cat_features = dataset.select_dtypes(include="object").columns.tolist()



            num_pipeline = Pipeline(
                steps= [
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaler ', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f' Numerical Columns :{num_features}')
            logging.info(f' Categorical Columns :{cat_features}')

            

            preprocessor = ColumnTransformer(
                [
                ('numerical_pipeline',num_pipeline,num_features),
                ('categorical_pipeline',cat_pipeline,cat_features)
                ]
            )

            


            return preprocessor
        
        except Exception as e :
            raise CustomException(e,sys)
        
        


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('The Train and Test data has been read')

            logging.info(' Obtaining preprocessing object')


            preprocessor_obj = self.get_data_transformer_obj(train_path)

            target_column_name = "math_score"
            

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)       # train_X
            target_feature_train_df = train_df[target_column_name]                            # train_y

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)         # test_X
            target_feature_test_df = test_df[target_column_name]                              # test_y

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)



            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df) ]  # Concatenating (along cols) transformed train and test X's with targets    

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            
            
            logging.info("Saved preprocessing object.")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj                                                    # Saving preproc object because you can use it on other datasets too

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)
            