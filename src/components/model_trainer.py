import os
import sys

from dataclasses import dataclass
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor



from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:

    train_model_path = os.path.join('artifact','model.pkl')
    model_report_path = os.path.join('artifact','model_report.csv')

class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            logging.info(" Splitting training and test input data")

            X_train,y_train,X_test,y_test = (

                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                
                )
            
            '''
            SETTING UP MODELS AND HYPERPARAMETERS
            '''

            models_params= {
                "Decision Tree": 
                            {
                            'model_obj' : DecisionTreeRegressor(),
                            'params' : {
                                                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                                    'max_depth':[5,10,15,20],
                                                    'ccp_alpha' :[0.0,0.005,0.01,0.02,0.03]
                                                    # 'max_features':['sqrt','log2'],
                                                    },
                        },
                
                "Random Forest": 
                            {
                            'model_obj' : RandomForestRegressor(),
                            'params' : {
                                                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                                        'max_features':['sqrt','log2',None],
                                                        'n_estimators': [8,16,32,64,128,256]
                                                    },
                            },

                "Gradient Boosting": 
                            {
                            'model_obj' : GradientBoostingRegressor(),
                            'params' : {
                                                        'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                                                        'learning_rate':[0.1,.01,.05,.001],
                                                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                                                        # 'criterion':['squared_error', 'friedman_mse'],
                                                        # 'max_features':['auto','sqrt','log2'],
                                                        'n_estimators': [8,16,32,64,128,256]
                                                        },
                            },

                "Linear Regression":
                            {
                            'model_obj' : LinearRegression(),
                            'params' : {}
                            },

                "XGBRegressor":
                            {
                            'model_obj' : XGBRegressor(),
                            'params' : {
                                                        'learning_rate':[.1,.01,.05,.001],
                                                        'n_estimators': [8,16,32,64,128,256]
                                                    },
                            },

                "CatBoosting Regressor": 
                            {
                            'model_obj' : CatBoostRegressor(verbose=False),
                            'params' :     {
                                                    'depth': [6,8,10],
                                                    'learning_rate': [0.01, 0.05, 0.1],
                                                    'iterations': [30, 50, 100]
                                                    },
                            },

                "AdaBoost Regressor": 
                            {
                            'model_obj' : AdaBoostRegressor(),
                            'params' :     {
                                                    'learning_rate':[.1,.01,0.5,.001],
                                                    'loss':['linear','square','exponential'],
                                                    'n_estimators': [8,16,32,64,128,256]
                                                    },
                            },
                "Lasso": 
                    {
                        'model_obj' : Lasso(),
                        'params' :{
                                            'alpha': [.1,.01,.05,.001,0.0001],
                                            'max_iter' : [10000]
                                        },
                    },

                "Ridge" : 
                    {
                        'model_obj' : Ridge(),
                        'params': {
                                            'alpha': [.1,.01,.05,.001,0.0001],
                                            'max_iter' : [10000]
                                        },
                    },

                "KNN" :
                    {
                        'model_obj' : KNeighborsRegressor(),
                    'params' : {
                                    'n_neighbors' : [3,5,8,11,15,20]
                                    },
                    },
                'SVR' :{
                        'model_obj' : SVR(),
                        'params':{
                                        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                                        'gamma' :['scale','auto'],
                                        'C': [0.1,1, 10, 100]
                                    },
                },
                
            }
            
            logging.info(" Models and Hyperparameters have been set")

            
            '''
            GET MODEL REPORT
            '''
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models_params=models_params)
            
            '''
            Create DataFrame of the Model Report in descending order
            '''
            model_report_df = pd.DataFrame(model_report).sort_values(by= 'Tuned_Score',ascending = False)


            '''
             TO GET BEST MODEL SCORE
            '''
            best_model_score = max(model_report_df.Tuned_Score)   


            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)


            '''
            TO GET BEST MODEL NAME AND BEST PARAMETERS
            '''
            
            best_model_name = model_report_df.loc[model_report_df.Tuned_Score == best_model_score]['Model_name'].tolist()[0]   #[0] to get first element of the list as string

            best_parameters_best_model = model_report_df.loc[model_report_df.Tuned_Score == best_model_score]['Best_Params'].tolist()[0]


            '''
            Saving the model object of best model with best parameters as pickle file
            '''

            # Since model with best parameters was already fitted in evaluate function in utils.py, we need not use set_params again.
            # We can directly use predict.
        
            best_model = list(models_params[best_model_name].values())[0]

            predicted_y = best_model.predict(X_test)
            r2_sc = r2_score(y_test,predicted_y)

            save_object(
                file_path=self.model_trainer_config.train_model_path,
                obj =  best_model,
                
                )
            
            logging.info('Best Model found on test dataset')


            '''
            SAVING MODEL REPORT AS CSV FILE
            '''

            model_report_df.to_csv(self.model_trainer_config.model_report_path,index=False)
            

            return r2_sc,best_model_name,best_parameters_best_model
            
            
            
        
        except Exception as e:
            raise CustomException(e,sys)

