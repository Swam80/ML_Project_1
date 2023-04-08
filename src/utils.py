import os
import sys

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

import pickle

from sklearn.metrics import r2_score



def save_object(file_path,obj):  # ( where to save, what to save)

    try:
        dir_path = os.path.dirname(file_path)  # Gets directory path (till artifacts in this case)

        os.makedirs(dir_path,exist_ok= True)

        with open(file_path,'wb') as file_obj: # Opens the pickle file and dumps the  obj
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train,y_train,X_test,y_test,models_params):
    
    try:
        
        report = []

        for model_name,m_p in models_params.items():

            model = m_p['model_obj']                    # Since models_params has nested dicts, m_p contains model objects and parameters.
            params = m_p['params']

            gs = GridSearchCV(model,params,cv=4)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)         # setting the best parameters for the respective models.
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            report.append(
                {
                'Model_name' : model_name,
                'Tuned_Score' : test_model_score,
                'Best_Params' : gs.best_params_

                }
            )

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
