import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

from src.mlproject.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "KNeighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Linear Regression": LinearRegression()
            }
            
            
            
            params = {
                "Decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best', 'random'],
                    # 'max_features':[None, 'sqrt', 'log2']
                },
                "Random Forest":{
                    'n_estimators':[10, 50, 100],
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':[None, 'sqrt', 'log2']
                },
                "Gradient Boosting":{
                    'n_estimators':[10, 50, 100],
                    'learning_rate':[0.01, 0.1, 0.2],
                    # 'max_depth':[3, 5, 7],
                    # 'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'subsample':[0.5, 0.75, 1.0],
                    # 'criterion':['friedman_mse', 'squared_error'],
                    # 'max_features':[None, 'sqrt', 'log2']
                },
                "Linear Regression":{},
                "XGBoost":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'n_estimators':[10, 50, 100]
                    },
                "CatBoost":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'iterations':[10, 50, 100],
                    'depth':[4, 6, 8]
                    },
                "AdaBoost":{
                    'n_estimators':[10, 50, 100],
                    'learning_rate':[0.01, 0.1, 0.2],
                    'loss':['linear', 'square', 'exponential']
                    },
                "KNeighbors":{
                    'n_neighbors':[3, 5, 7],
                    'weights':['uniform', 'distance'],
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # 'leaf_size':[10, 20, 30]
                }
            }
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            best_model_score = max(sorted(model_report.values()))
            # best_model_name = max(sorted(model_report), key=model_report.get)
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)        