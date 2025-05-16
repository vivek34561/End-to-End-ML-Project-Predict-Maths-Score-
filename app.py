from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig , DataTransformation

if __name__ =="__main__":
    logging.info("the execution has started")
    
    try:
        data_ingestion = DataIngestion()
        # data_ingestion_config = DataIngestionConfig()
        train_data_paths , test_data_paths = data_ingestion.initiate_data_ingestion()
        # data_ingestion_config = DataIngestionConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_paths , test_data_paths)
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e , sys)     