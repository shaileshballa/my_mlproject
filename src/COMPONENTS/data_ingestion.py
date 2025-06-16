import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method started')
        try:
            df = pd.read_csv('notebooks/data/Stud.csv')
            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Train and Test data saved')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, raw_data = obj.initiate_data_ingestion()
    print(f"Train Data Path: {train_data}")
    print(f"Test Data Path: {test_data}")
    print(f"Raw Data Path: {raw_data}")
