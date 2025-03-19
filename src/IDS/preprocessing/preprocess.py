import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import os
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self, output_dir: str = "../output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None  
        self.labels: Optional[pd.Series] = None
        self.scaler = StandardScaler()
        logging.info("Preprocessor initialized.")

    def load_data(self, df: pd.DataFrame, label_col: str) -> None:
        logging.info("Loading data...")
        try:
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in the dataset.")
            self.df = df.drop(columns=[label_col]).copy()
            self.labels = df[label_col]
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            logging.info(f"Data loaded successfully with shape: {self.df.shape}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Handling missing values...")
        try:
            for col in self.df.columns:
                fill_value = self.df[col].mode()[0] if self.df[col].dtype == 'object' else self.df[col].median()
                self.df[col].fillna(fill_value, inplace=True)
            logging.info("Missing values handled successfully.")
        except Exception as e:
            logging.error(f"Error handling missing values: {e}")
            raise

    def normalize_data(self, left_skewed: List[str], right_skewed: List[str]) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Normalizing data...")
        try:
            for col in right_skewed:
                if col in self.df.columns:
                    self.df[col] = np.log1p(self.df[col])
            for col in left_skewed:
                if col in self.df.columns:
                    self.df[col] = self.df[col] ** 2
            logging.info("Data normalization completed.")
        except Exception as e:
            logging.error(f"Error normalizing data: {e}")
            raise

    def drop_unnecessary_columns(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Dropping unnecessary columns...")
        try:
            cols_to_drop = self.df.nunique()[self.df.nunique() == 1].index.tolist()
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.numeric_columns = self.df.select_dtypes(include=['number']).columns
            logging.info("Unnecessary columns dropped successfully.")
        except Exception as e:
            logging.error(f"Error dropping unnecessary columns: {e}")
            raise

    def encode_categorical_features(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Encoding categorical features...")
        try:
            self.df = pd.get_dummies(self.df, drop_first=True, dtype=int)
            logging.info("Categorical features encoded successfully.")
        except Exception as e:
            logging.error(f"Error encoding categorical features: {e}")
            raise

    def standardize_features(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Standardizing features...")
        try:
            numeric_columns = self.df.select_dtypes(include=['number']).columns
            self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])
            logging.info(f"Feature standardization completed. ")
        except Exception as e:
            logging.error(f"Error standardizing features: {e}")
            raise

    def remove_outliers(self, threshold: float = 3.0) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Removing outliers...")
        try:
            mask = (np.abs(self.df[self.numeric_columns]) < threshold).all(axis=1)
            self.df = self.df[mask]
            self.labels = self.labels[mask]
            logging.info(f"Outliers removed successfully.")
        except Exception as e:
            logging.error(f"Error removing outliers: {e}")
            raise

    def apply_svd(self, n_components: int = 37) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Applying SVD...")
        try:
            self.svd = TruncatedSVD(n_components=n_components)
            self.df = pd.DataFrame(self.svd.fit_transform(self.df))
            logging.info(f"SVD applied successfully.")
        except Exception as e:
            logging.error(f"Error applying SVD: {e}")
            raise

    def save_data(self, filename: str = "processed_data.csv") -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Saving processed data...")
        try:
            self.df.to_csv(os.path.join(self.output_dir, filename), index=False)
            logging.info(f"Data saved successfully at {os.path.join(self.output_dir, filename)}.")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise

    def process(self, left_skewed: List[str], right_skewed: List[str], n_components: int = 37) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Starting data preprocessing pipeline...")
        try:
            self.handle_missing_values()
            self.normalize_data(left_skewed, right_skewed)
            self.drop_unnecessary_columns()
            self.encode_categorical_features()
            self.standardize_features()
            self.remove_outliers()
            self.apply_svd(n_components)
            self.save_data()
            logging.info("Data preprocessing pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error during preprocessing pipeline: {e}")
            raise

    def svd_trans(self, n_components: int = 37) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Applying SVD...")
        try:
            self.df = pd.DataFrame(self.svd.transform(self.df))
            logging.info(f"SVD applied successfully.")
        except Exception as e:
            logging.error(f"Error applying SVD: {e}")
            raise

    def transform(self, left_skewed: List[str], right_skewed: List[str], n_components: int = 37) -> None:
        if self.df is None:
            raise ValueError("Data not loaded.")
        logging.info("Starting data preprocessing pipeline...")
        try:
            self.handle_missing_values()
            self.normalize_data(left_skewed, right_skewed)
            self.drop_unnecessary_columns()
            self.encode_categorical_features()
            self.standardize_features()
            self.svd_trans(n_components)
            logging.info("Data preprocessing pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error during preprocessing pipeline: {e}")
            raise

# Example Usage
# if __name__ == "__main__":
#     try:
#         preprocessor = Preprocessor()
#         df = pd.read_csv("C:/Users/Vishruth V Srivatsa/OneDrive/Desktop/IDS/data/raw/KDDTrain+.csv")
#         preprocessor.load_data(df, 'normal')
#         right_skewed = ['0', '491', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.18', '2', '2.1', '0.00', '0.00.1', '0.00.2']
#         left_skewed = ['20', '150', '1.00']
#         preprocessor.process(left_skewed, right_skewed)
#     except Exception as e:
#         logging.error(f"Processing failed: {e}")
