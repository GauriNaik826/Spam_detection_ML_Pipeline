import logging
import os 
import pandas as pd
from sklearn.model_selection import train_test_split

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test data into the specified directory.
    """
    try:
        raw_dir = os.path.join(data_path, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        train_path = os.path.join(raw_dir, "train.csv")
        test_path = os.path.join(raw_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.debug("Train and test data saved successfully at %s", raw_dir)
    except Exception as e:
        logger.error("Failed to save data at %s: %s", data_path, e)
        raise

def main():
    try:
        # Define test data size
        test_size = 0.2
        
        # Define the data path (URL to the dataset hosted on GitHub)
        data_path = 'https://raw.githubusercontent.com/GauriNaik826/Spam_detection_ML_Pipeline/refs/heads/main/spam.csv'
        
        # Load the data from the URL
        df = load_data(data_url=data_path)
        
        # Preprocess the loaded data
        final_df = preprocess_data(df)
        
        # Split the data into train and test sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save the train and test data to the ./data directory
        save_data(train_data, test_data, data_path='./data')
    
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

# Run the main function if this script is executed directly
if __name__ == '__main__':
    main()

