import pandas as pd
from data_preprocessing import DataPreprocessor

# Load raw train dataset
train_df = pd.read_csv('data/train.csv')

# Initialize preprocessor and clean the data
preprocessor = DataPreprocessor()
train_cleaned = preprocessor.fit_transform(train_df, is_train=True)

# Save preprocessed data to CSV
train_cleaned.to_csv('train_preprocessed.csv', index=False)

print("Preprocessed train.csv saved as train_preprocessed.csv")
