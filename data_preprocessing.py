"""
Data Preprocessing Utilities for AdAstraa AI Sales Prediction

"""

import pandas as pd
import numpy as np
from dateutil import parser
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for marketing campaign data.

    Handles:
    - Currency format cleaning
    - Multiple date format parsing
    - Text standardization (campaign names, locations, devices, keywords)
    - Missing value imputation
    - Duplicate detection and removal
    - Data validation and correction
    - Feature engineering (CTR, CPC, conversion metrics)
    - Categorical encoding
    - Outlier detection and handling
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.medians = {}
        self.cr_median = None

    def clean_currency(self, value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            cleaned = value.replace('$', '').replace(',', '').strip()
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        return float(value)

    def parse_date(self, date_str):
        if pd.isna(date_str):
            return None
        try:
            date_str = str(date_str).strip().replace('/', '-')
            return parser.parse(date_str, dayfirst=False)
        except:
            return None

    def standardize_text(self, text):
        if pd.isna(text):
            return text
        return ' '.join(str(text).lower().strip().split())

    def fix_campaign_name(self, name):
        if pd.isna(name):
            return name
        name = self.standardize_text(name)
        if 'anlytics' in name or 'analytcis' in name or 'analytic' in name:
            return 'data analytics course'
        return name

    def fix_location(self, location):
        if pd.isna(location):
            return location
        location = self.standardize_text(location)
        if 'hyder' in location or 'hydreb' in location:
            return 'hyderabad'
        return location

    def fix_device(self, device):
        if pd.isna(device):
            return device
        device = self.standardize_text(device)
        if 'desktop' in device:
            return 'desktop'
        elif 'mobile' in device:
            return 'mobile'
        elif 'tablet' in device:
            return 'tablet'
        return device

    def fix_keyword(self, keyword):
        if pd.isna(keyword):
            return keyword
        keyword = self.standardize_text(keyword)
        corrections = {'anaytics': 'analytics', 'analitics': 'analytics', 'analitic': 'analytic'}
        for typo, correct in corrections.items():
            keyword = keyword.replace(typo, correct)
        return keyword

    def calculate_conversion_rate(self, row):
        if pd.notna(row['Clicks']) and pd.notna(row['Conversions']) and row['Clicks'] > 0:
            return round(row['Conversions'] / row['Clicks'], 4)
        return np.nan

    def validate_conversion_rate(self, row, cr_column='Conversion Rate'):
        if pd.isna(row[cr_column]) or pd.isna(row['Clicks']) or row['Clicks'] == 0:
            return True
        expected_cr = row['Conversions'] / row['Clicks'] if pd.notna(row['Conversions']) else 0
        actual_cr = row[cr_column]
        return abs(expected_cr - actual_cr) > 0.01

    def remove_duplicates(self, df, subset=None, keep='first'):
        if subset is None:
            subset = [col for col in df.columns if col != 'Ad_ID']
        duplicates_before = df.duplicated(subset=subset).sum()
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        print(f"Duplicates removed: {duplicates_before}")
        return df_clean

    def fit_transform(self, df, is_train=True, remove_duplicates=True):
        df = df.copy()
        print("Starting preprocessing pipeline...")
        print(f"Initial shape: {df.shape}")

        if remove_duplicates:
            df = self.remove_duplicates(df)
            print(f"Shape after removing duplicates: {df.shape}")

        # Currency
        df['Cost'] = df['Cost'].apply(self.clean_currency)
        if 'Sale_Amount' in df.columns:
            df['Sale_Amount'] = df['Sale_Amount'].apply(self.clean_currency)
            sale_median = df['Sale_Amount'].median() if is_train else self.medians.get('Sale_Amount', 0)
            df['Sale_Amount'] = df['Sale_Amount'].fillna(sale_median)
            if is_train:
                self.medians['Sale_Amount'] = sale_median

        # Dates
        df['Ad_Date_Parsed'] = df['Ad_Date'].apply(self.parse_date)
        unparsed_dates = df['Ad_Date_Parsed'].isna().sum()
        if unparsed_dates > 0:
            print(f"   Warning: {unparsed_dates} dates could not be parsed")
        df['Day'] = df['Ad_Date_Parsed'].dt.day
        df['Month'] = df['Ad_Date_Parsed'].dt.month
        df['Weekday'] = df['Ad_Date_Parsed'].dt.weekday
        df['Week'] = df['Ad_Date_Parsed'].dt.isocalendar().week

        # Standardize text
        df['Campaign_Name'] = df['Campaign_Name'].apply(self.fix_campaign_name)
        df['Location'] = df['Location'].apply(self.fix_location)
        df['Device'] = df['Device'].apply(self.fix_device)
        df['Keyword'] = df['Keyword'].apply(self.fix_keyword)

        # Conversion Rate
        cr_column = 'Conversion Rate' if 'Conversion Rate' in df.columns else 'Conversion_Rate'
        if cr_column in df.columns:
            df['CR_Needs_Fix'] = df.apply(lambda row: self.validate_conversion_rate(row, cr_column), axis=1)
            incorrect_cr = df['CR_Needs_Fix'].sum()
            print(f"   Found {incorrect_cr} incorrect or missing conversion rates")
            df.rename(columns={cr_column: 'Conversion_Rate_Original'}, inplace=True)
            df['Conversion_Rate_Original'] = df['Conversion_Rate_Original'].fillna(df['Conversion_Rate_Original'].median())

        df['Conversion_Rate_Fixed'] = df.apply(self.calculate_conversion_rate, axis=1)
        df['Conversion_Rate_Fixed'] = df['Conversion_Rate_Fixed'].fillna(df['Conversion_Rate_Fixed'].median())

        # Numeric missing values
        numeric_cols = ['Clicks', 'Impressions', 'Cost', 'Leads', 'Conversions']
        for col in numeric_cols:
            median_val = df[col].median() if is_train else self.medians.get(col, df[col].median())
            df[col] = df[col].fillna(median_val)
            if is_train:
                self.medians[col] = median_val

        # Temporal missing values
        for col in ['Day', 'Month', 'Weekday', 'Week']:
            if df[col].isna().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].median()
                df[col] = df[col].fillna(mode_val)

        # Feature engineering
        df['CTR'] = df['Clicks'] / df['Impressions'].replace(0, 1)
        df['CPC'] = df['Cost'] / df['Clicks'].replace(0, 1)
        df['Cost_Per_Lead'] = df['Cost'] / df['Leads'].replace(0, 1)
        df['Lead_Conversion_Rate'] = df['Conversions'] / df['Leads'].replace(0, 1)
        df['Conversion_Per_Impression'] = df['Conversions'] / df['Impressions'].replace(0, 1)
        df['Leads_Per_Click'] = df['Leads'] / df['Clicks'].replace(0, 1)

        # Sale_Amount derived features
        if 'Sale_Amount' in df.columns:
            df['Revenue_Per_Conversion'] = df['Sale_Amount'] / df['Conversions'].replace(0, 1)
            df['ROI'] = (df['Sale_Amount'] - df['Cost']) / df['Cost'].replace(0, 1)
            df['Cost_Per_Sale'] = df['Cost'] / df['Sale_Amount'].replace(0, 1)
        else:
            df['Revenue_Per_Conversion'] = 0
            df['ROI'] = 0
            df['Cost_Per_Sale'] = 0

        # Encode categoricals
        categorical_cols = ['Campaign_Name', 'Location', 'Device', 'Keyword']
        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col + '_Encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Feature columns
        feature_cols = [
            'Clicks', 'Impressions', 'Cost', 'Leads', 'Conversions',
            'Conversion_Rate_Fixed', 'CTR', 'CPC', 'Cost_Per_Lead',
            'Lead_Conversion_Rate', 'Conversion_Per_Impression', 'Leads_Per_Click',
            'Day', 'Month', 'Weekday', 'Week',
            'Campaign_Name_Encoded', 'Location_Encoded', 'Device_Encoded',
            'Keyword_Encoded',
            'Revenue_Per_Conversion', 'ROI', 'Cost_Per_Sale'
        ]

        # Handle infinite and NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feature_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(median_val) else median_val)

        if is_train:
            self.feature_columns = feature_cols

        print(f"\nPreprocessing complete! Final shape: {df.shape}")
        return df

    def transform(self, df):
        return self.fit_transform(df, is_train=False, remove_duplicates=False)

    def get_preprocessing_summary(self, df_original, df_processed):
        summary = {
            'rows_before': len(df_original),
            'rows_after': len(df_processed),
            'rows_removed': len(df_original) - len(df_processed),
            'columns_before': len(df_original.columns),
            'columns_after': len(df_processed.columns),
            'columns_added': len(df_processed.columns) - len(df_original.columns),
            'missing_values_before': df_original.isnull().sum().sum(),
            'missing_values_after': df_processed.isnull().sum().sum()
        }
        return summary
