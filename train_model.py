"""
Model Training Script for AdAstraa AI - Sales Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor

# Try optional boosting libraries
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print('='*80)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print(f"Train MAE: ${train_mae:.2f}")
    print(f"Test MAE:  ${test_mae:.2f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Test R²:   {test_r2:.4f}")

    return {
        'model': model,
        'name': model_name,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }


def get_advanced_gbr():
    """Improved Gradient Boosting Model"""
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        max_features='sqrt',
        random_state=42
    )


def main():

    print("="*80)
    print("ADASTRAA AI - SALES PREDICTION MODEL TRAINING")
    print("="*80)

    # Load data
    train_df = pd.read_csv('train_preprocessed.csv')
    print(f"\nLoaded {len(train_df)} rows")

    preprocessor = DataPreprocessor()
    train_cleaned = preprocessor.fit_transform(train_df, is_train=True)

    train_cleaned = train_cleaned[train_cleaned['Sale_Amount'].notna()].copy()
    print(f"Rows after filtering NA Sale_Amount: {len(train_cleaned)}")

    # Features
    X = train_cleaned[preprocessor.feature_columns]
    y = train_cleaned['Sale_Amount']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {len(preprocessor.feature_columns)}")

    # -----------------------------------------------------------------------------------
    # MODELS LIST
    # -----------------------------------------------------------------------------------

    models = [
        (Ridge(alpha=1.0), "Ridge Regression (Baseline)"),

        (RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ), "Random Forest"),

        (get_advanced_gbr(), "Improved Gradient Boosting")
    ]

    if XGB_AVAILABLE:
        models.append((
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            ),
            "XGBoost Regressor"
        ))

    if LGB_AVAILABLE:
        models.append((
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=7,
                random_state=42
            ),
            "LightGBM Regressor"
        ))

    # Train & Evaluate
    results = []
    for model, name in models:
        results.append(evaluate_model(model, X_train, X_test, y_train, y_test, name))

    # -----------------------------------------------------------------------------------
    # MODEL COMPARISON TABLE
    # -----------------------------------------------------------------------------------
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    results_df = pd.DataFrame([
        {
            'Model': r['name'],
            'Train MAE': f"${r['train_mae']:.2f}",
            'Test MAE': f"${r['test_mae']:.2f}",
            'RMSE': f"${r['test_rmse']:.2f}",
            'R²': f"{r['test_r2']:.4f}"
        }
        for r in results
    ])

    print(results_df.to_string(index=False))

    # Best model
    best_model_data = min(results, key=lambda r: r['test_mae'])
    best_model = best_model_data['model']

    print(f"\n✓ BEST MODEL SELECTED: {best_model_data['name']}")
    print(f"  Test MAE: ${best_model_data['test_mae']:.2f}")

    # Cross-validation
    print("\n" + "="*80)
    print("CROSS-VALIDATION (5-Fold)")
    print("="*80)
    cv_scores = cross_val_score(best_model, X, y, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    print("CV MAE:", -cv_scores)
    print(f"Mean: ${(-cv_scores).mean():.2f}  ±  ${(-cv_scores).std():.2f}")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\nTOP FEATURES")
        importance_df = pd.DataFrame({
            'Feature': preprocessor.feature_columns,
            'Importance': best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        print(importance_df.head(12).to_string(index=False))

    # Save model
    print("\nSaving model...")
    joblib.dump(best_model, 'ml_models/sales_prediction_model.pkl')
    joblib.dump(preprocessor, 'ml_models/preprocessor.pkl')

    print("\nMODEL SAVED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    main()
