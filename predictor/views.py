
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import joblib
import os
from django.conf import settings
import traceback


# Load model and preprocessor at startup
try:
    model = joblib.load(settings.MODEL_PATH)
    preprocessor = joblib.load(settings.PREPROCESSOR_PATH)
    print("✓ Model and preprocessor loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")
    model = None
    preprocessor = None


def home(request):
    """Render the home page with upload form"""
    context = {
        'title': 'AdAstraa AI - Sales Prediction System',
        'description': 'Upload your marketing campaign CSV to predict Sale_Amount'
    }
    return render(request, 'predictor/home.html', context)


def predict(request):
    """Handle CSV upload and generate predictions"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        # Read uploaded CSV file
        uploaded_file = request.FILES['file']

        if not uploaded_file.name.endswith('.csv'):
            return JsonResponse({'error': 'File must be a CSV'}, status=400)

        # Load data
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_columns = [
            'Ad_ID', 'Campaign_Name', 'Clicks', 'Impressions', 'Cost',
            'Leads', 'Conversions', 'Conversion Rate', 'Ad_Date',
            'Location', 'Device', 'Keyword'
        ]

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return JsonResponse({
                'error': f'Missing required columns: {list(missing_columns)}'
            }, status=400)

        # Preprocess data
        df_cleaned = preprocessor.transform(df)

        # Extract features
        X = df_cleaned[preprocessor.feature_columns]

        # Make predictions
        predictions = model.predict(X)

        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['Predicted_Sale_Amount'] = predictions.round(2)

        # Generate CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        result_df.to_csv(response, index=False)

        return response

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return JsonResponse({
            'error': f'Prediction failed: {str(e)}',
            'details': error_trace
        }, status=500)
