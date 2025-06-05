import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

class LaptopPricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None

    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath, encoding='latin1')
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {filepath}. Please ensure that file exists.")

        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        if 'Weight' in df.columns:
            df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Inches'] = df['Inches'].astype(float)

        df['Memory_Type'] = df['Memory'].apply(self._extract_memory_type)
        df['Memory_Size_GB'] = df['Memory'].apply(self._extract_memory_size)

        df['CPU_Brand'] = df['Cpu'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Unknown')
        df['CPU_Type'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[1:3]) if isinstance(x, str) else 'Unknown')
        df['CPU_Speed_GHz'] = df['Cpu'].apply(self._extract_cpu_speed)

        df['GPU_Brand'] = df['Gpu'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Unknown')

        df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen').fillna(0).astype(int)
        df['IPS'] = df['ScreenResolution'].str.contains('IPS').fillna(0).astype(int)
        resolution = df['ScreenResolution'].apply(self._extract_resolution)
        df['Resolution_X'] = resolution.apply(lambda x: int(x.split('x')[0]) if x else 1366)
        df['Resolution_Y'] = resolution.apply(lambda x: int(x.split('x')[1]) if x else 768)

        cols_to_drop = ['Memory', 'Cpu', 'Gpu', 'ScreenResolution', 'laptop_ID']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

        return df

    def _extract_memory_type(self, memory_str):
        if not isinstance(memory_str, str):
            return 'Other'

        if 'SSD' in memory_str:
            return 'SSD'
        elif 'HDD' in memory_str:
            return 'HDD'
        elif 'Flash Storage' in memory_str:
            return 'Flash'
        elif 'Hybrid' in memory_str:
            return 'Hybrid'
        else:
            return 'Other'

    def _extract_memory_size(self, memory_str):
        if not isinstance(memory_str, str):
            return 0

        sizes = re.findall(r'(\d+\.?\d*)\s*(GB|TB)', memory_str)

        total = 0.0
        for size, unit in sizes:
            try:
                if unit == 'TB':
                    total += float(size) * 1024
                else:
                    total += float(size)
            except ValueError:
                continue

        return round(total)

    def _extract_cpu_speed(self, cpu_str):
        if not isinstance(cpu_str, str):
            return 2.0

        speed_str = re.findall(r'\d\.\d+GHz', cpu_str)
        if speed_str:
            return float(speed_str[0].replace('GHz', ''))
        return 2.0

    def _extract_resolution(self, resolution_str):
        if not isinstance(resolution_str, str):
            return '1366x768'

        res = re.findall(r'(\d+x\d+)', resolution_str)
        return res[0] if res else '1366x768'

    def train_model(self, df):
        if 'Price_euros' not in df.columns:
            raise ValueError("Price_euros column not found in the dataset")

        X = df.drop('Price_euros', axis=1)
        y = df['Price_euros']

        categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
        numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        numerical_transformer = 'passthrough'
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        model = LinearRegression()

        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

        self.model.fit(X_train, y_train)
        self.feature_columns = X.columns.tolist()

        preds = self.model.predict(X_valid)
        r2 = r2_score(y_valid, preds)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))

        print(f"Model trained successfully with R²: {r2:.3f}, RMSE: {rmse:.2f}")
        return r2, rmse

    def save_model(self, filename):
        if not self.model:
            raise ValueError("No model to save. Train the model first.")

        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, filename)
        print(f"Model saved to {filename}")

    def load_saved_model(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")

        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.feature_columns = loaded['feature_columns']
        print(f"Model loaded from {filename}")

    def predict_price(self, input_features):
        if not self.model:
            raise ValueError("Model not trained or loaded")

        input_df = pd.DataFrame([input_features])

        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0 if col in input_df.select_dtypes(include=np.number).columns else ''

        input_df = input_df[self.feature_columns]
        predicted_price = self.model.predict(input_df)[0]
        return round(predicted_price, 2)

# --- Script usage example ---

if __name__ == "__main__":
    predictor = LaptopPricePredictor()
    df = predictor.load_data('laptop_price.csv')
    r2, rmse = predictor.train_model(df)
    predictor.save_model('laptop_price_regression_model.joblib')