import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Handles all data preprocessing tasks for the retail dataset
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.column_transformer = None
        self.feature_columns = []
        
    def load_data(self, filepath):
        """Load dataset from CSV file"""
        return pd.read_csv(filepath)
    
    def get_feature_types(self):
        """Define numerical and categorical features"""
        numerical_features = [
            'Customer_Age', 'Annual_Income', 'Purchase_Frequency', 
            'Discount_Offered', 'Marketing_Spend', 'Seasonal_Demand_Index'
        ]
        
        categorical_features = [
            'Gender', 'Product_Category', 'Store_Location'
        ]
        
        return numerical_features, categorical_features
    
    def preprocess_for_regression(self, data, target_column='Monthly_Sales'):
        """
        Preprocess data for regression task (Linear Regression)
        """
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_column, 'Purchase_Decision', 'Loyalty_Category'])
        y = df[target_column]
        
        # Get feature types
        numerical_features, categorical_features = self.get_feature_types()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        feature_names = (numerical_features + 
                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
        
        self.column_transformer = preprocessor
        self.feature_columns = feature_names
        
        return X_processed, y, feature_names
    
    def preprocess_for_classification(self, data, target_column='Purchase_Decision'):
        """
        Preprocess data for classification task (Decision Tree)
        """
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_column, 'Monthly_Sales', 'Loyalty_Category'])
        y = df[target_column]
        
        # Encode target variable
        if target_column not in self.label_encoders:
            self.label_encoders[target_column] = LabelEncoder()
        y_encoded = self.label_encoders[target_column].fit_transform(y)
        
        # Get feature types
        numerical_features, categorical_features = self.get_feature_types()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        feature_names = (numerical_features + 
                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
        
        return X_processed, y_encoded, feature_names
    
    def preprocess_for_knn(self, data, target_column='Loyalty_Category'):
        """
        Preprocess data for KNN classification
        """
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_column, 'Monthly_Sales', 'Purchase_Decision'])
        y = df[target_column]
        
        # Encode target variable
        if target_column not in self.label_encoders:
            self.label_encoders[target_column] = LabelEncoder()
        y_encoded = self.label_encoders[target_column].fit_transform(y)
        
        # Get feature types
        numerical_features, categorical_features = self.get_feature_types()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        return X_processed, y_encoded
    
    def preprocess_for_clustering(self, data):
        """
        Preprocess data for K-Means clustering
        """
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Select features for clustering (exclude target variables)
        X = df.drop(columns=['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category'])
        
        # Get feature types
        numerical_features, categorical_features = self.get_feature_types()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        return X_processed
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None)
    
    def preprocess_single_prediction(self, input_data):
        """
        Preprocess single instance for prediction
        """
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Use the fitted column transformer
        if self.column_transformer is None:
            raise ValueError("Column transformer not fitted. Call preprocess method first.")
        
        # Transform the data
        X_processed = self.column_transformer.transform(df)
        
        return X_processed
    
    def get_target_encoder(self, target_column):
        """Get label encoder for target variable"""
        return self.label_encoders.get(target_column)
    
    def inverse_transform_target(self, encoded_values, target_column):
        """Inverse transform encoded target values"""
        encoder = self.get_target_encoder(target_column)
        if encoder is None:
            raise ValueError(f"No encoder found for target column: {target_column}")
        return encoder.inverse_transform(encoded_values)

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    data = preprocessor.load_data('../data/synthetic_data.csv')
    print("Data loaded successfully!")
    print(f"Data shape: {data.shape}")
    
    # Test regression preprocessing
    X_reg, y_reg, feature_names = preprocessor.preprocess_for_regression(data)
    print(f"\nRegression preprocessing complete:")
    print(f"Features shape: {X_reg.shape}")
    print(f"Target shape: {y_reg.shape}")
    print(f"Feature names: {len(feature_names)}")
    
    # Test classification preprocessing
    X_clf, y_clf, _ = preprocessor.preprocess_for_classification(data)
    print(f"\nClassification preprocessing complete:")
    print(f"Features shape: {X_clf.shape}")
    print(f"Target shape: {y_clf.shape}")
    print(f"Target classes: {np.unique(y_clf)}")
    
    # Test KNN preprocessing
    X_knn, y_knn = preprocessor.preprocess_for_knn(data)
    print(f"\nKNN preprocessing complete:")
    print(f"Features shape: {X_knn.shape}")
    print(f"Target shape: {y_knn.shape}")
    print(f"Target classes: {np.unique(y_knn)}")
    
    # Test clustering preprocessing
    X_cluster = preprocessor.preprocess_for_clustering(data)
    print(f"\nClustering preprocessing complete:")
    print(f"Features shape: {X_cluster.shape}")
