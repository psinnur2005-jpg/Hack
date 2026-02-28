import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import DataPreprocessor

class ModelTrainer:
    """
    Train and evaluate all ML models for the retail dataset
    """
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.metrics = {}
        
    def train_linear_regression(self, data):
        """
        Train Linear Regression model for Monthly_Sales prediction
        """
        print("Training Linear Regression model...")
        
        # Preprocess data
        X, y, feature_names = self.preprocessor.preprocess_for_regression(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store model and metrics
        self.models['linear_regression'] = model
        self.metrics['linear_regression'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'feature_names': feature_names,
            'coefficients': model.coef_
        }
        
        print(f"Linear Regression trained successfully!")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return model, self.metrics['linear_regression']
    
    def train_decision_tree(self, data):
        """
        Train Decision Tree Classifier for Purchase_Decision prediction
        """
        print("Training Decision Tree Classifier...")
        
        # Preprocess data
        X, y, feature_names = self.preprocessor.preprocess_for_classification(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        # Train model
        model = DecisionTreeClassifier(random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store model and metrics
        self.models['decision_tree'] = model
        self.metrics['decision_tree'] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'feature_names': feature_names,
            'feature_importances': model.feature_importances_
        }
        
        print(f"Decision Tree trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        
        return model, self.metrics['decision_tree']
    
    def train_knn(self, data):
        """
        Train KNN Classifier for Loyalty_Category prediction
        """
        print("Training KNN Classifier...")
        
        # Preprocess data
        X, y = self.preprocessor.preprocess_for_knn(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        # Train model
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store model and metrics
        self.models['knn'] = model
        self.metrics['knn'] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        print(f"KNN trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        
        return model, self.metrics['knn']
    
    def train_kmeans(self, data, n_clusters=4):
        """
        Train K-Means Clustering for customer segmentation
        """
        print("Training K-Means Clustering...")
        
        # Preprocess data
        X = self.preprocessor.preprocess_for_clustering(data)
        
        # Train model
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(X)
        
        # Calculate metrics
        inertia = model.inertia_
        
        # Store model and metrics
        self.models['kmeans'] = model
        self.metrics['kmeans'] = {
            'inertia': inertia,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels
        }
        
        print(f"K-Means trained successfully!")
        print(f"Inertia: {inertia:.2f}")
        print(f"Cluster distribution: {np.bincount(cluster_labels)}")
        
        return model, self.metrics['kmeans']
    
    def train_all_models(self, data):
        """
        Train all models at once
        """
        print("=" * 50)
        print("TRAINING ALL MODELS")
        print("=" * 50)
        
        # Train all models
        self.train_linear_regression(data)
        print()
        
        self.train_decision_tree(data)
        print()
        
        self.train_knn(data)
        print()
        
        self.train_kmeans(data)
        print()
        
        print("All models trained successfully!")
        return self.models, self.metrics
    
    def save_models(self, model_dir='../models'):
        """
        Save all trained models to disk
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.models['linear_regression'], f'{model_dir}/linear_regression.pkl')
        joblib.dump(self.models['decision_tree'], f'{model_dir}/decision_tree.pkl')
        joblib.dump(self.models['knn'], f'{model_dir}/knn.pkl')
        joblib.dump(self.models['kmeans'], f'{model_dir}/kmeans.pkl')
        
        # Save preprocessor
        joblib.dump(self.preprocessor, f'{model_dir}/preprocessor.pkl')
        
        # Save metrics
        joblib.dump(self.metrics, f'{model_dir}/metrics.pkl')
        
        print(f"All models saved to {model_dir}")
    
    def load_models(self, model_dir='../models'):
        """
        Load all trained models from disk
        """
        # Load models
        self.models['linear_regression'] = joblib.load(f'{model_dir}/linear_regression.pkl')
        self.models['decision_tree'] = joblib.load(f'{model_dir}/decision_tree.pkl')
        self.models['knn'] = joblib.load(f'{model_dir}/knn.pkl')
        self.models['kmeans'] = joblib.load(f'{model_dir}/kmeans.pkl')
        
        # Load preprocessor
        self.preprocessor = joblib.load(f'{model_dir}/preprocessor.pkl')
        
        # Load metrics
        self.metrics = joblib.load(f'{model_dir}/metrics.pkl')
        
        print(f"All models loaded from {model_dir}")
        return self.models, self.metrics
    
    def get_model_summary(self):
        """
        Get summary of all trained models
        """
        summary = []
        
        for model_name, metrics in self.metrics.items():
            if model_name == 'linear_regression':
                summary.append({
                    'Model': 'Linear Regression',
                    'Task': 'Regression (Monthly_Sales)',
                    'MAE': f"{metrics['MAE']:.2f}",
                    'RMSE': f"{metrics['RMSE']:.2f}",
                    'R²': f"{metrics['R2']:.4f}"
                })
            elif model_name == 'decision_tree':
                summary.append({
                    'Model': 'Decision Tree',
                    'Task': 'Classification (Purchase_Decision)',
                    'Accuracy': f"{metrics['accuracy']:.4f}"
                })
            elif model_name == 'knn':
                summary.append({
                    'Model': 'KNN',
                    'Task': 'Classification (Loyalty_Category)',
                    'Accuracy': f"{metrics['accuracy']:.4f}"
                })
            elif model_name == 'kmeans':
                summary.append({
                    'Model': 'K-Means',
                    'Task': 'Clustering (Customer Segments)',
                    'Clusters': metrics['n_clusters'],
                    'Inertia': f"{metrics['inertia']:.2f}"
                })
        
        return pd.DataFrame(summary)

if __name__ == "__main__":
    # Test model training
    trainer = ModelTrainer()
    
    # Load data
    data = pd.read_csv('../data/synthetic_data.csv')
    print(f"Data loaded: {data.shape}")
    
    # Train all models
    models, metrics = trainer.train_all_models(data)
    
    # Save models
    trainer.save_models()
    
    # Display summary
    summary = trainer.get_model_summary()
    print("\nMODEL SUMMARY:")
    print(summary.to_string(index=False))
