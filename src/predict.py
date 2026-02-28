import pandas as pd
import numpy as np
import joblib
from preprocessing import DataPreprocessor

class PredictionSystem:
    """
    Handle predictions for all trained models
    """
    
    def __init__(self, model_dir='../models'):
        self.model_dir = model_dir
        self.models = {}
        self.preprocessor = None
        self.load_models()
    
    def load_models(self):
        """
        Load all trained models and preprocessor
        """
        try:
            # Load models
            self.models['linear_regression'] = joblib.load(f'{self.model_dir}/linear_regression.pkl')
            self.models['decision_tree'] = joblib.load(f'{self.model_dir}/decision_tree.pkl')
            self.models['knn'] = joblib.load(f'{self.model_dir}/knn.pkl')
            self.models['kmeans'] = joblib.load(f'{self.model_dir}/kmeans.pkl')
            
            # Load preprocessor
            self.preprocessor = joblib.load(f'{self.model_dir}/preprocessor.pkl')
            
            print("All models loaded successfully!")
            
        except FileNotFoundError:
            print("Models not found. Please train models first.")
            raise
    
    def predict_monthly_sales(self, input_data):
        """
        Predict Monthly Sales using Linear Regression
        """
        # Preprocess input
        X_processed = self.preprocessor.preprocess_single_prediction(input_data)
        
        # Make prediction
        prediction = self.models['linear_regression'].predict(X_processed)[0]
        
        return round(prediction, 2)
    
    def predict_purchase_decision(self, input_data):
        """
        Predict Purchase Decision using Decision Tree
        """
        # Preprocess input
        X_processed = self.preprocessor.preprocess_single_prediction(input_data)
        
        # Make prediction
        prediction_encoded = self.models['decision_tree'].predict(X_processed)[0]
        
        # Get the label encoder
        encoder = self.preprocessor.get_target_encoder('Purchase_Decision')
        prediction = encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probability
        probabilities = self.models['decision_tree'].predict_proba(X_processed)[0]
        confidence = max(probabilities) * 100
        
        return prediction, round(confidence, 2)
    
    def predict_loyalty_category(self, input_data):
        """
        Predict Loyalty Category using KNN
        """
        # Preprocess input
        X_processed = self.preprocessor.preprocess_single_prediction(input_data)
        
        # Make prediction
        prediction_encoded = self.models['knn'].predict(X_processed)[0]
        
        # Get the label encoder
        encoder = self.preprocessor.get_target_encoder('Loyalty_Category')
        prediction = encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probability
        probabilities = self.models['knn'].predict_proba(X_processed)[0]
        confidence = max(probabilities) * 100
        
        return prediction, round(confidence, 2)
    
    def predict_customer_cluster(self, input_data):
        """
        Predict Customer Cluster using K-Means
        """
        # Preprocess input
        X_processed = self.preprocessor.preprocess_single_prediction(input_data)
        
        # Make prediction
        cluster = self.models['kmeans'].predict(X_processed)[0]
        
        # Get cluster center distance for confidence
        distances = self.models['kmeans'].transform(X_processed)[0]
        confidence = (1 - distances[cluster] / np.sum(distances)) * 100
        
        return int(cluster), round(confidence, 2)
    
    def make_comprehensive_prediction(self, input_data):
        """
        Make predictions using all models
        """
        results = {}
        
        try:
            # Monthly Sales Prediction
            results['monthly_sales'] = self.predict_monthly_sales(input_data)
            
            # Purchase Decision Prediction
            purchase_decision, purchase_confidence = self.predict_purchase_decision(input_data)
            results['purchase_decision'] = purchase_decision
            results['purchase_confidence'] = purchase_confidence
            
            # Loyalty Category Prediction
            loyalty_category, loyalty_confidence = self.predict_loyalty_category(input_data)
            results['loyalty_category'] = loyalty_category
            results['loyalty_confidence'] = loyalty_confidence
            
            # Customer Cluster Prediction
            cluster, cluster_confidence = self.predict_customer_cluster(input_data)
            results['customer_cluster'] = cluster
            results['cluster_confidence'] = cluster_confidence
            
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'error'
            results['error_message'] = str(e)
        
        return results
    
    def get_cluster_characteristics(self, cluster_id):
        """
        Get characteristics of a specific cluster
        """
        # Load original data to analyze cluster characteristics
        try:
            data = pd.read_csv('../data/synthetic_data.csv')
            
            # Get cluster labels for all data
            X_all = self.preprocessor.preprocess_for_clustering(data)
            all_clusters = self.models['kmeans'].predict(X_all)
            
            # Filter data for the specific cluster
            cluster_data = data[all_clusters == cluster_id]
            
            if len(cluster_data) == 0:
                return "No data found for this cluster"
            
            # Calculate cluster characteristics
            characteristics = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(data)) * 100,
                'avg_income': cluster_data['Annual_Income'].mean(),
                'avg_sales': cluster_data['Monthly_Sales'].mean(),
                'avg_age': cluster_data['Customer_Age'].mean(),
                'common_categories': cluster_data['Product_Category'].mode().tolist(),
                'common_locations': cluster_data['Store_Location'].mode().tolist(),
                'loyalty_distribution': cluster_data['Loyalty_Category'].value_counts().to_dict()
            }
            
            return characteristics
            
        except Exception as e:
            return f"Error analyzing cluster: {str(e)}"
    
    def get_feature_importance(self):
        """
        Get feature importance from Decision Tree model
        """
        try:
            # Load metrics to get feature importance
            metrics = joblib.load(f'{self.model_dir}/metrics.pkl')
            
            if 'decision_tree' in metrics:
                feature_names = metrics['decision_tree']['feature_names']
                importances = metrics['decision_tree']['feature_importances']
                
                # Create DataFrame for better display
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                return "Feature importance not available"
                
        except Exception as e:
            return f"Error loading feature importance: {str(e)}"
    
    def get_model_metrics(self):
        """
        Get performance metrics for all models
        """
        try:
            metrics = joblib.load(f'{self.model_dir}/metrics.pkl')
            return metrics
        except Exception as e:
            return f"Error loading metrics: {str(e)}"

if __name__ == "__main__":
    # Test the prediction system
    try:
        predictor = PredictionSystem()
        
        # Test with sample input
        sample_input = {
            'Customer_Age': 35,
            'Gender': 'Male',
            'Annual_Income': 75000,
            'Purchase_Frequency': 8,
            'Discount_Offered': 15.0,
            'Product_Category': 'Electronics',
            'Marketing_Spend': 5000,
            'Seasonal_Demand_Index': 1.2,
            'Store_Location': 'Downtown'
        }
        
        print("Testing prediction system...")
        results = predictor.make_comprehensive_prediction(sample_input)
        
        print("Prediction Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # Test cluster characteristics
        print("\nTesting cluster characteristics...")
        cluster_info = predictor.get_cluster_characteristics(0)
        print(f"Cluster 0 characteristics: {cluster_info}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure models are trained first by running train_models.py")
