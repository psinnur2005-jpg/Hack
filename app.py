import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import generate_synthetic_data, save_dataset
from src.train_models import ModelTrainer
from src.predict import PredictionSystem
from src.visualization import DataVisualizer

# Set page configuration
st.set_page_config(
    page_title="Smart Retail Sales Forecasting & Customer Segmentation",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load and cache dataset"""
    try:
        return pd.read_csv('data/synthetic_data.csv')
    except FileNotFoundError:
        return None

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_prediction_system():
    """Load and cache prediction system"""
    try:
        return PredictionSystem()
    except:
        return None

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_visualizer():
    """Load and cache visualizer"""
    return DataVisualizer()

def main():
    # Sidebar navigation
    st.sidebar.title("🏪 Navigation")
    page = st.sidebar.selectbox("Select a page", [
        "Home",
        "Data Generator/Upload", 
        "Model Training",
        "Prediction Interface",
        "Visualization",
        "Model Performance"
    ])
    
    if page == "Home":
        home_page()
    elif page == "Data Generator/Upload":
        data_page()
    elif page == "Model Training":
        training_page()
    elif page == "Prediction Interface":
        prediction_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Model Performance":
        performance_page()

def home_page():
    """Home page with project information"""
    st.markdown('<h1 class="main-header">🏪 Smart Retail Sales Forecasting & Customer Segmentation System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 Project Overview
    
    This comprehensive machine learning system provides advanced analytics for retail businesses including:
    
    - **Sales Forecasting**: Predict monthly sales using Linear Regression
    - **Purchase Decision Analysis**: Classify customer purchase decisions using Decision Tree
    - **Customer Loyalty Prediction**: Predict loyalty categories using KNN
    - **Customer Segmentation**: Group customers into meaningful segments using K-Means Clustering
    
    ## 🎯 Key Features
    
    - **Multi-Model Approach**: Four different ML algorithms for different tasks
    - **Interactive Dashboard**: Streamlit-based web interface for easy interaction
    - **Real-time Predictions**: Get instant predictions for new customer data
    - **Comprehensive Visualizations**: Detailed charts and graphs for insights
    - **Performance Metrics**: Track model accuracy and performance
    
    ## 🛠️ Technology Stack
    
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning algorithms
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    - **Pandas & NumPy**: Data manipulation and analysis
    - **Joblib**: Model serialization
    
    ## 👥 Team Information
    
    **ML & Full Stack AI Engineering Team**
    - Senior ML Engineer
    - Full Stack Developer
    - Data Scientist
    - UI/UX Designer
    
    ---
    *Built with ❤️ for retail analytics and business intelligence*
    """)

def data_page():
    """Data generation and upload page"""
    st.markdown('<h1 class="main-header">📁 Data Generator & Upload</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="sub-header">Generate Synthetic Data</h2>', unsafe_allow_html=True)
        
        n_samples = st.slider("Number of records", min_value=500, max_value=5000, value=1500, step=100)
        
        if st.button("🔄 Generate Dataset", type="primary"):
            with st.spinner("Generating synthetic dataset..."):
                try:
                    data = generate_synthetic_data(n_samples)
                    save_dataset(data, 'data/synthetic_data.csv')
                    st.markdown('<div class="success-message">✅ Dataset generated successfully!</div>', unsafe_allow_html=True)
                    st.success(f"Generated {len(data)} records")
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">Upload Custom Data</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Shape: {data.shape}")
                
                # Save uploaded data
                data.to_csv('data/synthetic_data.csv', index=False)
                st.markdown('<div class="success-message">✅ Data saved successfully!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error reading file: {str(e)}</div>', unsafe_allow_html=True)
    
    # Data preview section
    st.markdown('<h2 class="sub-header">Dataset Preview</h2>', unsafe_allow_html=True)
    
    try:
        data = pd.read_csv('data/synthetic_data.csv')
        
        # Basic statistics
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Features", data.shape[1])
        with col3:
            st.metric("Numerical Features", data.select_dtypes(include=[np.number]).shape[1])
        with col4:
            st.metric("Categorical Features", data.select_dtypes(include=['object']).shape[1])
        
        # Data preview
        st.subheader("👁️ Data Preview")
        show_rows = st.slider("Number of rows to show", min_value=5, max_value=50, value=10)
        st.dataframe(data.head(show_rows))
        
        # Column information
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Unique Values': data.nunique()
        })
        st.dataframe(col_info)
        
    except FileNotFoundError:
        st.info("No dataset found. Please generate or upload data first.")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

def training_page():
    """Model training page"""
    st.markdown('<h1 class="main-header">🤖 Model Training</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Train all four machine learning models on your dataset:
    
    1. **Linear Regression** - Predict Monthly Sales
    2. **Decision Tree** - Predict Purchase Decision  
    3. **KNN Classifier** - Predict Loyalty Category
    4. **K-Means Clustering** - Customer Segmentation
    """)
    
    # Check if data exists
    try:
        data = pd.read_csv('data/synthetic_data.csv')
        st.success(f"Dataset found with {len(data)} records")
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes..."):
                try:
                    trainer = ModelTrainer()
                    models, metrics = trainer.train_all_models(data)
                    trainer.save_models()
                    
                    st.markdown('<div class="success-message">✅ All models trained successfully!</div>', unsafe_allow_html=True)
                    
                    # Display training results
                    st.subheader("📊 Training Results")
                    
                    # Linear Regression Results
                    if 'linear_regression' in metrics:
                        with st.expander("📈 Linear Regression Results"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MAE", f"{metrics['linear_regression']['MAE']:.2f}")
                            with col2:
                                st.metric("RMSE", f"{metrics['linear_regression']['RMSE']:.2f}")
                            with col3:
                                st.metric("R²", f"{metrics['linear_regression']['R2']:.4f}")
                    
                    # Decision Tree Results
                    if 'decision_tree' in metrics:
                        with st.expander("🌳 Decision Tree Results"):
                            st.metric("Accuracy", f"{metrics['decision_tree']['accuracy']:.4f}")
                            
                            # Feature importance
                            feature_names = metrics['decision_tree']['feature_names']
                            importances = metrics['decision_tree']['feature_importances']
                            
                            feature_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            st.subheader("Top 10 Important Features")
                            st.dataframe(feature_df)
                    
                    # KNN Results
                    if 'knn' in metrics:
                        with st.expander("🎯 KNN Results"):
                            st.metric("Accuracy", f"{metrics['knn']['accuracy']:.4f}")
                    
                    # K-Means Results
                    if 'kmeans' in metrics:
                        with st.expander("🎨 K-Means Clustering Results"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Number of Clusters", metrics['kmeans']['n_clusters'])
                            with col2:
                                st.metric("Inertia", f"{metrics['kmeans']['inertia']:.2f}")
                            
                            # Cluster distribution
                            cluster_labels = metrics['kmeans']['cluster_labels']
                            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                            
                            st.subheader("Cluster Distribution")
                            fig = px.pie(values=cluster_counts.values, 
                                        names=[f"Cluster {i}" for i in cluster_counts.index],
                                        title="Customer Segment Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Training failed: {str(e)}</div>', unsafe_allow_html=True)
        
        # Show model summary if models exist
        if os.path.exists('models/linear_regression.pkl'):
            st.subheader("📋 Model Summary")
            try:
                trainer = ModelTrainer()
                trainer.load_models()
                summary = trainer.get_model_summary()
                st.dataframe(summary)
            except Exception as e:
                st.error(f"Error loading model summary: {str(e)}")
                
    except FileNotFoundError:
        st.error("No dataset found. Please generate or upload data first in the Data page.")

def prediction_page():
    """Prediction interface page"""
    st.markdown('<h1 class="main-header">🔮 Prediction Interface</h1>', unsafe_allow_html=True)
    
    # Check if models exist
    if not os.path.exists('models/linear_regression.pkl'):
        st.error("Models not found. Please train models first in the Model Training page.")
        return
    
    try:
        predictor = PredictionSystem()
        
        # Input form
        st.subheader("📝 Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_age = st.slider("Customer Age", min_value=18, max_value=80, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            annual_income = st.number_input("Annual Income ($)", min_value=20000, max_value=200000, value=75000, step=1000)
            purchase_frequency = st.slider("Purchase Frequency (per month)", min_value=1, max_value=30, value=8)
            discount_offered = st.slider("Discount Offered (%)", min_value=0.0, max_value=30.0, value=15.0, step=0.5)
        
        with col2:
            product_category = st.selectbox("Product Category", 
                                          ["Electronics", "Clothing", "Groceries", "Home & Garden", 
                                           "Sports", "Books", "Toys", "Beauty"])
            marketing_spend = st.number_input("Marketing Spend ($)", min_value=500, max_value=50000, value=5000, step=100)
            seasonal_demand_index = st.slider("Seasonal Demand Index", min_value=0.3, max_value=2.0, value=1.0, step=0.1)
            store_location = st.selectbox("Store Location", ["Downtown", "Suburban", "Mall", "Airport", "Online"])
        
        # Prediction button
        if st.button("🎯 Make Predictions", type="primary"):
            # Prepare input data
            input_data = {
                'Customer_Age': customer_age,
                'Gender': gender,
                'Annual_Income': annual_income,
                'Purchase_Frequency': purchase_frequency,
                'Discount_Offered': discount_offered,
                'Product_Category': product_category,
                'Marketing_Spend': marketing_spend,
                'Seasonal_Demand_Index': seasonal_demand_index,
                'Store_Location': store_location
            }
            
            with st.spinner("Making predictions..."):
                try:
                    results = predictor.make_comprehensive_prediction(input_data)
                    
                    if results['status'] == 'success':
                        st.markdown('<div class="success-message">✅ Predictions completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Display results
                        st.subheader("📊 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown("### 💰 Monthly Sales Prediction")
                            st.markdown(f"#### ${results['monthly_sales']:,.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown("### 🛒 Purchase Decision")
                            st.markdown(f"#### {results['purchase_decision']}")
                            st.markdown(f"**Confidence:** {results['purchase_confidence']:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown("### ⭐ Loyalty Category")
                            st.markdown(f"#### {results['loyalty_category']}")
                            st.markdown(f"**Confidence:** {results['loyalty_confidence']:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown("### 🎨 Customer Segment")
                            st.markdown(f"#### Cluster {results['customer_cluster']}")
                            st.markdown(f"**Confidence:** {results['cluster_confidence']:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Cluster characteristics
                        st.subheader("🔍 Customer Segment Analysis")
                        cluster_info = predictor.get_cluster_characteristics(results['customer_cluster'])
                        
                        if isinstance(cluster_info, dict):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Segment Size", cluster_info['size'])
                                st.metric("% of Customers", f"{cluster_info['percentage']:.1f}%")
                            
                            with col2:
                                st.metric("Avg Income", f"${cluster_info['avg_income']:,.0f}")
                                st.metric("Avg Sales", f"${cluster_info['avg_sales']:,.0f}")
                            
                            with col3:
                                st.metric("Avg Age", f"{cluster_info['avg_age']:.1f} years")
                                st.markdown("**Common Categories:**")
                                for category in cluster_info['common_categories'][:3]:
                                    st.markdown(f"• {category}")
                        
                        else:
                            st.info(cluster_info)
                    
                    else:
                        st.markdown(f'<div class="error-message">❌ Prediction failed: {results.get("error_message", "Unknown error")}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading prediction system: {str(e)}")

def visualization_page():
    """Visualization page"""
    st.markdown('<h1 class="main-header">📊 Data Visualization</h1>', unsafe_allow_html=True)
    
    try:
        data = pd.read_csv('data/synthetic_data.csv')
        visualizer = DataVisualizer()
        
        # Visualization selection
        viz_type = st.selectbox("Select Visualization", [
            "Sales Distribution",
            "Income vs Sales",
            "Sales by Category",
            "Customer Age Distribution",
            "Purchase Frequency Analysis",
            "Seasonal Demand Impact",
            "Correlation Heatmap"
        ])
        
        if viz_type == "Sales Distribution":
            fig = visualizer.plot_sales_distribution(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Income vs Sales":
            fig = visualizer.plot_income_vs_sales(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Sales by Category":
            fig = visualizer.plot_sales_by_category(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Customer Age Distribution":
            fig = visualizer.plot_customer_age_distribution(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Purchase Frequency Analysis":
            fig = visualizer.plot_purchase_frequency_analysis(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Seasonal Demand Impact":
            fig = visualizer.plot_seasonal_demand_impact(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Heatmap":
            fig = visualizer.plot_correlation_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster visualization if models exist
        if os.path.exists('models/kmeans.pkl'):
            st.subheader("🎨 Customer Segments")
            try:
                predictor = PredictionSystem()
                metrics = predictor.get_model_metrics()
                
                if 'kmeans' in metrics:
                    cluster_labels = metrics['kmeans']['cluster_labels']
                    fig = visualizer.plot_cluster_visualization(data, cluster_labels)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating cluster visualization: {str(e)}")
        
        # Feature importance if models exist
        if os.path.exists('models/decision_tree.pkl'):
            st.subheader("🎯 Feature Importance")
            try:
                predictor = PredictionSystem()
                feature_importance = predictor.get_feature_importance()
                
                if isinstance(feature_importance, pd.DataFrame):
                    fig = visualizer.plot_feature_importance(
                        feature_importance['feature'], 
                        feature_importance['importance'],
                        "Decision Tree Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating feature importance plot: {str(e)}")
    
    except FileNotFoundError:
        st.error("No dataset found. Please generate or upload data first.")
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")

def performance_page():
    """Model performance page"""
    st.markdown('<h1 class="main-header">📈 Model Performance</h1>', unsafe_allow_html=True)
    
    if not os.path.exists('models/metrics.pkl'):
        st.error("No model metrics found. Please train models first.")
        return
    
    try:
        predictor = PredictionSystem()
        metrics = predictor.get_model_metrics()
        
        # Performance dashboard
        st.subheader("🎯 Performance Dashboard")
        
        # Regression metrics
        if 'linear_regression' in metrics:
            st.markdown("### 📊 Linear Regression Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Absolute Error", f"{metrics['linear_regression']['MAE']:.2f}")
                st.info("Lower is better")
            
            with col2:
                st.metric("Root Mean Square Error", f"{metrics['linear_regression']['RMSE']:.2f}")
                st.info("Lower is better")
            
            with col3:
                st.metric("R² Score", f"{metrics['linear_regression']['R2']:.4f}")
                st.info("Higher is better (closer to 1.0)")
        
        # Classification metrics
        st.markdown("### 🎯 Classification Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'decision_tree' in metrics:
                st.markdown("#### 🌳 Decision Tree")
                st.metric("Accuracy", f"{metrics['decision_tree']['accuracy']:.4f}")
                
                # Confusion Matrix
                visualizer = DataVisualizer()
                cm = metrics['decision_tree']['confusion_matrix']
                class_names = ['No', 'Yes']  # For Purchase_Decision
                
                fig = visualizer.plot_confusion_matrix(cm, class_names)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'knn' in metrics:
                st.markdown("#### 🎯 KNN Classifier")
                st.metric("Accuracy", f"{metrics['knn']['accuracy']:.4f}")
                
                # Confusion Matrix
                cm = metrics['knn']['confusion_matrix']
                class_names = ['Low', 'Medium', 'High']  # For Loyalty_Category
                
                visualizer = DataVisualizer()
                fig = visualizer.plot_confusion_matrix(cm, class_names)
                st.plotly_chart(fig, use_container_width=True)
        
        # Clustering metrics
        if 'kmeans' in metrics:
            st.markdown("### 🎨 Clustering Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Clusters", metrics['kmeans']['n_clusters'])
                st.info("Optimal segments for customer grouping")
            
            with col2:
                st.metric("Inertia", f"{metrics['kmeans']['inertia']:.2f}")
                st.info("Lower is better (compact clusters)")
            
            # Cluster distribution
            cluster_labels = metrics['kmeans']['cluster_labels']
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            
            fig = px.bar(x=[f"Cluster {i}" for i in cluster_counts.index], 
                        y=cluster_counts.values,
                        title="Customer Distribution Across Clusters")
            st.plotly_chart(fig, use_container_width=True)
        
        # Overall performance comparison
        st.markdown("### 📊 Overall Model Comparison")
        
        # Create comparison chart
        model_names = []
        accuracies = []
        
        if 'decision_tree' in metrics:
            model_names.append('Decision Tree')
            accuracies.append(metrics['decision_tree']['accuracy'])
        
        if 'knn' in metrics:
            model_names.append('KNN')
            accuracies.append(metrics['knn']['accuracy'])
        
        if accuracies:
            fig = px.bar(x=model_names, y=accuracies, title="Classification Model Accuracy Comparison")
            fig.update_layout(yaxis_title="Accuracy", xaxis_title="Model")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance analysis
        if 'decision_tree' in metrics:
            st.markdown("### 🎯 Feature Importance Analysis")
            
            feature_names = metrics['decision_tree']['feature_names']
            importances = metrics['decision_tree']['feature_importances']
            
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Top 15 features
            top_features = feature_df.head(15)
            
            fig = px.bar(top_features, x='Importance', y='Feature', 
                        orientation='h', title="Top 15 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("📋 Detailed Feature Importance")
            st.dataframe(top_features)
    
    except Exception as e:
        st.error(f"Error loading performance metrics: {str(e)}")

if __name__ == "__main__":
    main()
