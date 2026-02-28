import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class DataVisualizer:
    """
    Create various visualizations for the retail dataset and model results
    """
    
    def __init__(self):
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_sales_distribution(self, data):
        """
        Create sales distribution histogram
        """
        fig = px.histogram(data, x='Monthly_Sales', nbins=30, 
                          title='Monthly Sales Distribution',
                          color_discrete_sequence=['#636EFA'])
        fig.update_layout(
            xaxis_title='Monthly Sales ($)',
            yaxis_title='Frequency',
            showlegend=False
        )
        return fig
    
    def plot_income_vs_sales(self, data):
        """
        Create scatter plot of Income vs Sales
        """
        fig = px.scatter(data, x='Annual_Income', y='Monthly_Sales',
                        color='Product_Category',
                        title='Annual Income vs Monthly Sales by Product Category',
                        hover_data=['Customer_Age', 'Purchase_Frequency'])
        fig.update_layout(
            xaxis_title='Annual Income ($)',
            yaxis_title='Monthly Sales ($)'
        )
        return fig
    
    def plot_feature_importance(self, feature_names, importances, title="Feature Importance"):
        """
        Create feature importance plot for Decision Tree
        """
        # Create DataFrame for better plotting
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_importance_df.tail(10), x='importance', y='feature',
                    orientation='h', title=title,
                    color_discrete_sequence=['#00CC96'])
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Features'
        )
        return fig
    
    def plot_cluster_visualization(self, data, cluster_labels):
        """
        Create cluster visualization using PCA for dimensionality reduction
        """
        from sklearn.decomposition import PCA
        
        # Prepare data for clustering
        numerical_features = ['Customer_Age', 'Annual_Income', 'Purchase_Frequency', 
                             'Discount_Offered', 'Marketing_Spend', 'Seasonal_Demand_Index']
        
        # Select numerical data and scale it
        from sklearn.preprocessing import StandardScaler
        X = data[numerical_features]
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': cluster_labels,
            'Annual_Income': data['Annual_Income'],
            'Monthly_Sales': data['Monthly_Sales']
        })
        
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Cluster',
                        title='Customer Segments (PCA Visualization)',
                        hover_data=['Annual_Income', 'Monthly_Sales'],
                        color_continuous_scale=px.colors.qualitative.Set3)
        fig.update_layout(
            xaxis_title=f'PCA1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            yaxis_title=f'PCA2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
        )
        return fig
    
    def plot_confusion_matrix(self, cm, class_names):
        """
        Create confusion matrix heatmap
        """
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       x=class_names, y=class_names,
                       title='Confusion Matrix',
                       color_continuous_scale='Blues')
        fig.update_layout(
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        return fig
    
    def plot_sales_by_category(self, data):
        """
        Create box plot of sales by product category
        """
        fig = px.box(data, x='Product_Category', y='Monthly_Sales',
                    title='Monthly Sales Distribution by Product Category')
        fig.update_layout(
            xaxis_title='Product Category',
            yaxis_title='Monthly Sales ($)',
            xaxis_tickangle=-45
        )
        return fig
    
    def plot_customer_age_distribution(self, data):
        """
        Create age distribution by loyalty category
        """
        fig = px.histogram(data, x='Customer_Age', color='Loyalty_Category',
                          barmode='overlay', nbins=20,
                          title='Customer Age Distribution by Loyalty Category')
        fig.update_layout(
            xaxis_title='Customer Age',
            yaxis_title='Frequency'
        )
        return fig
    
    def plot_purchase_frequency_analysis(self, data):
        """
        Analyze purchase frequency patterns
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Frequency vs Sales', 'Frequency by Loyalty',
                          'Frequency Distribution', 'Discount vs Frequency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Frequency vs Sales
        fig.add_trace(
            go.Scatter(x=data['Purchase_Frequency'], y=data['Monthly_Sales'],
                      mode='markers', name='Sales vs Frequency'),
            row=1, col=1
        )
        
        # Frequency by Loyalty
        for loyalty in data['Loyalty_Category'].unique():
            subset = data[data['Loyalty_Category'] == loyalty]
            fig.add_trace(
                go.Histogram(x=subset['Purchase_Frequency'], name=f'Loyalty: {loyalty}',
                           opacity=0.7), row=1, col=2
            )
        
        # Frequency Distribution
        fig.add_trace(
            go.Histogram(x=data['Purchase_Frequency'], name='Frequency Dist'),
            row=2, col=1
        )
        
        # Discount vs Frequency
        fig.add_trace(
            go.Scatter(x=data['Discount_Offered'], y=data['Purchase_Frequency'],
                      mode='markers', name='Discount vs Frequency'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Purchase Frequency Analysis",
            showlegend=True
        )
        
        return fig
    
    def plot_seasonal_demand_impact(self, data):
        """
        Analyze seasonal demand impact on sales
        """
        # Create seasonal bins
        data['Seasonal_Bin'] = pd.cut(data['Seasonal_Demand_Index'], 
                                     bins=[0, 0.7, 1.0, 1.3, 2.0],
                                     labels=['Low', 'Normal', 'High', 'Peak'])
        
        fig = px.box(data, x='Seasonal_Bin', y='Monthly_Sales',
                    title='Sales Impact by Seasonal Demand Index',
                    color='Seasonal_Bin')
        fig.update_layout(
            xaxis_title='Seasonal Demand Category',
            yaxis_title='Monthly Sales ($)'
        )
        return fig
    
    def plot_correlation_heatmap(self, data):
        """
        Create correlation heatmap for numerical features
        """
        numerical_features = ['Customer_Age', 'Annual_Income', 'Purchase_Frequency', 
                             'Discount_Offered', 'Marketing_Spend', 
                             'Seasonal_Demand_Index', 'Monthly_Sales']
        
        corr_matrix = data[numerical_features].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title='Feature Correlation Heatmap',
                       color_continuous_scale='RdBu_r')
        fig.update_layout(
            width=600, height=500
        )
        return fig
    
    def create_model_performance_dashboard(self, metrics):
        """
        Create a comprehensive dashboard showing model performance
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Regression Metrics', 'Classification Accuracy',
                          'Cluster Distribution', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Regression metrics
        if 'linear_regression' in metrics:
            reg_metrics = metrics['linear_regression']
            fig.add_trace(
                go.Bar(x=['MAE', 'RMSE'], y=[reg_metrics['MAE'], reg_metrics['RMSE']],
                      name='Regression Metrics'), row=1, col=1
            )
        
        # Classification accuracy
        clf_accuracies = []
        clf_names = []
        for model_name in ['decision_tree', 'knn']:
            if model_name in metrics:
                clf_accuracies.append(metrics[model_name]['accuracy'])
                clf_names.append(model_name.replace('_', ' ').title())
        
        if clf_accuracies:
            fig.add_trace(
                go.Bar(x=clf_names, y=clf_accuracies, name='Accuracy'),
                row=1, col=2
            )
        
        # Cluster distribution
        if 'kmeans' in metrics:
            cluster_labels = metrics['kmeans']['cluster_labels']
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            fig.add_trace(
                go.Pie(labels=[f'Cluster {i}' for i in cluster_counts.index],
                      values=cluster_counts.values, name='Clusters'),
                row=2, col=1
            )
        
        # Feature importance (from decision tree)
        if 'decision_tree' in metrics:
            feature_names = metrics['decision_tree']['feature_names']
            importances = metrics['decision_tree']['feature_importances']
            
            # Get top 5 features
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(5)
            
            fig.add_trace(
                go.Bar(x=feature_imp_df['importance'], y=feature_imp_df['feature'],
                      orientation='h', name='Feature Importance'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Model Performance Dashboard",
            showlegend=False
        )
        
        return fig

if __name__ == "__main__":
    # Test the visualizer
    visualizer = DataVisualizer()
    
    # Load data
    data = pd.read_csv('../data/synthetic_data.csv')
    print("Data loaded for visualization testing")
    
    # Test a few visualizations
    fig1 = visualizer.plot_sales_distribution(data)
    fig2 = visualizer.plot_income_vs_sales(data)
    fig3 = visualizer.plot_correlation_heatmap(data)
    
    print("Visualizations created successfully!")
    print("To see the plots, use fig.show() in your Streamlit app")
