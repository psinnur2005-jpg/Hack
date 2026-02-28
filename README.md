# 🏪 Smart Retail Sales Forecasting & Customer Segmentation System

A comprehensive machine learning system for retail analytics that provides sales forecasting, customer behavior prediction, and segmentation analysis.

## 📊 Project Overview

This system implements four different machine learning algorithms to solve various retail business problems:

- **Linear Regression**: Predict monthly sales based on customer and market features
- **Decision Tree Classifier**: Predict customer purchase decisions (Yes/No)
- **KNN Classifier**: Predict customer loyalty categories (Low/Medium/High)
- **K-Means Clustering**: Segment customers into meaningful groups

## 🎯 Key Features

- **Multi-Model Approach**: Four different ML algorithms for different tasks
- **Interactive Web Interface**: Streamlit-based dashboard with 6 pages
- **Real-time Predictions**: Get instant predictions for new customer data
- **Comprehensive Visualizations**: Interactive charts and graphs using Plotly
- **Performance Metrics**: Track model accuracy and performance
- **Data Generation**: Synthetic dataset generator with realistic distributions
- **Model Persistence**: Save and load trained models using Joblib

## 🛠️ Technology Stack

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Additional plotting capabilities
- **Joblib**: Model serialization

## 📁 Project Structure

```
project/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation

├── data/
│   └── synthetic_data.csv          # Generated dataset

├── models/
│   ├── linear_regression.pkl       # Trained Linear Regression model
│   ├── decision_tree.pkl           # Trained Decision Tree model
│   ├── knn.pkl                     # Trained KNN model
│   ├── kmeans.pkl                  # Trained K-Means model
│   ├── preprocessor.pkl            # Data preprocessing pipeline
│   └── metrics.pkl                 # Model performance metrics
│
└── src/
    ├── data_generator.py           # Synthetic data generation
    ├── preprocessing.py            # Data preprocessing pipeline
    ├── train_models.py             # Model training logic
    ├── predict.py                  # Prediction system
    └── visualization.py            # Visualization utilities
```

## 🚀 Installation & Setup

### 1. Clone/Download the Project

```bash
# Navigate to the project directory
cd project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## 📖 Usage Guide

### Page 1: Home
- Project overview and team information
- Key features and technology stack

### Page 2: Data Generator/Upload
- Generate synthetic retail dataset (1000+ records)
- Upload custom CSV files
- Preview dataset with statistics

### Page 3: Model Training
- Train all four ML models with one click
- View training progress and results
- Display model metrics and feature importance

### Page 4: Prediction Interface
- Input customer information through interactive form
- Get real-time predictions from all models:
  - Monthly Sales prediction
  - Purchase Decision (Yes/No)
  - Loyalty Category (Low/Medium/High)
  - Customer Cluster assignment
- View customer segment analysis

### Page 5: Visualization
- Interactive charts and graphs:
  - Sales distribution
  - Income vs Sales scatter plot
  - Sales by product category
  - Customer age distribution
  - Purchase frequency analysis
  - Seasonal demand impact
  - Feature correlation heatmap
  - Customer segment visualization

### Page 6: Model Performance
- Detailed performance metrics for all models
- Confusion matrices for classification models
- Feature importance analysis
- Model comparison charts

## 📊 Dataset Features

The synthetic dataset includes the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Customer_Age | Numerical | Customer age (18-80) |
| Gender | Categorical | Male/Female |
| Annual_Income | Numerical | Annual income in USD |
| Purchase_Frequency | Numerical | Purchases per month |
| Discount_Offered | Numerical | Discount percentage (0-30%) |
| Product_Category | Categorical | 8 different categories |
| Marketing_Spend | Numerical | Marketing budget in USD |
| Seasonal_Demand_Index | Numerical | Seasonal demand factor (0.3-2.0) |
| Store_Location | Categorical | 5 different locations |
| Monthly_Sales | Numerical | Target for regression |
| Purchase_Decision | Categorical | Target for classification (Yes/No) |
| Loyalty_Category | Categorical | Target for classification (Low/Medium/High) |

## 🤖 Machine Learning Models

### 1. Linear Regression
- **Task**: Predict Monthly_Sales
- **Features**: All columns except target variables
- **Metrics**: MAE, RMSE, R²
- **Preprocessing**: Standard scaling + One-hot encoding

### 2. Decision Tree Classifier
- **Task**: Predict Purchase_Decision
- **Features**: All columns except target variables
- **Metrics**: Accuracy, Confusion Matrix
- **Parameters**: max_depth=10, random_state=42

### 3. KNN Classifier
- **Task**: Predict Loyalty_Category
- **Features**: All columns except target variables
- **Metrics**: Accuracy, Confusion Matrix
- **Parameters**: n_neighbors=5, weights='distance'

### 4. K-Means Clustering
- **Task**: Customer segmentation
- **Features**: All columns except target variables
- **Clusters**: 4 segments
- **Metrics**: Inertia, cluster distribution

## 📈 Model Performance

Expected performance metrics (based on synthetic data):

- **Linear Regression**: R² > 0.7, Low MAE/RMSE
- **Decision Tree**: Accuracy > 80%
- **KNN**: Accuracy > 75%
- **K-Means**: 4 distinct customer segments

## 🎨 Visualizations

The system provides various interactive visualizations:

- **Sales Distribution**: Histogram of monthly sales
- **Income vs Sales**: Scatter plot with product category coloring
- **Feature Importance**: Bar chart of most important features
- **Cluster Visualization**: 2D PCA plot of customer segments
- **Confusion Matrices**: Heatmap for classification performance
- **Correlation Heatmap**: Feature correlations

## 🔧 Customization

### Adding New Models
1. Update `train_models.py` to include new model training
2. Update `predict.py` to handle new model predictions
3. Update `app.py` to display new model results

### Modifying Features
1. Update `data_generator.py` to add new columns
2. Update `preprocessing.py` to handle new features
3. Retrain models with updated data

### Changing Visualization
1. Update `visualization.py` to add new chart types
2. Update `app.py` to include new visualization options

## 🐛 Troubleshooting

### Common Issues

1. **Models not found**: Ensure models are trained first in the Model Training page
2. **Data loading error**: Check that dataset exists in `data/` directory
3. **Import errors**: Verify all dependencies are installed correctly
4. **Memory issues**: Reduce dataset size for training if needed

### Performance Tips

- Use smaller dataset sizes for faster training during development
- Close unnecessary browser tabs when running the app
- Ensure sufficient RAM for model training (recommended: 8GB+)

## 📱 Browser Compatibility

The application works best on modern browsers:
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Test with the provided synthetic data
4. Verify all dependencies are installed

---

**Built with ❤️ for retail analytics and business intelligence**

*Last updated: February 2026*
