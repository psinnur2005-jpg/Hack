import pandas as pd
import numpy as np
from datetime import datetime
import random

def generate_synthetic_data(n_samples=1500):
    """
    Generate synthetic retail dataset with realistic distributions
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate customer demographics
    customer_age = np.random.normal(35, 12, n_samples).astype(int)
    customer_age = np.clip(customer_age, 18, 80)
    
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
    
    # Annual income with realistic distribution (in thousands)
    annual_income = np.random.lognormal(4.2, 0.5, n_samples) * 1000
    annual_income = np.clip(annual_income, 20000, 200000)
    
    # Purchase behavior
    purchase_frequency = np.random.poisson(8, n_samples)
    purchase_frequency = np.clip(purchase_frequency, 1, 30)
    
    discount_offered = np.random.uniform(0, 30, n_samples).round(1)
    
    # Product categories
    product_categories = ['Electronics', 'Clothing', 'Groceries', 'Home & Garden', 
                         'Sports', 'Books', 'Toys', 'Beauty']
    product_category = np.random.choice(product_categories, n_samples, 
                                      p=[0.15, 0.20, 0.25, 0.10, 0.08, 0.07, 0.10, 0.05])
    
    # Marketing spend (in thousands)
    marketing_spend = np.random.exponential(5, n_samples) * 1000
    marketing_spend = np.clip(marketing_spend, 500, 50000)
    
    # Seasonal demand index (1.0 = normal, >1.0 = high demand)
    seasonal_demand_index = np.random.normal(1.0, 0.3, n_samples)
    seasonal_demand_index = np.clip(seasonal_demand_index, 0.3, 2.0)
    
    # Store locations
    store_locations = ['Downtown', 'Suburban', 'Mall', 'Airport', 'Online']
    store_location = np.random.choice(store_locations, n_samples, 
                                    p=[0.25, 0.30, 0.20, 0.10, 0.15])
    
    # Calculate Monthly_Sales based on features with some noise
    base_sales = (annual_income / 12) * 0.05  # 5% of monthly income
    frequency_factor = purchase_frequency * 50
    marketing_factor = marketing_spend * 0.01
    seasonal_factor = seasonal_demand_index * 200
    discount_factor = discount_offered * 10
    
    monthly_sales = (base_sales + frequency_factor + marketing_factor + 
                    seasonal_factor + discount_factor + np.random.normal(0, 100, n_samples))
    monthly_sales = np.clip(monthly_sales, 50, 5000)
    
    # Purchase Decision (Yes/No) based on probability
    purchase_prob = (seasonal_demand_index * 0.3 + 
                    (discount_offered / 30) * 0.4 + 
                    (purchase_frequency / 30) * 0.3)
    purchase_prob = np.clip(purchase_prob, 0.1, 0.9)  # Ensure valid probabilities
    purchase_decision = []
    for prob in purchase_prob:
        decision = np.random.choice(['Yes', 'No'], p=[prob, 1-prob])
        purchase_decision.append(decision)
    
    # Loyalty Category based on purchase frequency and income
    loyalty_score = (purchase_frequency / 30) + (annual_income / 200000)
    loyalty_category = []
    for score in loyalty_score:
        if score < 0.3:
            loyalty_category.append('Low')
        elif score < 0.7:
            loyalty_category.append('Medium')
        else:
            loyalty_category.append('High')
    
    # Create DataFrame
    data = pd.DataFrame({
        'Customer_Age': customer_age,
        'Gender': gender,
        'Annual_Income': annual_income.round(2),
        'Purchase_Frequency': purchase_frequency,
        'Discount_Offered': discount_offered,
        'Product_Category': product_category,
        'Marketing_Spend': marketing_spend.round(2),
        'Seasonal_Demand_Index': seasonal_demand_index.round(2),
        'Store_Location': store_location,
        'Monthly_Sales': monthly_sales.round(2),
        'Purchase_Decision': purchase_decision,
        'Loyalty_Category': loyalty_category
    })
    
    return data

def save_dataset(data, filepath):
    """Save dataset to CSV file"""
    data.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Dataset shape: {data.shape}")
    print("\nDataset info:")
    print(data.info())
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nSummary statistics:")
    print(data.describe())

if __name__ == "__main__":
    # Generate and save dataset
    data = generate_synthetic_data(1500)
    save_dataset(data, '../data/synthetic_data.csv')
