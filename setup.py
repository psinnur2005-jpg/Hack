from setuptools import setup, find_packages

setup(
    name="smart-retail-system",
    version="1.0.0",
    description="Smart Retail Sales Forecasting & Customer Segmentation System",
    author="ML & Full Stack AI Team",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "matplotlib",
        "seaborn",
        "joblib",
    ],
    python_requires=">=3.8",
)
