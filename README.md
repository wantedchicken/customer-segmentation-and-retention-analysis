# Customer Segmentation & Retention Analysis

## Overview
This project builds an end-to-end customer analytics pipeline on a large-scale e-commerce dataset (~541K transactions) to segment customers and analyze retention behavior. Using RFM-based segmentation, K-Means clustering, and cohort-based retention analysis, the project identifies high-value and at-risk customers and derives actionable insights to improve retention and revenue.


## Project Structure
Customer-segmentation-retention/
├── data/
│   └── Online Retail.xlsx
├── notebooks/
│   └── customer_segmentation.ipynb
├── images/
│   └── retention_heatmap.png
├── README.md
└── requirements.txt


## Dataset
Online Retail Dataset (UCI Machine Learning Repository), containing ~541K transactions from Dec 2010 to Dec 2011.

## Approach
- Data cleaning and validation
- Feature engineering (Recency, Frequency, Monetary)
- RFM segmentation and customer personas
- K-Means clustering with elbow method
- Cohort-based retention analysis and visualization

## Key Insights
- A small fraction of customers contributes a large share of revenue
- Customer churn is highest within the first two months after acquisition
- ML-based clusters closely align with business-driven RFM segments

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook
