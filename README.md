# Mall Customer Segmentation


## Project Abstract:
This project applies K-Means clustering to segment customers of a mall based on their age, annual income, and spending score. Using unsupervised learning, the model groups customers into distinct segments, allowing for targeted marketing strategies and personalized services. The project explores multiple cluster numbers, evaluates model performance using silhouette scores, and identifies six optimal customer groups. This segmentation provides insights into consumer behavior, helping businesses to enhance customer engagement and optimize marketing efforts.

## Why Use Customer Segmentation in "MallCustomerSeg":
Customer segmentation in the MallCustomerSeg project helps to group customers based on their spending behavior, age, and income. It allows businesses to understand different customer groups and tailor marketing strategies accordingly. Segmentation can identify high-spending customers, frequent visitors, or those with potential for upselling. This approach enhances personalized marketing, improves customer satisfaction, and drives business growth.

## Overview
This project uses *K-Means clustering* to segment customers of a mall based on their *age, **annual income, and **spending score*. By grouping customers into clusters, businesses can better understand customer behavior and tailor marketing strategies to specific segments, improving customer satisfaction and sales.

### Features of the Dataset:
- *CustomerID*: Unique ID for each customer.
- *Genre*: Gender of the customer (Male/Female).
- *Age*: Age of the customer.
- *Annual Income (k$)*: Annual income in thousands of dollars.
- *Spending Score (1-100)*: A score based on customer spending habits (1 being low and 100 being high).

## Project Objectives
The goal of this project is to:
- Segment customers into different groups based on their demographics and spending patterns.
- Identify customer segments that exhibit similar behaviors, which can then be targeted for specific marketing campaigns.
  
## Files
- *Mall_Customers.csv*: The dataset containing customer information.
- *mallcustomerseg.ipynb*: Jupyter Notebook with the implementation of the K-Means clustering algorithm.
- *kmeans_model.pkl*: A saved K-Means model for predicting clusters on new customer data.

## How to Run
1. *Requirements*:
   - Python 3.x
   - Libraries: 
     - pandas
     - numpy
     - matplotlib
     - seaborn
     - scikit-learn
     - joblib

   Install the required libraries using:
   bash
   pip install -r requirements.txt
   

2. *Steps*:
   - Load the dataset (Mall_Customers.csv).
   - Preprocess the data (handle missing values if necessary, and scale the features).
   - Apply *K-Means clustering* on the selected features (Age, Annual Income (k$), and Spending Score (1-100)).
   - Visualize the clusters.
   - Save the model for future use.

3. *Run the Notebook*:
   - Open the mallcustomerseg.ipynb file using Jupyter Notebook and run all cells to execute the clustering analysis.
   - The model will predict the clusters and visualize the customer segments.

## Key Components

### 1. Data Preprocessing
   The features (Age, Annual Income, and Spending Score) are extracted and scaled using StandardScaler to normalize the data.

### 2. K-Means Clustering
   The K-Means algorithm is applied to segment customers into groups. Different values of *k* (number of clusters) are tested, and the optimal number of clusters is determined using the *silhouette score*.

   Example code for training:
   python
   from sklearn.cluster import KMeans
   
   # Initializing and fitting the model
   kmeans = KMeans(n_clusters=6, random_state=42)
   kmeans.fit(X)  # X is the selected features

   # Predict clusters
   df['cluster'] = kmeans.predict(X)
   

### 3. Model Saving and Loading
   The trained model is saved using joblib for reuse. It can be loaded later for predicting clusters on new data.

   Example code for saving:
   python
   import joblib
   joblib.dump(kmeans, 'kmeans_model.pkl')
   

### 4. Evaluation
   The clustering is evaluated using the *silhouette score* to determine how well-defined the clusters are. Visualizations like scatter plots of the clusters are also generated to help understand the distribution of customer segments.

## Results
- The dataset was segmented into *6 clusters* based on the chosen features.
- Customers were grouped into distinct segments with similar spending patterns and demographics.
- This segmentation can be used by mall management to target different customer groups with personalized marketing strategies.

## Usage
You can use the saved K-Means model to predict customer segments on new data. Simply load the model and use the predict method to assign a customer to a segment.

Example:
python
loaded_model = joblib.load('kmeans_model.pkl')
new_data = [[25, 50, 65]]  # Example: Age 25, Annual Income 50k, Spending Score 65
cluster = loaded_model.predict(new_data)
print(f"The customer belongs to cluster: {cluster}")


## Conclusion
The *Mall Customer Segmentation* project demonstrates how unsupervised machine learning can be applied to segment customers based on their behavior. The K-Means algorithm was effective in identifying distinct customer groups, providing valuable insights for targeted marketing.

---
