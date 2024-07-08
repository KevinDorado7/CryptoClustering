# CryptoClustering
Module 19 - Crypto Clustering Challenge

# Cryptocurrency Clustering Analysis

This repository contains the analysis and clustering of cryptocurrencies using K-Means clustering and Principal Component Analysis (PCA). The goal is to identify the impact of 24-hour and 7-day price changes on the clustering of cryptocurrencies.

## Libraries Used

- pandas
- hvplot.pandas
- scikit-learn (StandardScaler, KMeans, PCA)

## Analysis Steps

1. **Import Required Libraries**:
   - Imported necessary libraries for data manipulation, visualization, and machine learning.

2. **Load and Preprocess Data**:
   - Loaded the cryptocurrency market data from a CSV file.
   - Displayed sample data and generated summary statistics.
   - Plotted initial data to visualize what is in the DataFrame.

3. **Prepare the Data**:
   - Used `StandardScaler` from scikit-learn to normalize the data.
   - Created a DataFrame with the scaled data.
   - Set the `coin_id` column as the index.

4. **Find the Best Value for k Using the Original Data**:
   - Implemented the elbow method algorithm to find the best value for `k` using a range from 1 to 11.
   - Plotted a line chart of all the inertia values to visually identify the optimal value for `k`.
   - Determined that the best value for `k` is 4.

5. **Cluster the Cryptocurrencies with K-Means Using the Original Data**:
   - Initialized the K-Means model with four clusters.
   - Fitted the K-Means model using the original data.
   - Predicted the clusters and reviewed the resulting array of cluster values.
   - Created a copy of the original data and added a new column with the predicted clusters.
   - Created a scatter plot using `hvPlot` to visualize the clusters.

6. **Optimize the Clusters with Principal Component Analysis (PCA)**:
   - Created a PCA model instance and set `n_components=3`.
   - Used the PCA model to reduce the features to three principal components.
   - Reviewed the first five rows of the DataFrame with PCA data.
   - Retrieved the explained variance to determine how much information can be attributed to each principal component.
   - Determined that the total explained variance of the three principal components is approximately 0.892.
   - Created a new DataFrame with the PCA data.

7. **Find the Best Value for k Using the PCA Data**:
   - Implemented the elbow method algorithm using the PCA data to find the best value for `k`.
   - Plotted a line chart of all the inertia values to visually identify the optimal value for `k`.
   - Determined that the best value for `k` is 4, consistent with the original data.

8. **Cluster the Cryptocurrencies with K-Means Using the PCA Data**:
   - Initialized the K-Means model with four clusters using the best value for `k`.
   - Fitted the K-Means model using the PCA data.
   - Predicted the clusters and reviewed the resulting array of cluster values.
   - Created a copy of the DataFrame with the PCA data and added a new column to store the predicted clusters.
   - Created a scatter plot using `hvPlot` to visualize the clusters with PCA data.

9. **Visualize and Compare the Results**:
   - Created composite plots to compare the elbow curves and the cryptocurrency clusters from both the original data and PCA data.
   - Analyzed the impact of using fewer features to cluster the data, concluding that PCA leads to clearer and more distinct clusters.

## Files in the Repository

- `crypto_market_data.csv`: The CSV file containing the cryptocurrency market data.
- `Crypto_Clustering_starter_code.ipynb`: The Jupyter Notebook with the complete analysis and clustering steps.
- `README.md`: This file, describing the analysis, steps taken, libraries used, and contents of the repository.

In this project, ChatGPT was utilized to assist with code functionality and optimization, also with several key tasks in clustering analysis, including:

- Explaining the process of optimizing clusters with Principal Component Analysis (PCA) and understanding the explained variance.

### Reference

- OpenAi. (n.d.). ChatGPT by OpenAi from https://openai.com/chatgpt
