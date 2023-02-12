# Hacklytics2023Project
I am pleased to present to you my project on stock market analysis using the K-Means clustering algorithm. In this project, I have applied K-Means clustering to stock market data to group similar stocks together. The aim of this project is to provide insights into the behavior of stocks and to determine which stocks move together, which can be useful for portfolio diversification.

The project begins by loading the stock market data and transforming it using the Normalizer class from scikit-learn. This ensures that each stock has an equal weight in the analysis and the results are not skewed by any one stock.

Next, I used the KMeans class from scikit-learn to perform the clustering. The number of clusters was set to 10, which provides a good balance between having enough granularity to see patterns in the data and not having so many clusters that the results become too cluttered.

In order to visualize the results, I reduced the dimensionality of the data using the PCA (Principal Component Analysis) class from scikit-learn. This allowed me to plot the data in two dimensions and see the clustering results. The plot shows each stock as a dot and the centroids of each cluster as a white X.

The results of this project are promising. By grouping stocks together, we can see which stocks tend to move in similar directions and which stocks are more volatile than others. This information can be used to construct a more diversified portfolio by including stocks from different clusters, which are less likely to move together.

In conclusion, this project demonstrates the power of K-Means clustering for analyzing stock market data and provides useful insights for portfolio diversification. I believe that this project has the potential to be further developed and applied to a wide range of financial data.

Thank you for considering my project. I look forward to your feedback.
