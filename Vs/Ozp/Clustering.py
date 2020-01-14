import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Artificial data creation (for testing purpose)
age = [18, 21, 22, 24, 26, 26, 27, 30, 31, 35, 39, 40, 41, 42, 44, 46, 47, 48, 49, 54]
salary = [10000, 11000, 22000, 15000, 12000, 13000, 14000, 33000, 39000, 37000, 44000, 27000, 29000, 20000, 28000, 21000, 30000, 31000, 23000, 24000]

customers_df = pd.DataFrame({'Age':age, 'Salary':salary})

# Normalization of input attributes
scaler = MinMaxScaler()
scaler.fit(customers_df)
customers_df = scaler.transform(customers_df)

# Set params for algorithm
kmeans_algorithm = KMeans(n_clusters = 3, max_iter = 100)

# Train model
cluster_model = kmeans_algorithm.fit(customers_df)

# Predict -- assign instances to their clusters
clusters = cluster_model.predict(customers_df) # predict returns cluster labels

customers_df = pd.DataFrame(customers_df)
customers_df.columns = ['Age', 'Salary']
customers_df['Cluster'] = clusters # adding cluster labels to dataframe
centers = cluster_model.cluster_centers_

# First we will assign color to each cluster laberl
cluster_color_map = {0:'g', 1:'b', 2:'r'}
# Than we will create a list with cluster colors for each instance (Note that this could be done with Map function)
cluster_color = [cluster_color_map[i] for i in customers_df.Cluster]
plt.close()
plt.xlabel('Age')
plt.scatter(customers_df.Age, customers_df.Salary, c = cluster_color, label = customers_df.Cluster)
plt.scatter(centers[:,0], centers[:,1], marker='+', s = 200, c = 'black')
plt.show()