import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\anish\Data Science with Gen AI\Class Codes\K-Clustering\Mall_Customers.csv")

x= dataset.iloc[:, [3,4]].values 
wcss = []
from sklearn.cluster import KMeans
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init= "k-means++",random_state =0 )
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("elbow graph")
plt.xlabel('number of cluester')
plt.ylabel("wcss")
plt.show()


kmeans = KMeans(n_clusters = 5, init = "k-means++",random_state = 0)
y_kmeans = kmeans.fit_predict(x)


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1], s = 100, c = 'red', label = "cluster 1")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1], s = 100, c = 'blue', label = "cluster 2")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,1], s = 100, c = 'green', label = "cluster 3")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3,1], s = 100, c = 'cyan', label = "cluster 4")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4,1], s = 100, c = 'magenta', label = "cluster 5")
plt.title("Cluster of customers")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()

import scipy.cluster.hierarchy as sch 
dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))


plt.title("Dendogram")
plt.xlabel("customer")
plt.ylabel("ecludean distance")
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5 , linkage  = "ward")
y_hc = hc.fit_predict(x)
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0,1], s = 100, c = 'red', label = "cluster 1")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1,1], s = 100, c = 'blue', label = "cluster 2")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2,1], s = 100, c = 'green', label = "cluster 3")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3,1], s = 100, c = 'cyan', label = "cluster 4")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4,1], s = 100, c = 'magenta', label = "cluster 5")
plt.title("Cluster of customers")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()
