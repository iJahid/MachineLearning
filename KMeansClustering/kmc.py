import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('income.csv')
# Name,Age,Income($)
# plt.scatter(df['Age'], df['Income($)'], marker="o")
# plt.show()


def mCluster(data):
    km = KMeans(n_clusters=3)
    y_pred = km.fit_predict(df[['Age', 'Income($)']])
    data['cluster'] = y_pred
    print(y_pred)
    df1 = data[data.cluster == 0]
    df2 = data[data.cluster == 1]
    df3 = data[data.cluster == 2]
    # df4 = df[df.cluster == 3]
    plt.scatter(df1['Age'], df1['Income($)'],
                marker="+", color='green', label="cl 1")
    plt.scatter(df2['Age'], df2['Income($)'],
                marker="*", color='red', label="cl 2")
    plt.scatter(df3['Age'], df3['Income($)'],
                marker="o", color='blue', label="cl 3")
    # plt.scatter(df4['Age'], df4['Income($)'], marker=".", color='yellow')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
                :, 1], marker=".", color='purple', linewidths=5, label="Centroid")
    plt.xlabel("Age")
    plt.ylabel("Income")
    # bbox_to_anchor=[0.1, 0.1],
    plt.legend(
        loc='upper left')
    plt.show()


mCluster(df)
# we find that kmean is not good enough to classify data.
# So we need Scale the income 1 to 0
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
print(df.head)
mCluster(df)

# as above its a small dataset we can manually get the kmean by add/sub n_cluster
# for big dataset we need elbo methode to find the kmean
# for that we need sum of squar error =sse of 1 to 10 or 100 of n_cluster

k_rng = range(1, 10)
sse = []

for k in k_rng:
    km = KMeans(k)
    km = km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

print(sse)

plt.plot(k_rng, sse, marker='o')
plt.show()
# from the plot we find 3 is the elbo point so cluster number 3 is best which already we found earlier
