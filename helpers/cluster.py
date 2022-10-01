import pandas as pd
import matplotlib.pyplot as plt

SEED = 8

def build_kmeans_clusterer(df, cols, k, random_state=SEED):
    from sklearn.cluster import KMeans
    clusterer = KMeans(n_clusters=k, random_state=random_state)
    clusterer.fit(df[cols])
    return clusterer

def get_kmeans_clusters(df, cols, k=5, clusterer=None):
    if clusterer == None:
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=k)
        clusterer.fit(df[cols])
    s = clusterer.predict(df[cols])
    return s

def find_k(df, cluster_vars, k_range, seed=SEED):
    from sklearn.cluster import KMeans
    sse = []
    for k in k_range:
        if k < 1: continue
        kmeans = KMeans(n_clusters=k, random_state=seed)

        # X[0] is our df dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(df[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

    return k_comparisons_df