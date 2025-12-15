import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture # Replaced KMeans with GMM
from sklearn.preprocessing import StandardScaler
import random

def pcaCalc (features, saveFig, outputPath, slideID, epoch, slideName):
    '''
    Performs PCA followed by Gaussian Mixture Model (GMM) clustering.
    Selects the optimal number of components using BIC.
    Specifically, the features are extracted from the penultimate layer of the
    fully connected layers of the MIL-ResNet model for each tile of a WSI.
    '''
    features = pd.DataFrame(features)
    # saveFig = True
    probs = list(features[4])

    features_for_pca = features.drop(4, axis=1)
    print(features_for_pca)


    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_pca)
    pca_clustering = PCA(n_components=0.95)
    clustering_features = pca_clustering.fit_transform(features_scaled)


    pca_plotting = PCA(n_components=2)
    plotting_features = pca_plotting.fit_transform(features_scaled)
    principal_tiles_Df = pd.DataFrame(data = plotting_features
                                      , columns = ['principal component 1', 'principal component 2'])


    n_components_range = range(2, min(6, len(probs)))
    best_gmm = None
    lowest_bic = np.inf

    if len(probs) < 2:
        return list(range(len(probs)))

    for n_c in n_components_range:
        gmm = GaussianMixture(n_components=n_c, n_init=5, random_state=42)
        gmm.fit(clustering_features)
        bic = gmm.bic(clustering_features)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    # Fallback if loop didn't run (rare)
    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=2, n_init=5, random_state=42).fit(clustering_features)

    cluster_probs = best_gmm.predict_proba(clustering_features)
    labels = best_gmm.predict(clustering_features)
    bestK = best_gmm.n_components

    # --- choose target GMM component (soft, probability-weighted) ---
    cluster_scores = []
    for k in range(cluster_probs.shape[1]):
        score = np.sum(cluster_probs[:, k] * probs) / (np.sum(cluster_probs[:, k]) + 1e-8)
        cluster_scores.append(score)

    target_cluster_label = int(np.argmax(cluster_scores))

    # --- percentile-based ROI selection ---
    target_probs = cluster_probs[:, target_cluster_label]

    perc = 95
    thr = np.percentile(target_probs, perc)
    selected_indices = np.where(target_probs >= thr)[0].tolist()

    # --- fallback for weak signal ---
    if len(selected_indices) == 0:
        print("used fallback")
        K = min(3, len(target_probs))
        selected_indices = list(np.argsort(target_probs)[-K:])

    if saveFig:
        labels_color = ['goldenrod' if x == max(probs) else 'darkred' if x > 0.5 else 'teal' for x in probs]
        edgecolor = ['darkred' if x > 0.5 else 'teal' for x in probs]

        plt.figure(figsize=(10, 8))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel('Principal Component - 1 (Scaled)',fontsize=20)
        plt.ylabel('Principal Component - 2 (Scaled)',fontsize=20)
        plt.title(f"GMM Selection (k={bestK})",fontsize=20)

        plt.scatter(principal_tiles_Df['principal component 1'],
                    principal_tiles_Df['principal component 2'],
                    c = probs, cmap='hot', s = 50, alpha=0.6, edgecolor=edgecolor)

        newPointsX, newPointsY = zip(*[[principal_tiles_Df['principal component 1'][x],principal_tiles_Df['principal component 2'][x]] for x in selected_indices])
        plt.scatter(newPointsX,newPointsY, alpha=0.6, edgecolor='lime', marker="*", s=120, label='Selected')



        plt.legend()
        plt.savefig(outputPath + "slide_" + slideName + "_epoch_" + str(epoch) + ".pdf",  bbox_inches='tight')
        plt.close()

    return(selected_indices)

def readFeatures (file):
    features = pd.read_csv(file, sep="\t", header=None, usecols=[i for i in range(4, 517)])
    return(features)