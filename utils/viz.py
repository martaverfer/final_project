# 📚 Basic libraries
import pandas as pd

# 📊 Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def two_plot_distribution(column1, column2):
    """
    Plot two histograms
    """
    # Plot the histogram
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 2, 1)
    plt.hist(column1, bins=40, color='#2e59a7', edgecolor='navy')
    # Adding labels and title to the plot
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Score', fontsize=14)
    # Show Grid for Better Readability
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.subplot(1, 2, 2)
    plt.hist(column2, bins=40, color='#fcac3d', edgecolor='navy')
    # Adding labels and title to the plot
    plt.xlabel('Sentiment Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Sentiment Score', fontsize=14)
    # Show Grid for Better Readability
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    # Show the plot
    plt.show()

def scatter_plot(column1, column2, df):
    """
    Scatter plot
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column1, y=column2, data=df, alpha=0.5)
    plt.title('Sentiment vs. Average Score')
    plt.xlabel('VADER Sentiment Score')
    plt.ylabel('Average Score')
    plt.grid(True)
    plt.show()

def bar_plot(column_counts):
    column_counts.plot(kind='bar', color='#2e59a7', edgecolor='black')
    plt.title('Number of Books per Genre')
    plt.ylabel('Count')
    plt.xlabel('Genre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.show()

def elbow_silhouette_plot(k_values, inertia, silhouette_sc):
    """
    Elbow and Silhouette plot
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia, marker='o', color='#2e59a7')
    plt.title('KMeans Inertia for Different Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_sc, marker='o', color='#f45830')
    plt.title('Silhouette Score for Different Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def kmeans_clusters_2D(X_pca, column):
    """
    KMeans Clustering 2D
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=column, cmap='viridis', alpha=0.6)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('KMeans Clusters (2D PCA)')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def kmeans_clusters_3D(X_pca, column):
    """
    KMeans Clustering 3D
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
        c=column, cmap='viridis', alpha=0.7, s=30
    )

    ax.set_title('KMeans Clusters (3D PCA)')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    fig.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.show()

def violin_plot(column1, column2, df, palette, title, ylabel):
    """
    Violin plot
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=column1, y=column2, data=df, palette=palette)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def heatmap_plot(corr_matrix):
    """
    Heatmap plot
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Proportion'})
    plt.title('Proportion of Each Genre in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()
