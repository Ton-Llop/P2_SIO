import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to project root
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'recommendation_dataset.csv')
TARGET_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'target_recommendations.csv')
OUTPUT_PATH_GLOBAL_MEAN = os.path.join(PROJECT_ROOT, 'data', 'output', 'global_mean_predictions.csv')
OUTPUT_PATH_USER_MEAN = os.path.join(PROJECT_ROOT, 'data', 'output', 'user_mean_predictions.csv')
OUTPUT_PATH_ITEM_KNN_BASELINE = os.path.join(PROJECT_ROOT, 'data', 'output', 'item_knn_baseline.csv')
OUTPUT_PATH_USER_MEAN_KNN = os.path.join(PROJECT_ROOT, 'data', 'output', 'user_mean_knn.csv')
OUTPUT_PATH_SVD = os.path.join(PROJECT_ROOT, 'data', 'output', 'svd_predictions.csv')
OUTPUT_PATH_SVD_KNN = os.path.join(PROJECT_ROOT, 'data', 'output', 'svd_knn_predictions.csv')
OUTPUT_PATH_KNN_WITH_ZSCORE = os.path.join(PROJECT_ROOT, 'data', 'output', 'knn_with_zscore_predictions.csv')
OUTPUT_PATH_CO_CLUSTERING = os.path.join(PROJECT_ROOT, 'data', 'output', 'co_clustering_predictions.csv')
OUTPUT_PATH_KNN_SVD_KNNZ = os.path.join(PROJECT_ROOT, 'data', 'output', 'knn_svd_knnz_predictions.csv')
