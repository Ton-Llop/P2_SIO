import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calc
from utils import loadData
from utils import optimizers
import config
from utils import computeMae

def global_mean():
    # Load data
    df = loadData.load_dataset(config.DATA_PATH)
    
    # Calculate global mean
    mean_val = calc.predict_global_mean(df)
    
    # Load target and apply prediction
    target_df = loadData.load_target(config.TARGET_PATH)
    target_df['Rating'] = mean_val
    
    # Save results
    loadData.save_results(target_df, config.OUTPUT_PATH_GLOBAL_MEAN)
    print(f"Global mean results saved to {config.OUTPUT_PATH_GLOBAL_MEAN}")
    
    # Compute MAE
    mae_output = calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_GLOBAL_MEAN)
    return [target_df, mae_output]

def user_mean():
    # Load data
    df = loadData.load_dataset(config.DATA_PATH)
    
    # Calculate user mean
    mean_val = calc.predict_user_mean(df)
    
    # Load target and apply prediction
    target_df = loadData.load_target(config.TARGET_PATH)
    
    # Extract predictions for the target pairs
    preds = []
    # Calculate global mean as fallback
    global_mean_val = calc.predict_global_mean(df)
    
    for row in target_df.itertuples(index=False):
        user = row.User
        item = row.Restaurant
        if user in mean_val.index and item in mean_val.columns:
            preds.append(mean_val.at[user, item])
        else:
            preds.append(global_mean_val)
            
    target_df['Rating'] = preds
    
    # Save results
    loadData.save_results(target_df, config.OUTPUT_PATH_USER_MEAN)
    print(f"User mean results saved to {config.OUTPUT_PATH_USER_MEAN}")
    
    # Compute MAE
    mae_output = calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_USER_MEAN)
    return [target_df, mae_output]

def item_knn_baseline(k = 12, min_common = 15, shrinkage_sim = 10.0, lambda_user = 15.0, lambda_item = 25.0):
    # Load data (matriz User x Restaurant with NaN in the 99)
    df = loadData.load_dataset(config.DATA_PATH)

    # Load target (User;Restaurant;Rating original)
    target_df = loadData.load_target(config.TARGET_PATH)

    # Parameters "good" by default (you can tune them later)
    params = {
        "k": k,
        "min_common": min_common,
        "shrinkage_sim": shrinkage_sim,
        "lambda_user": lambda_user,
        "lambda_item": lambda_item,
    }

    # Generate predictions
    preds_df = calc.predict_item_knn_baseline(
        df,
        target_df,
        **params
    )

    # Save results in CSV with the correct format
    loadData.save_results(preds_df, config.OUTPUT_PATH_ITEM_KNN_BASELINE)
    print(f"Item-kNN + baseline saved to {config.OUTPUT_PATH_ITEM_KNN_BASELINE}")

    # Calculate MAE using the official script
    mae_output = calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_ITEM_KNN_BASELINE)
    return [preds_df, mae_output]

def svd(n_factors = 30, n_epochs = 75, lr = 0.001, reg = 0.2, use_plus_plus = False):
    # Load data (matriz User x Restaurant with NaN in the 99)
    df = loadData.load_dataset(config.DATA_PATH)

    # Load target (User;Restaurant;Rating original)
    target_df = loadData.load_target(config.TARGET_PATH)

    # Parameters "good" by default (you can tune them later)
    params = {
        "n_factors": n_factors,
        "n_epochs": n_epochs,
        "lr": lr,
        "reg": reg,
        "use_plus_plus": use_plus_plus,
    }

    # Generate predictions
    preds_df = calc.predict_svd(
        df,
        target_df,
        **params
    )

    # Save results in CSV with the correct format
    loadData.save_results(preds_df, config.OUTPUT_PATH_SVD)
    print(f"SVD saved to {config.OUTPUT_PATH_SVD}")

    # Calculate MAE using the official script
    mae_output = calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_SVD)
    return [preds_df, mae_output]

def svd_knn(alpha = 0.5):
    # Load data (matriz User x Restaurant with NaN in the 99)
    df = loadData.load_dataset(config.DATA_PATH)

    # Load target (User;Restaurant;Rating original)
    target_df = loadData.load_target(config.TARGET_PATH)

    # Generate knn predictions
    knn_df = item_knn_baseline()

    # Generate svd predictions
    svd_df = svd()
    
    # Parameters "good" by default (you can tune them later)
    params = {
        "alpha": alpha,
    }

    # Generate predictions
    preds_df = calc.svd_knn(
        svd_df[0],
        knn_df[0],
        target_df,
        **params
    )

    # Save results in CSV with the correct format
    loadData.save_results(preds_df, config.OUTPUT_PATH_SVD_KNN)
    print(f"SVD + kNN saved to {config.OUTPUT_PATH_SVD_KNN}")

    # Calculate MAE using the official script
    mae_output = calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_SVD_KNN)
    return [preds_df, mae_output]

def knn_with_zscore(k = 20, min_common = 15, shrinkage_sim = 20.0):
    # Load data (matriz User x Restaurant with NaN in the 99)
    df = loadData.load_dataset(config.DATA_PATH)

    # Load target (User;Restaurant;Rating original)
    target_df = loadData.load_target(config.TARGET_PATH)

    # Parameters "good" by default (you can tune them later)
    params = {
        "k": k,
        "min_common": min_common,
        "shrinkage_sim": shrinkage_sim,
    }

    # Generate predictions
    preds_df = calc.knn_with_zscore(
        df,
        target_df,
        **params
    )

    # Save results in CSV with the correct format
    loadData.save_results(preds_df, config.OUTPUT_PATH_KNN_WITH_ZSCORE)
    print(f"Item-kNN with ZScore saved to {config.OUTPUT_PATH_KNN_WITH_ZSCORE}")

    # Calculate MAE using the official script
    mae_output = calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_KNN_WITH_ZSCORE)
    return [preds_df, mae_output]
    


def knn_svd_knnz(alpha_knn=0.55, alpha_svd=0.4, alpha_knnz=0.05):
    # Load data (matriz User x Restaurant with NaN in the 99)
    df = loadData.load_dataset(config.DATA_PATH)

    # Load target (User;Restaurant;Rating original)
    target_df = loadData.load_target(config.TARGET_PATH)

    # Generate predictions
    knn_df = knn_with_zscore()[0]
    svd_df = svd()[0]
    knnz_df = knn_with_zscore()[0]

    # Parameters "good" by default (you can tune them later)
    params = {
        "alpha_knn": alpha_knn,
        "alpha_svd": alpha_svd,
        "alpha_knnz": alpha_knnz,
    }

    # Generate predictions
    preds_df = calc.knn_svd_knnz(
        knn_df,
        svd_df,
        knnz_df,
        target_df,
        **params
    )

    # Save results in CSV with the correct format
    loadData.save_results(preds_df, config.OUTPUT_PATH_KNN_SVD_KNNZ)
    print(f"KNN + SVD + KNNZ saved to {config.OUTPUT_PATH_KNN_SVD_KNNZ}")

    # Calculate MAE using the official script
    mae_output = calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_KNN_SVD_KNNZ)
    return [preds_df, mae_output]

if __name__ == "__main__":
    print("MAE Global mean: ", global_mean()[1])
    print("MAE User mean: ", user_mean()[1])       
    print("MAE Item-kNN + baseline: ", item_knn_baseline()[1])
    print("MAE SVD: ", svd()[1])
    print("MAE SVD + kNN: ", svd_knn()[1])
    print("MAE Item-kNN with ZScore: ", knn_with_zscore()[1])
    print("MAE KNN + SVD + KNNZ: ", knn_svd_knnz()[1])
    
    # Use of optimizers
    # optimizers.optimize_item_knn()
    # optimizers.optimize_weights_knn_svd()
    # optimizers.optimize_weights_knn_svd_knnz()