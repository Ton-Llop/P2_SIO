import pandas as pd
import numpy as np
import os
import subprocess
import sys
from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithZScore

def computeMae(target_file_path, predicted_file_path):
    try: 
        result = subprocess.run(
            [sys.executable, os.path.join("utils", "computeMae.py"), target_file_path, predicted_file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return result.stderr

        return result.stdout.strip().split(': ')[1]
    except Exception as e:
        return e

def predict_global_mean(df):
    # Calculate the global mean ignoring nan values

    # Use stack, converts a matrix into a large column vector
    global_mean = df.stack().mean()
    return global_mean

def predict_user_mean(df):
    # Calculate the user mean ignoring nan values
    user_means = df.mean(axis=1)
    
    # Fill NaN values with the user mean
    df_filled = df.T.fillna(user_means).T
    
    # If NaN use global
    global_mean = predict_global_mean(df)
    df_filled = df_filled.fillna(global_mean)

    return df_filled

def compute_baseline(df, lambda_user=10.0, lambda_item=25.0):
    """
    Calcula:
      - media global (mu)
      - sesgo de usuario (b_u)
      - sesgo de ítem (b_i)
    con regularización tipo ridge.
    """
    # Usamos solo valores conocidos (no NaN)
    mu = df.stack().mean()

    mask = df.notna()
    user_counts = mask.sum(axis=1)   # nº ratings por usuario
    item_counts = mask.sum(axis=0)   # nº ratings por ítem

    # Sesgo de usuario: promedio de (r_ui - mu) con regularización
    centered_user = df.sub(mu)
    bu = centered_user.sum(axis=1) / (lambda_user + user_counts)

    # Sesgo de ítem: promedio de (r_ui - mu - b_u) con regularización
    tmp = df.sub(mu).sub(bu, axis=0)
    bi = tmp.sum(axis=0) / (lambda_item + item_counts)

    return mu, bu, bi


def compute_residuals(df, mu, bu, bi):
    """
    Residuales: e_ui = r_ui - (mu + b_u + b_i)
    """
    baseline_pred = pd.DataFrame(mu, index=df.index, columns=df.columns)
    baseline_pred = baseline_pred.add(bu, axis=0)
    baseline_pred = baseline_pred.add(bi, axis=1)

    residuals = df - baseline_pred
    residuals[df.isna()] = np.nan
    return residuals


def compute_item_similarity(residuals, min_common=10, shrinkage=20.0):
    """
    Similaridad entre ítems usando Pearson sobre los residuos + shrinkage.
    residuals: DataFrame (users x items)
    """
    items = residuals.columns
    sim = pd.DataFrame(0.0, index=items, columns=items)

    for i in items:
        ei = residuals[i]
        for j in items:
            if i == j:
                sim.loc[i, j] = 1.0
                continue

            ej = residuals[j]
            common = ei.notna() & ej.notna()
            n_common = common.sum()

            if n_common < min_common:
                continue

            ci = ei[common]
            cj = ej[common]

            num = (ci * cj).sum()
            denom = np.sqrt((ci ** 2).sum()) * np.sqrt((cj ** 2).sum())

            if denom == 0:
                continue

            s = num / denom
            # shrinkage por nº de usuarios en común
            s_adj = (n_common / (n_common + shrinkage)) * s
            sim.loc[i, j] = s_adj

    return sim


def predict_single_item_knn_baseline(df, residuals, sim_matrix,
                                     mu, bu, bi,
                                     user_id, item_id,
                                     k=30):
    """
    Predicción para un par (user, item) con:
      r_hat = baseline + combinación lineal de residuos vía item-kNN.
    """
    # Si el usuario o el ítem no existen, fallback a media global
    if user_id not in df.index or item_id not in df.columns:
        return float(mu)

    # Predicción baseline
    baseline_ui = mu
    if user_id in bu.index:
        baseline_ui += bu[user_id]
    if item_id in bi.index:
        baseline_ui += bi[item_id]

    # Ítems que el usuario ha valorado
    user_row = df.loc[user_id]
    rated_items = user_row.dropna().index

    # Si el usuario no tiene ratings, devolvemos baseline
    if len(rated_items) == 0:
        return float(np.clip(baseline_ui, -10, 10))

    # Similaridades entre el ítem objetivo y los ítems valorados
    sims = sim_matrix.loc[item_id, rated_items]

    # Nos quedamos con los vecinos con mayor |similaridad|
    sims = sims.replace(0, np.nan).dropna()
    if sims.empty:
        return float(np.clip(baseline_ui, -10, 10))

    sims_abs_sorted = sims.reindex(sims.abs().sort_values(ascending=False).index)
    neighbors = sims_abs_sorted.iloc[:k]

    if neighbors.empty:
        return float(np.clip(baseline_ui, -10, 10))

    neighbor_items = neighbors.index
    neighbor_sims = neighbors.values

    # Residuos del usuario en esos ítems vecinos
    user_residuals = residuals.loc[user_id, neighbor_items].values

    # Puede haber NaN en residuos, filtramos
    mask_valid = ~np.isnan(user_residuals)
    if mask_valid.sum() == 0:
        return float(np.clip(baseline_ui, -10, 10))

    neighbor_sims = neighbor_sims[mask_valid]
    user_residuals = user_residuals[mask_valid]

    denom = np.sum(np.abs(neighbor_sims))
    if denom == 0:
        return float(np.clip(baseline_ui, -10, 10))

    delta = np.sum(neighbor_sims * user_residuals) / denom
    pred = baseline_ui + delta

    # Clip al rango [-10, 10]
    pred = float(np.clip(pred, -10, 10))
    return pred


def predict_item_knn_baseline(df, target_df,
                              k=30,
                              min_common=15,
                              shrinkage_sim=20.0,
                              lambda_user=10.0,
                              lambda_item=25.0):
    """
    df: matriz User x Restaurant con NaN en los 99.
    target_df: DataFrame con columnas ['User', 'Restaurant', 'Rating'].
    Devuelve un DataFrame con mismas columnas, con 'Rating' = predicción.
    """
    # Calculate baseline (mu, b_u, b_i)
    mu, bu, bi = compute_baseline(df,
                                  lambda_user=lambda_user,
                                  lambda_item=lambda_item)

    # Calculate residuals
    residuals = compute_residuals(df, mu, bu, bi)

    # Calculate item similarity matrix
    sim_matrix = compute_item_similarity(residuals,
                                         min_common=min_common,
                                         shrinkage=shrinkage_sim)

    # Generate predictions for the target set
    preds = []
    for row in target_df.itertuples(index=False):
        user_id = row.User
        item_id = row.Restaurant
        pred = predict_single_item_knn_baseline(
            df, residuals, sim_matrix,
            mu, bu, bi,
            user_id, item_id,
            k=k
        )
        preds.append(pred)

    result = target_df.copy()
    result['Rating'] = preds
    return result

## Using machine learning recomendation 

def predict_svd(df, target_df, n_factors=50, n_epochs=30, lr=0.005, reg=0.02, use_plus_plus=False):
    """
    This trains a SVD or SVD++ model and returns the predictions for the target set.

    The parameters are: 
    - n_factors: number of latent factors
    - n_epochs: number of epochs
    - lr: learning rate
    - reg: regularization parameter
    - use_plus_plus: whether to use SVD++ or SVD
    """

    # 1. Prepare data for surprise
    
    
    if 'Rating' not in df.columns:
        train_data = df.stack().reset_index()
        train_data.columns = ['User', 'Restaurant', 'Rating']
    else:
        train_data = df.copy()
    
    train_data = train_data[train_data['Rating'].notna()] # Remove rows with NaN ratings for training
    
    # Load train dataset
    reader = Reader(rating_scale=(-10, 10))
    data = Dataset.load_from_df(train_data, reader)
    trainset = data.build_full_trainset()

    #2. Define model algorithm
    if use_plus_plus:
        algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg)
    else:
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg)

    # 3. Train model
    model = algo.fit(trainset)
    
    # 4. Generate predictions
    preds = []

    # Itreate for each row to predict
    for row in target_df.itertuples(index=False):
        user_id = row.User
        item_id = row.Restaurant
        pred = model.predict(user_id, item_id).est
        preds.append(pred)

    result = target_df.copy()
    result['Rating'] = preds
    return result
    
def svd_knn(svd_df, knn_df, target_df, alpha=0.65):
    """
    Use results of SVD and Item KNN to generate final predictions using hibrid method

    Args:
    - svd_df: DataFrame with predictions from SVD
    - knn_df: DataFrame with predictions from Item KNN
    - target_df: DataFrame with target values
    - alpha: weight of SVD predictions
    """

    # Generate predictions for the target set
    vals_knn = knn_df['Rating'].values
    vals_svd = svd_df['Rating'].values

    mixed_preds = (vals_knn * alpha) + (vals_svd * (1 - alpha))

    mixed_preds = np.clip(mixed_preds, -10, 10)
    result_df = target_df.copy()
    result_df['Rating'] = mixed_preds

    return result_df

def knn_with_zscore(df, target_df, k=30, min_common=15, shrinkage_sim=20.0, lambda_user=10.0, lambda_item=25.0):
    """
    Implements Item-KNN with Z-Score normalization.
    """

    # 1. Prepare data
    if 'Rating' not in df.columns:
        train_data = df.stack().reset_index()
        train_data.columns = ['User', 'Restaurant', 'Rating']
    else:
        train_data = df.copy()
    
    # 🔥 IMPORTANT: Filter explicitly the value 99
    train_data = train_data[train_data['Rating'] != 99] 
    train_data = train_data[train_data['Rating'].notna()]

    # Load dataset
    reader = Reader(rating_scale=(-10, 10))
    data = Dataset.load_from_df(train_data[['User', 'Restaurant', 'Rating']], reader)
    trainset = data.build_full_trainset()

    # 2. Define algorithm (The corrected part)
    # Surprise needs 'shrinkage' to go inside this dictionary
    sim_options = {
        'name': 'pearson_baseline', # The best metric for Z-Score
        'user_based': False,        # False = Item-Based (Restaurants)
        'shrinkage': shrinkage_sim      # Here is where the shrinkage is defined
    }

    # Instantiate without the lambdas
    instance = KNNWithZScore(k=k, min_k=min_common, sim_options=sim_options, verbose=False)

    # 3. Train
    instance.fit(trainset)
    
    # 4. Predict
    preds = []
    for row in target_df.itertuples(index=False):
        # .est returns the float estimation
        pred = instance.predict(row.User, row.Restaurant).est
        preds.append(pred)

    result = target_df.copy()
    result['Rating'] = preds
    return result


def knn_svd_knnz(knn_df, svd_df, knnz_df, target_df, alpha_knn=0.45, alpha_svd=0.30, alpha_knnz=0.15):
    """
    Use results of KNN, SVD, KNNZ and CoClustering to generate final predictions using hibrid method

    Args:
    - knn_df: DataFrame with predictions from KNN
    - svd_df: DataFrame with predictions from SVD
    - knnz_df: DataFrame with predictions from KNNZ
    - target_df: DataFrame with target values
    - alpha_knn: weight of KNN predictions
    - alpha_svd: weight of SVD predictions
    - alpha_knnz: weight of KNNZ predictions
    """

    if (alpha_knn + alpha_svd + alpha_knnz) != 1:
        raise ValueError("The sum of the weights must be 1")
    
    # Generate predictions for the target set
    vals_knn = knn_df['Rating'].values
    vals_svd = svd_df['Rating'].values
    vals_knnz = knnz_df['Rating'].values

    mixed_preds = (vals_knn * alpha_knn) + (vals_svd * alpha_svd) + (vals_knnz * alpha_knnz)

    mixed_preds = np.clip(mixed_preds, -10, 10)
    
    result_df = target_df.copy()
    result_df['Rating'] = mixed_preds

    return result_df
