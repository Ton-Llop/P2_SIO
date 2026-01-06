# Optimizers algorithms for diferent recomendation systems 
import pandas as pd
import os
from utils import loadData
from src import calc
from src import config

def optimize_item_knn(k = [5,10,15,20,30,40,50], min_common = [15], shrinkage_sim = [10], lambda_user = [15], lambda_item = [25], threshold = 3):

    best_mae = float('inf')
    best_params = {}


    #Load data
    df = loadData.load_dataset(config.DATA_PATH)
    target_df = loadData.load_target(config.TARGET_PATH)
    thresholdForBreak = 0
    for k_val in k:
        for min_common_val in min_common:
            for shrinkage_sim_val in shrinkage_sim:
                for lambda_user_val in lambda_user:
                    for lambda_item_val in lambda_item:
                        params = {
                            "k": k_val,
                            "min_common": min_common_val,
                            "shrinkage_sim": shrinkage_sim_val,
                            "lambda_user": lambda_user_val,
                            "lambda_item": lambda_item_val,
                        }
                        
                        preds_df = calc.predict_item_knn_baseline(
                            df,
                            target_df,
                            **params
                        )
                        
                        loadData.save_results(preds_df, config.OUTPUT_PATH_ITEM_KNN_BASELINE)
                        mae = float(calc.computeMae(config.TARGET_PATH, config.OUTPUT_PATH_ITEM_KNN_BASELINE))
                        if mae < best_mae:
                            best_mae = mae
                            best_params = {
                                "k": k_val,
                                "min_common": min_common_val,
                                "shrinkage_sim": shrinkage_sim_val,
                                "lambda_user": lambda_user_val,
                                "lambda_item": lambda_item_val,
                            }
                        else: # If the MAE is not better than the best MAE more than threshold times, break
                            if thresholdForBreak >= threshold: 
                                break
                            else:
                                thresholdForBreak += 1

    return [best_mae, best_params]

def optimize_weights_knn_svd_zscore():
    files = {
        "knn": config.OUTPUT_PATH_ITEM_KNN_BASELINE,
        "svd": config.OUTPUT_PATH_SVD,
        "knnz": config.OUTPUT_PATH_KNN_WITH_ZSCORE,
    }

    target_df = loadData.load_target(config.TARGET_PATH)
    
    dfs = {}
    for name, path in files.items():
        try:
            dfs[name] = pd.read_csv(path, sep=';').iloc[:, 2].values
            print(f"   ✅ Cargado {name}")
        except Exception as e:
            print(f"   ❌ Error cargando {path}: {e}")
            return
    best_mae = float('inf')
    best_weights = None
    temp_path = "data/output/temp_opt.csv"
    print("   Probando combinaciones (esto puede tardar 1-2 mins)...")
    
    # Rango de pesos (0 a 100 en pasos de 5)
    steps = range(0, 105, 5)
    
    for w_knn in steps:
        for w_svd in steps:
            for w_zsc in steps:
                if (w_knn + w_svd + w_zsc) == 100:
                    # Normalizamos a 0.0 - 1.0
                    wk = w_knn / 100
                    ws = w_svd / 100
                    wz = w_zsc / 100
                    
                    # Calcular mezcla
                    mixed_vals = (dfs['knn'] * wk) + (dfs['svd'] * ws) + (dfs['knnz'] * wz)
                
                    # Usamos uno de plantilla para el formato
                    plantilla = pd.read_csv(config.OUTPUT_PATH_ITEM_KNN_BASELINE, sep=';')
                    plantilla.iloc[:, 2] = mixed_vals
                    plantilla.to_csv(temp_path, sep=';', index=False, float_format='%.3f')
                    
                    # Calcular MAE Real
                    try:
                        mae_str = calc.computeMae(config.TARGET_PATH, temp_path)
                        mae = float(mae_str)
                        
                        # Si encontramos un nuevo récord
                        if mae < best_mae:
                            best_mae = mae
                            best_weights = (wk, ws, wz)
                            print(f"   🌟 NEW RECORD: {mae:.4f} | Pesos: kNN={wk}, SVD={ws}, ZScore={wz}")
                    except:
                        continue

    print("\n" + "="*40)
    print(f"🏆 MEJOR MAE ENCONTRADO: {best_mae}")
    print(f"⚖️ MEJORES PESOS: kNN={best_weights[0]}, SVD={best_weights[1]}, ZScore={best_weights[2]}")
    print("="*40)
    os.remove(temp_path)

def optimize_weights_knn_svd():
    files = {
        "knn": config.OUTPUT_PATH_ITEM_KNN_BASELINE,
        "svd": config.OUTPUT_PATH_SVD,
    }

    target_df = loadData.load_target(config.TARGET_PATH)
    
    dfs = {}
    for name, path in files.items():
        try:
            dfs[name] = pd.read_csv(path, sep=';').iloc[:, 2].values
            print(f"   ✅ Cargado {name}")
        except Exception as e:
            print(f"   ❌ Error cargando {path}: {e}")
            return
    best_mae = float('inf')
    best_weights = None
    temp_path = "data/output/temp_opt.csv"
    print("   Probando combinaciones (esto puede tardar 1-2 mins)...")
    
    # Rango de pesos (0 a 100 en pasos de 5)
    steps = range(0, 105, 5)
    
    for w_knn in steps:
        for w_svd in steps:
            if (w_knn + w_svd) == 100:
                # Normalizamos a 0.0 - 1.0
                wk = w_knn / 100
                ws = w_svd / 100
                
                # Calcular mezcla
                mixed_vals = (dfs['knn'] * wk) + (dfs['svd'] * ws)
                
                # Usamos uno de plantilla para el formato
                plantilla = pd.read_csv(config.OUTPUT_PATH_ITEM_KNN_BASELINE, sep=';')
                plantilla.iloc[:, 2] = mixed_vals
                plantilla.to_csv(temp_path, sep=';', index=False, float_format='%.3f')
                    
                # Calcular MAE Real
                try:
                    mae_str = calc.computeMae(config.TARGET_PATH, temp_path)
                    mae = float(mae_str)
                        
                    # Si encontramos un nuevo récord
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = (wk, ws)
                        print(f"   🌟 NEW RECORD: {mae:.4f} | Pesos: kNN={wk}, SVD={ws}")
                except:
                    continue

    print("\n" + "="*40)
    print(f"🏆 MEJOR MAE ENCONTRADO: {best_mae}")
    print(f"⚖️ MEJORES PESOS: kNN={best_weights[0]}, SVD={best_weights[1]}")
    print("="*40)
    os.remove(temp_path)