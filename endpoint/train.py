import os
import zipfile
import glob
import pandas as pd
import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
M = 3  # Window size for rolling features and lags. Adjust this to your original value!
ZIP_FILE = 'dataset_rework.zip'
OUTPUT_DIR = 'dataset_flattened'

def unpack_dataset(zip_path, output_dir):
    """Extracts, flattens, and filters the dataset from a zip file."""
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            original_path = file_info.filename
            
            # Skip directories and macOS metadata
            if file_info.is_dir() or original_path.endswith('/'): 
                continue
            if '__MACOSX' in original_path: 
                continue
                
            # Flatten the filename
            flat_filename = original_path.replace('/', '_').replace('\\', '_')
            dest_path = os.path.join(output_dir, flat_filename)
            
            # Skip if already exists
            if os.path.exists(dest_path):
                continue
                
            print(f"Extracting: {flat_filename}")
            with zip_ref.open(file_info) as source_file, open(dest_path, 'wb') as target_file:
                target_file.write(source_file.read())

def prepare_data_and_extract_chunks(df):
    """Filters out chunks that are too small to calculate rolling features."""
    # Reconstructed this based on your snippet!
    chunk_sizes = df.groupby('chunk_full_id').size()
    valid_chunks = chunk_sizes[chunk_sizes > M + 2].index
    return df[df['chunk_full_id'].isin(valid_chunks)].copy()

def create_features(df, scaler):  
    """Generates scaled, lagged, and rolling features."""
    df = df.copy()
    
    df['rd_value_scaled'] = scaler.fit_transform(df['rd_value'].values.reshape(-1, 1)).flatten()
    
    feature_cols = ['rd_value_scaled']

    for i in range(1, M + 1):
        lag_col = f"rd_scaled_lag_{i}"
        df[lag_col] = df.groupby('chunk_full_id')['rd_value_scaled'].shift(i)
        feature_cols.append(lag_col)

    df[f"rd_rolling_mean_{M}"] = df.groupby('chunk_full_id')['rd_value_scaled'].transform(lambda x: x.rolling(M).mean())
    df[f"rd_rolling_std_{M}"] = df.groupby('chunk_full_id')['rd_value_scaled'].transform(lambda x: x.rolling(M).std())
    feature_cols.extend([f"rd_rolling_mean_{M}", f"rd_rolling_std_{M}"])

    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols

def main():

    scaler = StandardScaler()
    # 1. Unpack the dataset
    if os.path.exists(ZIP_FILE):
        print(f"Checking/unpacking dataset from {ZIP_FILE}...")
        unpack_dataset(ZIP_FILE, OUTPUT_DIR)
    else:
        print(f"Warning: {ZIP_FILE} not found. Ensure it is in the same directory.")

    # 2. Read the files
    files = glob.glob(os.path.join(OUTPUT_DIR, '*.csv'))
    
    if not files:
        print("No CSV files found to train on! Exiting.")
        return
        
    print(f"Loading {len(files)} CSV files into Pandas...")
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f)
        chunk_id = os.path.basename(f).replace('.csv', '')
        temp_df['chunk_full_id'] = chunk_id
        df_list.append(temp_df)
        
    df_raw = pd.concat(df_list, ignore_index=True)

    # 3. Process Data & Create Features
    print("Preparing data and creating features...")
    df_chunks = prepare_data_and_extract_chunks(df_raw)
    df_feat, feature_names = create_features(df_chunks, scaler)  

    # 4. Train/Test Split
    df_feat = df_feat.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df_feat) * 0.8)
    train_data, test_data = df_feat.iloc[:split_idx], df_feat.iloc[split_idx:]

    X_train, y_train = train_data[feature_names], train_data['signal_barrier']
    X_test, y_test   = test_data[feature_names], test_data['signal_barrier']

    # 5. Model Training
    print("Training HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    model.fit(X_train, y_train)

    # 6. Evaluation
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

    # 7. Save Model
    training_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    joblib.dump({
        'model': model,
        'features': feature_names,
        'M': M,
        'trained_at': training_time, 
        'scaler': scaler          
    }, 'model_weights.pkl')
    print(f"Веса сохранены. Время генерации: {training_time}")


if __name__ == "__main__":
    main()
