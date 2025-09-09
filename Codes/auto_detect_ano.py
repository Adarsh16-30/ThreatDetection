import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
CSV_FILE_PATH = "email.csv"
RANDOM_STATE = 42
# Split: 60% train, 20% validation (for threshold tuning), 20% test
TRAIN_SIZE_INITIAL = 0.8 # 80% for train+validation
VALIDATION_SIZE_FROM_TRAIN = 0.25 # 0.25 * 0.8 = 0.2 (20% of total for validation)
# Anomaly definition parameters (for ground truth creation for evaluation)
RARE_EXTENSION_THRESHOLD_PERCENT = 0.01
RARE_EXTENSION_MIN_COUNT = 2
SIZE_ANOMALY_PERCENTILE = 98
# Autoencoder parameters
ENCODING_DIM_RATIO = 0.25
AE_EPOCHS = 50 # Keep relatively low for faster iteration during threshold search
AE_BATCH_SIZE = 32
AE_VALIDATION_SPLIT_DURING_TRAINING = 0.1 # For AE internal validation, not for threshold tuning

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# --- Utility Functions (same as before) ---
def extract_extensions(attachment_string):
    if pd.isna(attachment_string) or not isinstance(attachment_string, str) or attachment_string == "":
        return []
    filenames = attachment_string.split(';')
    extensions = []
    for f_name in filenames:
        match = re.search(r'\.([a-zA-Z0-9]+)$', f_name.strip())
        if match:
            extensions.append(match.group(0).lower())
    return extensions

# --- 1. Load Data (same as before) ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH, parse_dates=['date'])
except FileNotFoundError:
    print(f"ERROR: File '{CSV_FILE_PATH}' not found.")
    exit()
except Exception as e:
    print(f"Error loading {CSV_FILE_PATH}: {e}")
    exit()
print(f"Data loaded successfully. Shape: {df.shape}")

# --- 2. Initial Data Cleaning & Preparation (same as before) ---
required_columns = {
    'date': 'datetime64[ns]', 'user': 'object', 'pc': 'object', 'to': 'object',
    'cc': 'object', 'bcc': 'object', 'from': 'object', 'activity': 'object',
    'size': 'int64', 'attachments': 'object'
}
for col, dtype in required_columns.items():
    if col not in df.columns:
        if dtype == 'object': df[col] = "MISSING"
        elif dtype == 'int64': df[col] = 0
        elif dtype == 'datetime64[ns]': print(f"CRITICAL: Date column '{col}' missing. Exiting."); exit()
    if col == 'date' and not pd.api.types.is_datetime64_any_dtype(df[col]): df[col] = pd.to_datetime(df[col], errors='coerce')
    if col == 'size': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    if col == 'attachments': df[col] = df[col].astype(str).fillna('')


# --- 3. Feature Engineering & Ground Truth Anomaly Definition (same as before) ---
df_eng = df.copy()
df_eng['num_attachments'] = df_eng['attachments'].apply(lambda x: len(extract_extensions(x)))
all_extensions_series = df_eng['attachments'].apply(extract_extensions)
all_extensions_list = [ext for sublist in all_extensions_series for ext in sublist]
extension_counts = pd.Series(all_extensions_list).value_counts()

rare_thresh_count_feat = max(RARE_EXTENSION_MIN_COUNT, int(len(df_eng[df_eng['num_attachments'] > 0]) * RARE_EXTENSION_THRESHOLD_PERCENT))
rare_extensions_for_feature = extension_counts[extension_counts < rare_thresh_count_feat].index.tolist()
df_eng['has_rare_attachment_feat'] = all_extensions_series.apply(
    lambda x_list: any(ext in rare_extensions_for_feature for ext in x_list)
).astype(int)

df_eng['hour_of_day'] = df_eng['date'].dt.hour.fillna(0)
df_eng['day_of_week'] = df_eng['date'].dt.dayofweek.fillna(0)

gt_rare_thresh_count = max(RARE_EXTENSION_MIN_COUNT, int(len(df_eng[df_eng['num_attachments'] > 0]) * RARE_EXTENSION_THRESHOLD_PERCENT))
gt_rare_extensions = extension_counts[extension_counts < gt_rare_thresh_count].index.tolist()
df_eng['gt_has_rare_attachment'] = all_extensions_series.apply(
    lambda x_list: any(ext in gt_rare_extensions for ext in x_list)
).astype(int)

gt_size_threshold = 0
if df_eng['size'].nunique() > 1 and not df_eng['size'].empty:
    valid_sizes = df_eng[df_eng['size'] > 0]['size']
    if not valid_sizes.empty: gt_size_threshold = np.percentile(valid_sizes, SIZE_ANOMALY_PERCENTILE)
df_eng['gt_is_large_file'] = (df_eng['size'] > gt_size_threshold).astype(int)
df_eng['is_anomaly_ground_truth'] = ((df_eng['gt_is_large_file'] == 1) | (df_eng['gt_has_rare_attachment'] == 1)).astype(int)
print("\nGround truth anomaly distribution:"); print(df_eng['is_anomaly_ground_truth'].value_counts(normalize=True))

# --- 4. Define Features for Autoencoder and Target for Evaluation (same as before) ---
numerical_features = ['size', 'num_attachments', 'hour_of_day', 'day_of_week', 'has_rare_attachment_feat']
categorical_features = ['user', 'pc', 'to', 'cc', 'bcc', 'from', 'activity']
for col in categorical_features: df_eng[col] = df_eng[col].astype(str).fillna('MISSING_VALUE')
X_features = df_eng[numerical_features + categorical_features]
y_ground_truth = df_eng['is_anomaly_ground_truth']

# --- 5. Split Data into Train, Validation (for threshold tuning), and Test ---
# First split into (train + validation) and test
X_train_val, X_test, y_train_val, y_test_eval = train_test_split(
    X_features, y_ground_truth,
    test_size=(1-TRAIN_SIZE_INITIAL), # This will be the test set size
    random_state=RANDOM_STATE,
    stratify=y_ground_truth if y_ground_truth.nunique() > 1 else None
)
# Then split (train + validation) into actual train and validation
X_train, X_val_thresh, y_train, y_val_thresh_eval = train_test_split(
    X_train_val, y_train_val,
    test_size=VALIDATION_SIZE_FROM_TRAIN,
    random_state=RANDOM_STATE,
    stratify=y_train_val if y_train_val.nunique() > 1 else None
)
print(f"\nShapes: X_train: {X_train.shape}, X_val_thresh: {X_val_thresh.shape}, X_test: {X_test.shape}")

# --- 6. Preprocessing Pipeline (Scaling and One-Hot Encoding) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_thresh_processed = preprocessor.transform(X_val_thresh) # For threshold tuning
X_test_processed = preprocessor.transform(X_test)           # For final testing
print(f"Processed shapes: X_train: {X_train_processed.shape}, X_val_thresh: {X_val_thresh_processed.shape}, X_test: {X_test_processed.shape}")

if X_train_processed.shape[1] == 0: print("ERROR: Processed training data has no features."); exit()

# --- 7. Build Autoencoder Model (same architecture as before) ---
input_dim = X_train_processed.shape[1]
encoding_dim_calculated = int(input_dim * ENCODING_DIM_RATIO)
encoding_dim = max(2, encoding_dim_calculated if encoding_dim_calculated < input_dim / 2 else int(input_dim / 4))
print(f"\nAutoencoder Architecture: Input Dim: {input_dim}, Encoding Dim (Bottleneck): {encoding_dim}")
autoencoder = Sequential([
    Dense(int(input_dim * 0.75) if int(input_dim * 0.75) > encoding_dim else encoding_dim + 2 , activation='relu', input_shape=(input_dim,)), Dropout(0.1),
    Dense(int(input_dim * 0.5) if int(input_dim * 0.5) > encoding_dim else encoding_dim +1, activation='relu'), Dropout(0.1),
    Dense(encoding_dim, activation='relu'),
    Dense(int(input_dim * 0.5) if int(input_dim * 0.5) > encoding_dim else encoding_dim +1, activation='relu'), Dropout(0.1),
    Dense(int(input_dim * 0.75) if int(input_dim * 0.75) > encoding_dim else encoding_dim +2, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# --- 8. Train Autoencoder ---
print("\nTraining Autoencoder...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
history = autoencoder.fit(
    X_train_processed, X_train_processed,
    epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, shuffle=True,
    validation_split=AE_VALIDATION_SPLIT_DURING_TRAINING, # AE's own validation split for early stopping
    callbacks=[early_stopping], verbose=1
)

# --- 9. Determine Optimal Anomaly Threshold using Validation Set ---
print("\nCalculating reconstruction errors on validation set for threshold tuning...")
val_reconstructions = autoencoder.predict(X_val_thresh_processed)
val_mse = np.mean(np.power(X_val_thresh_processed - val_reconstructions, 2), axis=1)

best_threshold = 0
best_accuracy = -1 # Initialize with a value that will be beaten
# Or if you want to optimize F1 for anomaly class:
# best_f1_anomaly = -1

# Iterate over a range of potential thresholds (e.g., percentiles of validation MSE)
# Or a fixed number of steps between min and max val_mse
threshold_candidates = np.linspace(np.min(val_mse), np.max(val_mse), num=100) # 100 candidates
if len(np.unique(val_mse)) < 2: # Handle cases with no variance in MSE
    print("Warning: Little to no variance in validation MSEs. Using median as threshold.")
    threshold_candidates = [np.median(val_mse)] if len(val_mse) > 0 else [0]


print(f"Searching for best threshold among {len(threshold_candidates)} candidates...")
for current_thresh in threshold_candidates:
    y_pred_val = (val_mse > current_thresh).astype(int)
    current_accuracy = accuracy_score(y_val_thresh_eval, y_pred_val)
    # If optimizing F1 for anomaly class:
    # current_f1_anomaly = f1_score(y_val_thresh_eval, y_pred_val, pos_label=1, zero_division=0)

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_threshold = current_thresh
    # If optimizing F1:
    # if current_f1_anomaly > best_f1_anomaly:
    #     best_f1_anomaly = current_f1_anomaly
    #     best_threshold = current_thresh
    #     best_accuracy_at_best_f1 = current_accuracy # Store accuracy too

print(f"Best threshold found on validation set: {best_threshold:.8f}")
print(f"This threshold achieved an accuracy of: {best_accuracy:.4f} on the validation set.")
# If optimizing F1:
# print(f"This threshold achieved an F1-score (anomaly) of: {best_f1_anomaly:.4f} on the validation set.")
# print(f"Accuracy at best F1 threshold: {best_accuracy_at_best_f1:.4f}")


# --- 10. Detect Anomalies on Test Data using the chosen best_threshold ---
print("\nPredicting on test data using the determined best threshold...")
test_reconstructions = autoencoder.predict(X_test_processed)
test_mse = np.mean(np.power(X_test_processed - test_reconstructions, 2), axis=1)

ae_predicted_anomalies_test = (test_mse > best_threshold).astype(int)

# --- 11. Evaluate Autoencoder Performance on Test Set ---
print("\n--- Autoencoder Anomaly Detection Evaluation on TEST SET ---")

# Default values in case of issues
accuracy_ae_test = None 
final_accuracy_statement = "Final Model Accuracy on Test Set: Not calculable due to issues."

if y_test_eval.nunique() < 2:
    print("WARNING: Ground truth test data has only one class. Full evaluation metrics might not be meaningful.")
    if len(y_test_eval) == len(ae_predicted_anomalies_test) and len(y_test_eval) > 0 :
        accuracy_ae_test = accuracy_score(y_test_eval, ae_predicted_anomalies_test)
        # The specific print statement you requested:
        final_accuracy_statement = f"Final Model Accuracy on Test Set: {accuracy_ae_test:.4f}"
        print(f"\nOverall Accuracy on TEST SET: {accuracy_ae_test:.4f}")
        print("\nConfusion Matrix (may be limited):")
        print(confusion_matrix(y_test_eval, ae_predicted_anomalies_test))
    else:
        print("Could not calculate accuracy due to mismatched lengths or empty test set.")
else:
    accuracy_ae_test = accuracy_score(y_test_eval, ae_predicted_anomalies_test)
    cm_ae_test = confusion_matrix(y_test_eval, ae_predicted_anomalies_test)
    cr_ae_test = classification_report(y_test_eval, ae_predicted_anomalies_test, target_names=['Not Anomaly (0)', 'Anomaly (1)'], zero_division=0)
    
    # The specific print statement you requested:
    final_accuracy_statement = f"Final Model Accuracy on Test Set: {accuracy_ae_test:.4f}"
    print(f"\nOverall Accuracy on TEST SET: {accuracy_ae_test:.4f}") 
    
    try:
        roc_auc_ae_test = roc_auc_score(y_test_eval, test_mse) # Use test_mse for ROC AUC
        print(f"ROC AUC Score (using MSE) on TEST SET: {roc_auc_ae_test:.4f}")
    except ValueError:
        print("Could not calculate ROC AUC score on test set (likely one class in y_test_eval).")

    print("\nConfusion Matrix on TEST SET:")
    print(cm_ae_test)
    print("\nClassification Report on TEST SET:")
    print(cr_ae_test)

# --- This is the new, very prominent print statement at the end ---
print("\n" + "="*50)
print(final_accuracy_statement)
print("="*50 + "\n")

print("\nScript finished.")