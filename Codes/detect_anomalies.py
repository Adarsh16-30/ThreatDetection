# detect_anomalies.py
# This script expects an 'email.csv' file in the same directory.
# The 'email.csv' should be preprocessed to include columns like:
# date, user, pc, to, cc, bcc, from, activity, size, attachments

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
import re # For regex to extract file extensions

# --- Feature Engineering ---
def extract_extensions(attachment_string):
    if pd.isna(attachment_string) or attachment_string == "":
        return []
    filenames = str(attachment_string).split(';')
    extensions = []
    for f_name in filenames:
        match = re.search(r'\.([a-zA-Z0-9]+)$', f_name.strip())
        if match:
            extensions.append(match.group(0).lower())
    return extensions

def engineer_features(df, fit_mode=True, learned_rare_extensions=None):
    df_eng = df.copy()
    df_eng['attachments'] = df_eng['attachments'].astype(str).fillna('')
    df_eng['num_attachments'] = df_eng['attachments'].apply(lambda x: len(extract_extensions(x)))

    all_extensions_series = df_eng['attachments'].apply(extract_extensions)
    all_extensions_list = [ext for sublist in all_extensions_series for ext in sublist]

    if fit_mode:
        extension_counts = pd.Series(all_extensions_list).value_counts()
        rare_threshold_count = max(2, int(len(df_eng[df_eng['num_attachments'] > 0]) * 0.01)) # Adjust threshold
        print(f"Calculated rare extension threshold count: {rare_threshold_count}")
        rare_extensions_list = extension_counts[extension_counts < rare_threshold_count].index.tolist()
        print(f"Identified rare extensions (first 10): {rare_extensions_list[:10]}")
    else:
        if learned_rare_extensions is None:
            raise ValueError("learned_rare_extensions must be provided when fit_mode is False")
        rare_extensions_list = learned_rare_extensions

    df_eng['has_rare_attachment'] = all_extensions_series.apply(
        lambda x_list: any(ext in rare_extensions_list for ext in x_list)
    )

    if not pd.api.types.is_datetime64_any_dtype(df_eng['date']):
        df_eng['date'] = pd.to_datetime(df_eng['date'], errors='coerce') # Add errors='coerce' for robustness
    df_eng['hour_of_day'] = df_eng['date'].dt.hour
    df_eng['day_of_week'] = df_eng['date'].dt.dayofweek

    categorical_cols = ['user', 'pc', 'to', 'cc', 'bcc', 'from', 'activity']
    for col in categorical_cols:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].astype(str).fillna('MISSING_VALUE').astype('category')
        else:
            print(f"Warning: Categorical column '{col}' not found in DataFrame during feature engineering.")

    if fit_mode:
        return df_eng, rare_extensions_list
    else:
        return df_eng

# --- Label Creation ---
def create_labels(df, fit_mode=True, learned_size_threshold=None):
    df_labeled = df.copy()

    if 'size' not in df_labeled.columns:
        print("Error: 'size' column not found. Cannot define large file anomalies.")
        df_labeled['is_large_file'] = False
        size_threshold = 0
    elif not pd.api.types.is_numeric_dtype(df_labeled['size']):
        print("Warning: 'size' column is not numeric. Attempting conversion.")
        df_labeled['size'] = pd.to_numeric(df_labeled['size'], errors='coerce').fillna(0)

    if fit_mode:
        if 'size' in df_labeled.columns and df_labeled['size'].nunique() > 1 and not df_labeled['size'].empty:
            valid_sizes = df_labeled[df_labeled['size'] > 0]['size']
            if not valid_sizes.empty:
                size_threshold = np.percentile(valid_sizes, 98) # E.g., 98th percentile, adjust!
            else:
                size_threshold = 100_000_000 # Default if no positive sizes
        else:
            size_threshold = 100_000_000
        print(f"Calculated size threshold for anomaly: {size_threshold:,.0f}")
    else:
        if learned_size_threshold is None:
            raise ValueError("learned_size_threshold must be provided when fit_mode is False")
        size_threshold = learned_size_threshold

    if 'size' in df_labeled.columns:
        df_labeled['is_large_file'] = df_labeled['size'] > size_threshold
    else:
        df_labeled['is_large_file'] = False

    df_labeled['is_anomaly'] = ((df_labeled['is_large_file']) | (df_labeled['has_rare_attachment'])).astype(int)

    if fit_mode:
        return df_labeled, size_threshold
    else:
        return df_labeled

# --- Main Workflow ---
if __name__ == "__main__":
    preprocessed_csv_path = "email.csv"
    try:
        raw_df = pd.read_csv(preprocessed_csv_path, parse_dates=['date'])
        print(f"Successfully loaded preprocessed data from: ./{preprocessed_csv_path}")
    except FileNotFoundError:
        print(f"ERROR: Preprocessed file './{preprocessed_csv_path}' not found.")
        print("Please make sure 'email.csv' is in the same directory as this script.")
        exit()
    except Exception as e:
        print(f"Error loading {preprocessed_csv_path}: {e}")
        exit()

    print("\nLoaded preprocessed data sample:")
    print(raw_df.head())
    print(f"\nTotal rows: {len(raw_df)}")
    print(f"\nData types:\n{raw_df.dtypes}")

    expected_cols = ['date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'activity', 'size', 'attachments']
    actual_cols = raw_df.columns.tolist()

    print("\nVerifying columns...")
    missing_critical_columns = False
    for col in expected_cols:
        if col not in actual_cols:
            print(f"WARNING: Expected column '{col}' not found in the loaded CSV.")
            if col in ['size', 'attachments', 'date']:
                print(f"   CRITICAL: Column '{col}' is essential. Results may be inaccurate or script may fail.")
                missing_critical_columns = True
        elif raw_df[col].isnull().all():
             print(f"WARNING: Column '{col}' is present but contains all Null/NaN values.")
             if col in ['size', 'attachments', 'date']:
                missing_critical_columns = True
    
    if missing_critical_columns:
        print("CRITICAL WARNING: One or more essential columns are missing or all-NaN. Review your email.csv")
        # Decide if you want to exit or try to proceed with dummy values
        # For now, let's create dummies for size/attachments if missing, date is harder
        if 'size' not in raw_df.columns: raw_df['size'] = 0
        if 'attachments' not in raw_df.columns: raw_df['attachments'] = ""
        if 'date' not in raw_df.columns:
            print("ERROR: 'date' column is missing. Cannot proceed with time-based features.")
            exit()


    if 'size' in raw_df.columns:
        raw_df['size'] = pd.to_numeric(raw_df['size'], errors='coerce').fillna(0)
    if 'attachments' in raw_df.columns:
         raw_df['attachments'] = raw_df['attachments'].astype(str).fillna('')


    stratify_on = None
    if 'activity' in raw_df.columns and raw_df['activity'].nunique() > 1:
        stratify_on = raw_df['activity']
    elif len(raw_df) > 0:
        print("Warning: 'activity' column not suitable for stratification. Proceeding without or consider another column.")

    if len(raw_df) < 20: # Increased minimum for meaningful split
        print("Error: Not enough data (less than 20 rows) to split into training and testing sets. Exiting.")
        exit()

    try:
        train_df, test_df = train_test_split(
            raw_df,
            test_size=0.3,
            random_state=42,
            stratify=stratify_on if stratify_on is not None and stratify_on.nunique() >=2 and len(raw_df)*0.3 >= stratify_on.nunique() else None
        )
    except ValueError as e:
        print(f"Could not stratify due to: {e}. Splitting without stratification.")
        train_df, test_df = train_test_split(raw_df, test_size=0.3, random_state=42)


    train_df_eng, learned_rare_extensions = engineer_features(train_df, fit_mode=True)
    train_df_labeled, learned_size_threshold = create_labels(train_df_eng, fit_mode=True)

    print(f"\nAnomalies defined in training data: {train_df_labeled['is_anomaly'].sum()} / {len(train_df_labeled)}")
    if train_df_labeled['is_anomaly'].sum() > 0:
        print(f"Breakdown of training anomalies:\n{train_df_labeled[train_df_labeled['is_anomaly']==1][['is_large_file', 'has_rare_attachment']].sum()}")
    else:
        print("No anomalies were defined in the training data based on current thresholds.")


    test_df_eng = engineer_features(test_df, fit_mode=False, learned_rare_extensions=learned_rare_extensions)
    test_df_labeled = create_labels(test_df_eng, fit_mode=False, learned_size_threshold=learned_size_threshold)

    print(f"\nAnomalies defined in test data: {test_df_labeled['is_anomaly'].sum()} / {len(test_df_labeled)}")

    base_features = ['size', 'num_attachments', 'hour_of_day', 'day_of_week']
    categorical_feature_names = ['user', 'pc', 'to', 'cc', 'bcc', 'from', 'activity']
    
    features_to_use = []
    for f in base_features + categorical_feature_names:
        if f in train_df_labeled.columns:
            features_to_use.append(f)
        else:
            print(f"Warning: Feature '{f}' not found in training data. It will be excluded.")

    if not features_to_use:
        print("ERROR: No features available for training. Exiting.")
        exit()
    
    print(f"\nUsing features for model: {features_to_use}")

    X_train = train_df_labeled[features_to_use]
    y_train = train_df_labeled['is_anomaly']
    X_test = test_df_labeled[features_to_use]
    y_test = test_df_labeled['is_anomaly']

    if y_train.empty:
        print("ERROR: Training target 'is_anomaly' is empty. Cannot train model.")
        exit()
    if y_train.nunique() < 2:
        print("WARNING: The training target 'is_anomaly' has only one unique value.")
        print(f"         Unique values in y_train: {y_train.unique()}. Model predictions will be trivial.")
        
    counts = np.bincount(y_train)
    scale_pos_weight_val = 1.0
    if len(counts) > 1 and counts[1] > 0 and counts[0] > 0:
        scale_pos_weight_val = counts[0] / counts[1]
    elif len(counts) == 1 or (len(counts) > 1 and counts[1] == 0):
        print("Warning: Only one class or no positive samples in y_train. scale_pos_weight set to 1.")
    print(f"\nCalculated scale_pos_weight: {scale_pos_weight_val:.2f}")

    lgb_clf = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', n_estimators=100,
        learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        scale_pos_weight=scale_pos_weight_val
    )

    final_categorical_features = [col for col in X_train.select_dtypes(include='category').columns.tolist() if col in features_to_use]
    print(f"\nTraining LightGBM model with categorical features: {final_categorical_features}")
    
    try:
        lgb_clf.fit(X_train, y_train, categorical_feature=final_categorical_features if final_categorical_features else 'auto')
    except Exception as e:
        print(f"Error during LightGBM training: {e}\nX_train dtypes:\n{X_train.dtypes}")
        exit()

    if y_test.empty or X_test.empty:
        print("Warning: Test data or target is empty. Skipping evaluation on test set.")
    else:
        y_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        if y_test.nunique() < 2:
            print("Warning: The test target 'is_anomaly' has only one unique value. Classification report might be limited.")
            accuracy = accuracy_score(y_test, y_pred)
            print("\n--- Model Evaluation (Limited due to single class in y_test) ---")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nConfusion Matrix:"); print(confusion_matrix(y_test, y_pred))
            try:
                print("\nClassification Report:"); print(classification_report(y_test, y_pred, target_names=['Not Anomaly (0)', 'Anomaly (1)'], zero_division=0))
            except ValueError as e_report: print(f"Could not generate classification report: {e_report}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, target_names=['Not Anomaly (0)', 'Anomaly (1)'], zero_division=0)
            print("\n--- Model Evaluation ---")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nConfusion Matrix:"); print(conf_matrix)
            print("\nClassification Report:"); print(class_report)

    print("\nFeature Importances:")
    try:
        importances = pd.Series(lgb_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print(importances)
    except Exception as e:
        print(f"Could not retrieve feature importances: {e}")

    print("\nAnomaly detection script finished.")