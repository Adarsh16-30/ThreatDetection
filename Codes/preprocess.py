import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from functools import reduce

# ───────────────────────────────────────────────────────────────────────────────
# 1. Load all CSV files
logon_df = pd.read_csv('../Dataset/logon.csv')
email_df = pd.read_csv('../Dataset/email.csv')
file_df = pd.read_csv('../Dataset/file.csv')

# ───────────────────────────────────────────────────────────────────────────────
# 2. Process Logon Data
logon_df['success'] = logon_df['activity'].str.contains('Logon', case=False).astype(int)
logon_df['failure'] = logon_df['activity'].str.contains('LogonFail', case=False).astype(int)

logon_agg = logon_df.groupby(['user', 'date']).agg(
    logon_events=('activity', 'count'),
    unique_pcs=('pc', pd.Series.nunique),
    successful_logons=('success', 'sum'),
    failed_logons=('failure', 'sum')
).reset_index()

# ───────────────────────────────────────────────────────────────────────────────
# 3. Process Email Data
# Fill missing fields
email_df.fillna('', inplace=True)

# Function to sum attachment sizes from the string
def extract_attachment_size(s):
    sizes = re.findall(r'\((\d+)\)', s)
    return sum(map(int, sizes)) if sizes else 0

# Apply size extraction
email_df['attachments'] = email_df['attachments'].apply(extract_attachment_size)

# Compute number of unique recipients
email_df['n_recipients'] = email_df[['to', 'cc', 'bcc']].apply(
    lambda x: len(set(';'.join(x).split(';')) - {''}), axis=1
)

# Aggregate by user and date
email_agg = email_df.groupby(['user', 'date']).agg(
    emails_sent=('activity', 'count'),
    avg_email_size=('size', 'mean'),
    total_attachments=('attachments', 'sum'),
    total_recipients=('n_recipients', 'sum')
).reset_index()

# ───────────────────────────────────────────────────────────────────────────────
# 4. Process File Data
file_agg = file_df.groupby(['user', 'date']).agg(
    file_events=('activity', 'count'),
    to_usb=('to_removable_media', 'sum'),
    from_usb=('from_removable_media', 'sum')
).reset_index()

# ───────────────────────────────────────────────────────────────────────────────
# 5. Merge All Datasets
dfs = [logon_agg, email_agg, file_agg]
merged_df = reduce(lambda left, right: pd.merge(left, right, on=['user', 'date'], how='outer'), dfs)
merged_df.fillna(0, inplace=True)

# ───────────────────────────────────────────────────────────────────────────────
# 6. Convert and sort by date
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df = merged_df.sort_values(by='date')

# ───────────────────────────────────────────────────────────────────────────────
# 7. Prepare features and metadata
meta = merged_df[['user', 'date']]
X = merged_df.drop(columns=['user', 'date'])

# Ensure all data is numeric and fill any conversion issues
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ───────────────────────────────────────────────────────────────────────────────
# 8. Train/Test split (by date)
train_idx = merged_df['date'] <= '2011-05-10'
test_idx = ~train_idx

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]

# ───────────────────────────────────────────────────────────────────────────────
# 9. Save outputs
# a. Full processed dataset
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df.to_csv("../Dataset/processed_data.csv", index=False)

# b. Train/test sets
np.savetxt("../Dataset/X_train.csv", X_train, delimiter=",")
np.savetxt("../Dataset/X_test.csv", X_test, delimiter=",")

# c. Optional: Save metadata
meta[train_idx].to_csv("../Dataset/train_metadata.csv", index=False)
meta[test_idx].to_csv("../Dataset/test_metadata.csv", index=False)

print("Processing complete. All files saved.")
