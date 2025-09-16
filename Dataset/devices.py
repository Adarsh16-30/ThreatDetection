import pandas as pd

# Step 1: Read the CSV and parse 'date' column as datetime
df = pd.read_csv("device.csv", parse_dates=["date"])

# Step 2: Drop the unwanted columns ('id' and 'file_tree')
df.drop(columns=['id', 'file_tree'], inplace=True)

# Step 3: Define the cutoff date (May 1, 2011)
cutoff = pd.Timestamp("2011-05-01")

# Step 4: Keep only rows on or after the cutoff date
df = df[df["date"] >= cutoff]

# Step 5: Overwrite the original CSV file with cleaned data
df.to_csv("device.csv", index=False)

# Step 6: Reload the cleaned CSV file (optional, just to verify)
df = pd.read_csv("device.csv")

# Step 7: Display number of rows
print("Number of rows:", df.shape[0])

# Step 8: Display column names as a list to verify
column_list = df.columns.tolist()
print("Columns:", column_list)

# Step 9: Show the first 10 rows to verify
print("First 10 rows:")
print(df.head(10))

# Step 10: Show the last 10 rows to verify
print("Last 10 rows:")
print(df.tail(10))
