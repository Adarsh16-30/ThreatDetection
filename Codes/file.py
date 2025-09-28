import pandas as pd

# Step 1: Read the CSV file and parse the 'date' column as datetime
df = pd.read_csv("../Dataset/file.csv", parse_dates=["date"])

"""# Step 2: Define the cutoff date (May 1, 2011)
cutoff = pd.Timestamp("2011-05-01")

# Step 3: Filter rows to keep only those on or after May 1, 2011
df = df[df["date"] >= cutoff]

# Step 4: Drop unwanted columns ('id', 'content', and 'filename')
df.drop(columns=['id', 'content', 'filename'], inplace=True)

# Step 5: Overwrite the original CSV file with the cleaned data
df.to_csv("file.csv", index=False)
"""
# Step 6: Display the remaining column names
column_list = df.columns.tolist()
print("Columns:", column_list)

# Step 7: Display the first 10 rows
print("First 10 rows:")
print(df.head(10))

# Step 8: Display the last 10 rows
print("Last 10 rows:")
print(df.tail(10))
