import pandas as pd

# Step 1: Read the CSV file and parse the 'date' column as datetime
df = pd.read_csv("../Dataset/logon.csv", parse_dates=["date"])

"""# Step 2: Define the cutoff date (May 1, 2011)
cutoff = pd.Timestamp("2011-05-01")

# Step 3: Keep only rows with dates on or after the cutoff
df = df[df["date"] >= cutoff]

# Step 4: Drop the unwanted column 'id'
df.drop(columns=['id'], inplace=True)

# Step 5: Save the cleaned DataFrame back to the same CSV file
df.to_csv("logon.csv", index=False)"""

# Step 6: Display the first 10 rows of the cleaned data
print("First 10 rows:")
print(df.head(10))

# Step 7: Display the last 10 rows of the cleaned data
print("Last 10 rows:")
print(df.tail(10))

# Step 8: Display column names as a list to verify
column_list = df.columns.tolist()
print("Columns:", column_list)
