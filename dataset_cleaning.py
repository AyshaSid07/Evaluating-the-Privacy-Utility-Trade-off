import pandas as pd

#1.Load the dataset
df = pd.read_csv("data/mendeley_data.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())  # check missing per column

#2.Handle missing values 
df.ffill(inplace=True)  #fill any missing values in the dataset using a forward-fill method (as per paper)
#df.bfill(inplace=True)

print("After filling:")
print(df.isnull().sum())

#3.Normalize Columns - Normalize numerical attributes using Min–Max scaling


