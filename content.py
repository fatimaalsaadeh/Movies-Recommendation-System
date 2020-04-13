import data 
import pandas as pd

df_train, df_test, df_links = data.get_train_test_data()

print(df_train.shape)

print(df_train.head())




df_links = pd.read_csv('ml-latest-small/links.csv')

print(df_links.shape)
print(df_links.head())