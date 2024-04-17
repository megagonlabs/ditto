import os
import pandas as pd
import csv

data_root = os.getcwd() + '/data/er_magellan/Dirty/'

for folder in os.listdir(data_root):
    subdata_path = data_root + folder
    
    df1 = pd.read_csv(os.path.join(subdata_path,'train.csv'))
    df2 = pd.read_csv(os.path.join(subdata_path,'valid.csv'))
    df3 = pd.read_csv(os.path.join(subdata_path,'test.csv'))

    df1 = df1[df1['label'] == 1]
    df2 = df2[df2['label'] == 1]
    df3 = df3[df3['label'] == 1]

    # Concatenate all dataframes
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)

    # Keep only 'ltable_id' and 'rtable_id' columns
    combined_df = combined_df[['ltable_id', 'rtable_id']]

    # Write the combined dataframe to a new CSV file
    combined_df.to_csv(subdata_path+'/matches.csv', index=False)


