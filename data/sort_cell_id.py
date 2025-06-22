import pandas as pd
#1


df = pd.read_csv('data/HumanLiverCancerPatient1_cell_metadata.csv')


sorted_df = df.sort_values(df.columns[0])


sorted_df.to_csv('data/sorted_id_file.csv', index=False)
