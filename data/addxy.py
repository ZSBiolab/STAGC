import pandas as pd

sorted_df = pd.read_csv('data/sorted_id_file.csv')
human_liver_df = pd.read_csv('data/HumanLiverCancerPatient1_cell_by_gene.csv')


center_x = sorted_df['center_x']
center_y = sorted_df['center_y']


human_liver_df['x'] = center_x
human_liver_df['y'] = center_y


human_liver_df.to_csv('data/data_addxy.csv', index=False)
