import pandas as pd
import numpy as np
import os
from tqdm import tqdm

print("Loading data...")
file_path = 'data/breast/breast_allcell.csv'  
df = pd.read_csv(file_path)
print("Loading data done。")


print("Normalize the data ......")
feature = 500
genes_data = df.iloc[:, 1:feature+1]
print("加载的列名：", genes_data.columns.tolist())


# genes_log_transformed = np.log1p(genes_data * 100) / np.log(10000)
# genes_log_transformed = np.log1p(genes_data * 10) / np.log(10)
genes_log_transformed = genes_data


genes_log_df = pd.DataFrame(genes_log_transformed, columns=genes_data.columns)
df.iloc[:, 1:feature+1] = genes_log_df

print("Log transformation of the data done")
print("Normalize the data done")


print("Sort data for split region...")
df_sorted = df.sort_values(by=['x', 'y'])
print("Sort data for split region done")


x_min, x_max = df_sorted['x'].min(), df_sorted['x'].max()
y_min, y_max = df_sorted['y'].min(), df_sorted['y'].max()


split_n = 18
window_width = (x_max - x_min) / split_n
window_height = (y_max - y_min) / split_n


df_sorted['region'] = pd.Series(dtype='object')


region_folder = 'data/regions'
if not os.path.exists(region_folder):
    os.makedirs(region_folder)


region_names = []


print("Split region...")

total_regions = split_n * split_n
for n, start_x in enumerate(tqdm(np.arange(x_min, x_max, window_width), desc="Processing X Regions", colour='blue', total=split_n), 1):
    for m, start_y in enumerate(tqdm(np.arange(y_min, y_max, window_height), desc="Processing Y Regions", colour='green', leave=False, total=split_n), 1):
        end_x = start_x + window_width
        end_y = start_y + window_height

        include_right_boundary = n == split_n
        include_bottom_boundary = m == split_n

      
        mask = ((df_sorted['x'] >= start_x) & (df_sorted['x'] < end_x)) | (include_right_boundary & (df_sorted['x'] == x_max))
        mask &= ((df_sorted['y'] >= start_y) & (df_sorted['y'] < end_y)) | (include_bottom_boundary & (df_sorted['y'] == y_max))

        region_name = f'region({n},{m})'
        df_sorted.loc[mask, 'region'] = region_name

       
        region_names.append(region_name)

       
        df_region = df_sorted[mask]

      
        df_region = df_region.loc[~(df_region.iloc[:, 1:feature+1] == 0).all(axis=1)]
     
        if df_region.empty:
            print(f"region {region_name} no data, skip。")
            region_names.remove(region_name)  
            continue
        df_region.to_csv(os.path.join(region_folder, f'{region_name}.csv'), index=False)


unassigned_points = df_sorted['region'].isnull().sum()
if unassigned_points > 0:
    print(f"waring： {unassigned_points} node no use，check boundary requirement。")
else:
    print("All points have been successfully assigned to the area.。")


region_class_file = os.path.join(region_folder, 'region_class.csv')
pd.DataFrame({'Region': region_names}).to_csv(region_class_file, index=False)
print(f"Region names saved to {region_class_file}")


print("Checking and cleaning region files with less than 100 rows...")


region_data = pd.read_csv(region_class_file)
region_names = region_data['Region'].tolist()


for idx in range(len(region_names) - 1, -1, -1):
    current_region = region_names[idx]
    current_file_path = os.path.join(region_folder, f'{current_region}.csv')

    df_current_region = pd.read_csv(current_file_path)

    if len(df_current_region) < 1000:
      
        found_non_empty_previous = False
        previous_idx = idx - 1
        next_idx = idx + 1  

       
        while previous_idx >= 0:
            previous_region = region_names[previous_idx]
            previous_file_path = os.path.join(region_folder, f'{previous_region}.csv')

            if os.path.exists(previous_file_path):
                df_previous_region = pd.read_csv(previous_file_path)
                if not df_previous_region.empty:
                    found_non_empty_previous = True
                    break
            previous_idx -= 1
       
        if not found_non_empty_previous and next_idx < len(region_names):
            next_region = region_names[next_idx]
            next_file_path = os.path.join(region_folder, f'{next_region}.csv')

            if os.path.exists(next_file_path):
                df_next_region = pd.read_csv(next_file_path)
                if not df_next_region.empty:
            
                    df_combined = pd.concat([df_current_region, df_next_region], ignore_index=True)
                    df_combined.to_csv(next_file_path, index=False)

                   
                    os.remove(current_file_path)

                    
                    region_names.pop(idx)
                    print(f"The previous area that is not empty was not found and has been merged with the next area.：{current_region} 和 {next_region}")
        else:
           
            if found_non_empty_previous:
             
                df_combined = pd.concat([df_previous_region, df_current_region], ignore_index=True)
                df_combined.to_csv(previous_file_path, index=False)

       
                os.remove(current_file_path)

          
                region_names.pop(idx)
            else:
               
                print(f"The previous area that is not empty was not found. Keep the current area.：{current_region}")

pd.DataFrame({'Region': region_names}).to_csv(region_class_file, index=False)
print(f"Updated region names saved to {region_class_file}")

print("Region files cleanup for insufficient data rows done.")


print("Combining all regions and checking against the original file...")


combined_df = pd.DataFrame()


for region_name in region_names:
    region_file_path = os.path.join(region_folder, f'{region_name}.csv')
    df_region = pd.read_csv(region_file_path)

    if not df_region.empty:
        combined_df = pd.concat([combined_df, df_region], ignore_index=True)
    else:
        print(f"region {region_name} empty，skip。")


combined_df_sorted = combined_df.sort_values(by=['cell'])


original_cells_sorted = df['cell'].sort_values().reset_index(drop=True)


combined_cells_sorted = combined_df_sorted['cell'].reset_index(drop=True)


mismatch = original_cells_sorted.compare(combined_cells_sorted)


if not mismatch.empty:
    print(f"Mismatch found between original file and combined regions in the 'cell' column:\n{mismatch}")
else:
    print("All cells match between the original file and combined regions.")

print("All processing complete.")
