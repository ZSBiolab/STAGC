import pandas as pd
import os
from tqdm import tqdm
#6
def merge_and_sort_embeddings(data_dir, embedding_dir, region_class_csv, output_csv):

    regions = pd.read_csv(region_class_csv, usecols=[0]).iloc[:, 0]

    all_embeddings = []
    for region in tqdm(regions, desc="Processing regions"):

        embedding_file = os.path.join(embedding_dir, f'embedding_{region}.csv')
        embeddings = pd.read_csv(embedding_file)


        index_file = os.path.join(data_dir, f'{region}.csv')
        indices = pd.read_csv(index_file, usecols=[0])


        if len(indices) != len(embeddings):
            raise ValueError(f"索引数量与嵌入数据数量不匹配: {region}")

        embeddings.insert(0, 'cell', indices.values)
        all_embeddings.append(embeddings)
    print('adding index...')

    combined_embeddings = pd.concat(all_embeddings, ignore_index=True)
    combined_embeddings.sort_values(by='cell', inplace=True)
    print('merging data...')

    combined_embeddings.to_csv(output_csv, index=False)


def main():
    data_dir = 'data/data/regions'
    embedding_dir = 'run/embedding'
    region_class_csv = 'data/data/regions/region_class.csv'
    #data_addxy_csv = 'data/data/data_addxy.csv'
    output_csv = 'run/embedding/final_embeddings.csv'

    #merge_and_sort_embeddings(data_dir, embedding_dir, region_class_csv, data_addxy_csv, output_csv)
    merge_and_sort_embeddings(data_dir, embedding_dir, region_class_csv, output_csv)

if __name__ == '__main__':
    main()
