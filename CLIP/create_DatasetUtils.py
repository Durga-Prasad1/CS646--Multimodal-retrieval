import pandas as pd
# import pickle 
import requests 
from PIL import Image
import io
from concat_item_metadata import concat_item_metadata_esci
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from PIL import Image
# import torch
# import torch.nn.functional as F
# from typing import Union, List
from pandarallel import pandarallel

# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

import os
class Product:
    def __init__(self,text,images, esci_label):
        self.text=text
        self.images=images
        self.esci_label=esci_label

def get_bytestream(url:str):
    retries = 30
    for attempt in range(retries):
        try:
            # print(url)
            response = requests.get(url)
            if response.status_code == 200:
                # print("here")
            # print(response.content)
                img = Image.open(io.BytesIO(response.content))

                # print("here2")
                # print(response.content)
                return response.content
                # print("Done")   
        except Exception as e:
            print(f"Error processing image URL {url}: {str(e)}")
            # return None
    return None
    
class Utils:
    def __init__(self, ):
        self.dataset=[]
    def add(self, group, imgl=1):

        # self.df = pd.read_csv(df_name, nrows=nrows)

        self.meta = [{'text': concat_item_metadata_esci(group.iloc[i]),
                       'images': group.iloc[i]['image_urls'].split("[P_IMG]")[1:imgl+1], 
                       'esci': group.iloc[i]['esci_label'],
                       'pid': group.iloc[i]['product_id']} for i in range(len(group))]
        # self.products = [ for i in range(len(self.df))]
        # print(self.meta)
        for i in range(len(self.meta)):
            # urls = self.meta[i]
            self.meta[i]['images'] = [get_bytestream(item.strip()) for item in self.meta[i]['images']]
            # self.meta[i] = met, bytes_u
            
        self.entry={'query': group['query'].iloc[0], 'product_meta': self.meta }
        self.dataset.append(self.entry)


    


    def get_images(self, urls : list):
        return [self.get_bytestream(url) for url in urls]
    
    def save(self, name: str):
        # with open(name, "wb") as F:
        #     pickle.dump(self.dataset, F)
         df = pd.json_normalize(self.dataset)
         df.to_csv(name, index=False)

# class ProductDataset(Dataset):
#     def __init__(self, df, processor):
#         self.df = df
#         self.processor = processor
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         title = concat_item_metadata_esci(row)
#         image_urls = row['image_urls'].split('[P_IMG]')[1:]
#         # Limit to first image URL to maintain consistency
#         image_urls = image_urls[:1] if image_urls else []
#         return {'title': title, 'image_urls': image_urls, 'idx': idx}

# def collate_fn(batch):
#     """Custom collate function to handle variable length data"""
#     titles = [item['title'] for item in batch]
#     # Ensure each item has at least one URL or an empty list
#     image_urls = [item['image_urls'][0] if item['image_urls'] else '' for item in batch]
#     idx = [item['idx'] for item in batch]
    
#     return {
#         'title': titles,
#         'image_urls': image_urls,
#         'idx': idx
#     }

def add_df(row):
    # print(row.keys())
    # p
    # return "a"
    # row['image_d']={}
    # meta = {'text': concat_item_metadata_esci(row),
    #             'images': row['image_urls'].split("[P_IMG]")[1:2], 
    #             'esci': row['esci_label'],
    #             'pid': row['product_id']} 
    try:
        

        images = row['image_urls'].split("[P_IMG]")[1:2]
        print("here")
    except:
        images=[]
    # print(images)
    for i in range(len(images)):
        # urls = self.meta[i]
        try:
            row['image'+str(i)] = get_bytestream(images[i].strip()) 
        except:
            pass
        # self.meta[i] = met, bytes_u
            
    return row

if __name__=='__main__':
    # pass
    # u = Utils()
    df = pd.read_csv('df_Examples_Products_IMG_URLS_test.csv').fillna('')
    df = df[df['product_locale']=='us']
    df = df[df['small_version']==1]
    # print(df.iloc[7])
    # exit(0)
    df = df.apply(add_df, axis=1)
    # df = df.apply(add_df, axis=1)


    import uuid

    # Generate a unique identifier
    unique_id = uuid.uuid4()

    # Create the file name with the UUID appended
    file_name = f'final_esci_all_final_test_{unique_id}.csv'

    #  exit(0)
    # df['images_d'] = images
    # df = pd.json_normalize(df)
    # print(df.iloc[1])
    df.to_csv(file_name, index=False)
    # u.add_df(df)
    # u.save('final_esci_all_final_test.csv')
# Modified dataloader creation (place this in your process_data function)
# def create_dataloader(dataset, world_size, rank, batch_size=32):
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         sampler=sampler,
#         collate_fn=collate_fn,
#         drop_last=True  # Drop the last incomplete batch
#     )
#     return dataloader