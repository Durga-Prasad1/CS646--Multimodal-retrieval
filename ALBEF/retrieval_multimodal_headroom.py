import pandas as pd
import numpy as np
import torch
# from clip.simple_tokenizer import SimpleTokenizer
# from clip.model import CLIP
# import clip
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
# import io.Byte
import requests
import time
from concat_item_metadata import concat_item_metadata_esci
import pickle

# Load the CLIP model and processor from transformers
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# load prev dict
# with open("image_dict_url_bytestream_2024-11-08_17-40-13.pickle","rb") as F:
#     dic = pickle.load(F)

# Load the data from the CSV
df = pd.read_csv('df_Examples_Products_IMG_URLS_test.csv', nrows=10000).fillna('')
# print(df.iloc[0])
# df = df[df['split']=='test']
# print(df)

im_dict={}

def encode_product(title, image_urls):
    # Encode the product title
    inputs = processor(text=[title], return_tensors="pt", padding=True,  truncation=True, max_length=77).to(device)
    title_embedding = model.get_text_features(**inputs).squeeze(0).to(device)
    

    # Encode the product images
    image_embeddings = []
    for url in image_urls[:1]:
        retries = 3
        for attempt in range(retries):
            try:
                # img = Image.open(io.BytesIO(dic[url]))
                # inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
                # img_tensor = model.get_image_features(**inputs).squeeze(0).to(device)
                # image_embeddings.append(img_tensor)
                # print("Image retrieved successfully")
                # im_dict[url] = response.content
                # break  # Exit the retry loop if successful
                response = requests.get(url)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
                    img_tensor = model.get_image_features(**inputs).squeeze(0).to(device)
                    image_embeddings.append(img_tensor)
                    print("Image retrieved successfully")
                    im_dict[url.strip()] = response.content
                    break  # Exit the retry loop if successful
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait for 2 seconds before retrying
                else:
                    print("Giving up after several attempts", url)
    
    if image_embeddings:
        image_embedding = torch.stack(image_embeddings).mean(dim=0)
    else:
        image_embedding = title_embedding

    normalized_title_embedding = torch.nn.functional.normalize(title_embedding, p=2, dim=0)
    normalized_image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=0)

    # print("img: ", image_embedding.size())
    # print("title: ",title_embedding.size() )
    # out = (normalized_title_embedding + normalized_image_embedding) / 2
    out = torch.cat((normalized_title_embedding, normalized_image_embedding), dim=0)
    return torch.nn.functional.normalize(out, p=2, dim=0)
    # return torch.cat((normalized_title_embedding, normalized_image_embedding), dim=0)

# Precompute all product embeddings
product_embeddings_t, product_embeddings_i = [], []
for i, row in df.iterrows():
    title = concat_item_metadata_esci(row)
    image_urls = row['image_urls'].split('[P_IMG]')[1:]
    # if use_images:
    product_embeddings_i.append(encode_product(title, image_urls))
    # else:
    product_embeddings_t.append(encode_product(title, []))
    # product_embeddings.append(product_embedding)

# Function to rank products based on the query
def rank_products(query, use_images=True, k=10):
    query_embedding = encode_product(query, [])
    
    if use_images:
        product_embeddings = torch.stack(product_embeddings_i)
    else:
        product_embeddings = torch.stack(product_embeddings_t)
    
    # print(query_embedding.unsqueeze(0).size(), product_embeddings.T.size())
    similarities = torch.mm(query_embedding.unsqueeze(0), product_embeddings.T).squeeze()
    
    ranked_indices = torch.argsort(similarities, descending=True).cpu()
    
    return df.iloc[ranked_indices].head(k)

# Function to evaluate recall and precision
def evaluate_recall_precision(query, relevant_products, k=10, use_images=True):
    ranked_products = rank_products(query, use_images)
    
    # Calculate recall
    relevant_in_top_k = ranked_products['product_id'].isin(relevant_products['product_id']).sum()
    recall = relevant_in_top_k / len(relevant_products)
    
    # Calculate precision
    precision = relevant_in_top_k / k
    
    return recall, precision

# Compute average recall and precision with and without images
total_recall_text = 0
total_precision_text = 0
total_recall_text_images = 0
total_precision_text_images = 0
num_queries = 0

for query, group in df.groupby('query'):
    relevant_products = df[df['query_id'] == group['query_id'].iloc[0]]
    # print(relevant_products)
    # Evaluate without images
    recall, precision = evaluate_recall_precision(query, relevant_products, k=10, use_images=False)
    total_recall_text += recall
    total_precision_text += precision
    
    # Evaluate with images
    recall, precision = evaluate_recall_precision(query, relevant_products, k=10, use_images=True)
    total_recall_text_images += recall
    total_precision_text_images += precision
    
    num_queries += 1

average_recall_text = total_recall_text / num_queries
average_precision_text = total_precision_text / num_queries
average_recall_text_images = total_recall_text_images / num_queries
average_precision_text_images = total_precision_text_images / num_queries

print(f"Number of queries {len(df.groupby('query'))}")
print(f"Average Recall@10 (Text Only): {average_recall_text}")
print(f"Average Precision@10 (Text Only): {average_precision_text}")
print(f"Average Recall@10 (Text + Images): {average_recall_text_images}")
print(f"Average Precision@10 (Text + Images): {average_precision_text_images}")

import pickle
import uuid
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# save dictionary to pickle file
with open(f"image_dict_url_bytestream_{timestamp}.pickle", 'wb') as file:
    pickle.dump(im_dict, file, protocol=pickle.HIGHEST_PROTOCOL)