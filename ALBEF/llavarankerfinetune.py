import pandas as pd
import pickle 
import requests 
from PIL import Image
import io
from concat_item_metadata import concat_item_metadata_esci
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Union, List
import os

# class Utils:
#     def __init__(self, ):
#         self.dataset=[]
#     def add(self, group, imgl=1):
#         # self.df = pd.read_csv(df_name, nrows=nrows)
#         self.meta = [(concat_item_metadata_esci(group.iloc[i]), group.iloc[i]['image_urls'].split("[P_IMG]")[1:imgl+1]) for i in range(len(group))]
#         # self.products = [ for i in range(len(self.df))]
#         # print(self.meta)
#         for i in range(len(self.meta)):
#             met, urls = self.meta[i]
#             bytes_u = [self.get_bytestream(item.strip()) for item in urls]
#             self.meta[i] = met, bytes_u
            
#         self.entry={'query': group['query'].iloc[0], 'product_meta': self.meta}
#         self.dataset.append(self.entry)


#     def get_bytestream(self, url:str):
#         retries = 30
#         for attempt in range(retries):
#             try:
#                 # print(url)
#                 response = requests.get(url)
#                 if response.status_code == 200:
#                     # print("here")
#                 # print(response.content)
#                     img = Image.open(io.BytesIO(response.content))

#                     # print("here2")
#                     # print(response.content)
#                     return response.content
#                     # print("Done")   
#             except Exception as e:
#                 print(f"Error processing image URL {url}: {str(e)}")
#                 # return None
#         return "inky"
#     def get_images(self, urls : list):
#         return [self.get_bytestream(url) for url in urls]
    
#     def save(self, name: str):
#         with open(name, "wb") as F:
#             pickle.dump(self.dataset, F)

class LlavaProductRanker:
    def __init__(self, model_name="liuhaotian/llava-v1.5-7b"):
        """
        Initialize LLaVA model for multimodal product ranking
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b", 
            use_fast=False, 
            trust_remote_code=True
        )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
    def encode(self, query, product_metadata):
                        # Get text embedding
        query_inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        pos_text_inputs = [self.processor(text=pi[0],return_tensors="pt",padding=True,truncation=True,max_length=77).to(self.device) for pi in product_metadata]
        pos_image_inputs = [self.processor(images=Image.open(pi[1]),return_tensors="pt",padding=True,truncation=True,max_length=77).to(self.device) for pi in product_metadata]

        neg_text_inputs = [self.processor(text=pi[0],return_tensors="pt",padding=True,truncation=True,max_length=77).to(self.device) for pi in neg_product_metadata]
        neg_image_inputs = [self.processor(images=Image.open(pi[1]),return_tensors="pt",padding=True,truncation=True,max_length=77).to(self.device) for pi in negproduct_metadata]
        return query_inputs, pos_image_inputs, pos_text_inputs, neg_text_inputs, neg_image_inputs
    
    def forward(self, query, product_metadata, neg_product_metadata):
        """
        Forward pass with InfoNCE loss computation, use info loss to maximize inner product between query representation
        and average of text and image embedding, this should be maximized for positive samples and minimized for negative samples.
        """
        query_inputs, pos_image_inputs, pos_text_inputs, neg_text_inputs, neg_image_inputs = self.encode(query, product_metadata, neg_product_metadata)
        
        # Get the embeddings from the model
        query_embedding = self.model(**query_inputs).last_hidden_state.mean(dim=1)
        pos_text_embeddings = [self.model(**pti).last_hidden_state.mean(dim=1) for pti in pos_text_inputs]
        pos_image_embeddings = [self.model(**pii).last_hidden_state.mean(dim=1) for pii in pos_image_inputs]
        neg_text_embeddings = [self.model(**nti).last_hidden_state.mean(dim=1) for nti in neg_text_inputs]
        neg_image_embeddings = [self.model(**nii).last_hidden_state.mean(dim=1) for nii in neg_image_inputs]

        # Compute the average embeddings for positive and negative samples
        pos_embeddings = [0.5 * (pte + pie) for pte, pie in zip(pos_text_embeddings, pos_image_embeddings)]
        neg_embeddings = [0.5 * (nte + nie) for nte, nie in zip(neg_text_embeddings, neg_image_embeddings)]

        # Compute the similarities
        pos_similarities = [torch.matmul(query_embedding, pe.T) for pe in pos_embeddings]
        neg_similarities = [torch.matmul(query_embedding, ne.T) for ne in neg_embeddings]

        # Compute the InfoNCE loss
        pos_similarities = torch.cat(pos_similarities, dim=0)
        neg_similarities = torch.cat(neg_similarities, dim=0)
        similarities = torch.cat([pos_similarities, neg_similarities], dim=0)

        labels = torch.cat([torch.ones_like(pos_similarities), torch.zeros_like(neg_similarities)], dim=0).to(self.device)
        loss = F.binary_cross_entropy_with_logits(similarities, labels)

        return loss


if __name__=='__main__':
    # Load data
    # df = pd.read_csv('df_Examples_Products_IMG_URLS_all.csv', nrows=10000).fillna('')
    # utils = Utils()

    # for query, group in df.groupby('query'):
    #     utils.add(group)
    # # print(utils.dataset)
    # utils.save("Final_df_llava_10k.pickle")
    pass
