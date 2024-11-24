import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import CLIPTokenizer, CLIPModel
from PIL import Image
from torchvision import transforms
import io
import os
import torch.nn.functional as F
import torch.multiprocessing as mp
import pickle
from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
import io
from concat_item_metadata import concat_item_metadata_esci
from sklearn.metrics import ndcg_score

import time
import logging
import requests
from PIL import Image
import torch
from torchvision import transforms
from requests.exceptions import RequestException
import datetime
from tqdm import tqdm
import os
import pandas as pd
os.system("OMPI_MCA_opal_cuda_support=true")

DDP_TIMEOUT = 1800  # 30 minutes
SOCKET_TIMEOUT = 60

# data = pickle.load(open("./pickles/Final_df_llava.pickle", "rb"))
# data = pickle.load(open("./pickles/Final_df_llava_10k.pickle", "rb"))
# final_esci_all_final.pkl
# data = pickle.load(open("final_esci_all_final.pkl", "rb"))[:100000]
data = pd.read_csv("final_esci_all_final_test_38286377-ca48-4bd0-9029-a4e43499daca.csv").fillna('')
# data.to_csv('test.csv')
print("Data is loaded now proceeding for inference")
# exit(0)


# Initialize tokenizer and preprocess (same as before)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def process_image(img, max_attempts=1, retry_delay=1, default_size=(224, 224)):
    # Define preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize(default_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(io.BytesIO(img))
        image_tensor = preprocess(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        pass

    return torch.zeros((1, 3, default_size[0], default_size[1]))

class CLIPDatasetRow:
    def __init__(self, query, text_tokens, image_tensor, id):
        self.query=query
        self.text_tokens=text_tokens
        self.image_tensor = image_tensor
        self.id=id
# Create a custom Dataset
class CLIPDataset(Dataset):
    def __init__(self, data, max_length=77):
        self.tokenized_data = []
        self.queries = []
        cnt=0
        self.gains=[]
        for i in range(len(data)):
            query = data.iloc[i]['query']
            
            # product_metadata = 
            
            query_tokens = tokenizer(text=query, return_tensors='pt', padding='max_length', 
                                   truncation=True, max_length=max_length)
            
            # for text, image_bytes in product_metadata:
            self.queries.append((query, i))
            # text_tokens = tokenizer(text=concat_item_metadata_esci(data.iloc[i]), return_tensors='pt', padding='max_length', 
                                    # truncation=True, max_length=max_length)
            # Use only product title as in esci paper
            text_tokens = tokenizer(text=data.iloc[i]['product_title'], return_tensors='pt', padding='max_length', 
                                    truncation=True, max_length=max_length)
            # Commenting image part for now
            try:
                image = Image.open(io.BytesIO(data.iloc[i]['image0']))
                image_tensor = preprocess(image).unsqueeze(0)
            except:
                image_tensor = torch.zeros((1, 3, 224, 224))    
            # image_tensor = torch.zeros((1, 3, 224, 224))    
            self.tokenized_data.append((query_tokens, text_tokens, image_tensor.squeeze(0), i, data.iloc[i]['esci_label']))
            self.gains.append(data.iloc[i]['gain'])
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]


# print("CLIP dataset loading is done")

# CLIPFineTuner class remains the same
class CLIPFineTuner(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(CLIPFineTuner, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
    
    def forward(self, query_tokens, text_tokens, image_tensor=None):

        if query_tokens is None:
            return self.clip_model.get_image_features(pixel_values=image_tensor), self.clip_model.get_image_features(pixel_values=image_tensor)
        # print(query_tokens['attention_mask'].size())
        query_embedding = self.clip_model.get_text_features(input_ids = query_tokens['input_ids'], attention_mask =  query_tokens['attention_mask'])
        text_embedding = self.clip_model.get_text_features(input_ids = text_tokens['input_ids'], attention_mask =  text_tokens['attention_mask'])
        
        if image_tensor is not None:
            image_embedding = self.clip_model.get_image_features(pixel_values=image_tensor)
            product_embedding = (text_embedding + image_embedding) / 2
            # product_embedding = (text_embedding * 0.9 + image_embedding * 0.1) 
        else:
            product_embedding = text_embedding
        
        return query_embedding, product_embedding

import socket

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=DDP_TIMEOUT))
    torch.cuda.set_device(rank)
    
    model = CLIPFineTuner().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model

def compute_recall(rank, world_size, data):
    # Setup DDP

    ddp_model = setup_ddp(rank, world_size)
    
    # Create dataset and dataloader with DistributedSampler
    dataset = CLIPDataset(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    print("Dataloading is complete")
    query_embeddings = []
    product_embeddings = []
    

    # Process data in batches
    ddp_model.eval()  # Set to evaluation mode
    print("Starting the inference")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            query_tokens, text_tokens, image_tensor, idx, _ = batch
            # print(idx.size())
            # Move everything to the correct device
            query_tokens = {k: v.squeeze().to(rank) for k, v in query_tokens.items()}
            text_tokens = {k: v.squeeze().to(rank) for k, v in text_tokens.items()}
            image_tensor = image_tensor.to(rank)
            idx=idx.to(rank)
            
            query_embedding, product_embedding = ddp_model(query_tokens, text_tokens, image_tensor)
            # query_embedding, product_embedding = ddp_model(query_tokens, text_tokens, None)

            # Gather embeddings from all GPUs
            gathered_queries = [torch.zeros_like(query_embedding) for _ in range(world_size)]
            gathered_products = [torch.zeros_like(product_embedding) for _ in range(world_size)]
            idxs = [torch.zeros_like(idx) for _ in range(world_size)]
            dist.all_gather(gathered_queries, query_embedding)
            dist.all_gather(gathered_products, product_embedding)
            dist.all_gather(idxs, idx)
            gathered_queries = [[(gathered_queries[ii][j].to(0), idxs[ii][j].cpu()) for j in range(gathered_queries[ii].shape[0])] for ii in range(world_size)]
            gathered_products = [[(gathered_products[ii][j].to(0), idxs[ii][j].cpu()) for j in range(gathered_products[ii].shape[0])] for ii in range(world_size)]

            
            # Empty the cuda cache
            # del query_embedding
            # del product_embedding
            # torch.cuda.empty_cache()

            if rank == 0:  # Only process on main GPU, pin them to the same GPU
                for k in range(world_size):
                    query_embeddings.extend(gathered_queries[k])
                    product_embeddings.extend(gathered_products[k])
            
            # del gathered_queries
            # del gathered_products
            # torch.cuda.empty_cache()
    if rank==0:


        # Get current timestamp in a filename-friendly format
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save query embeddings with timestamp
        
        print(len(query_embeddings))
        query_embeddings.sort(key=lambda x:x[1])
        product_embeddings.sort(key=lambda x:x[1])
        query_embeddings = [x[0] for x in query_embeddings]
        product_embeddings = [x[0] for x in product_embeddings]
        # Save product embeddings with timestamp
        product_filename = f'product_embeddings_text_title_imageinclude_{timestamp}.pt'
        query_filename = f'query_embeddings_text_only_{timestamp}.pt'
        torch.save(product_embeddings, product_filename)
        torch.save(query_embeddings, query_filename)
        print("Timestamp is :", timestamp)
        # a = torch.load('product_embeddings_text_title_only.pt')
        # b = torch.load('query_embeddings_text_only.pt')
        
    #     print("Inference completed, computing metrics")


    #     unique_queries = [0]

    #     for i, (q_tokens, p_tokens, _, _, _) in enumerate(dataset):
    #         # print(dataset.queries[i])
    #         if i > 0 and dataset.queries[i][0]!=dataset.queries[i-1][0]:
    #             unique_queries.append(i)

    #     ndcg = 0.0
    #     total_queries = 0
    #     ranked_lists = []
    #     print("Queries: ", len(unique_queries))
    #     esci_label2relevance_pos = {
    #         "E" : 4,
    #         "S" : 3,
    #         "C" : 2,
    #         "I" : 1,
    #     }
    #     all_rel_labels = torch.tensor([esci_label2relevance_pos[dataset.tokenized_data[i][-1]] for i in range(len(dataset))]).to(0)
    #     ranked_indices=[]
    #     for k, j in tqdm(enumerate(unique_queries)):
    #         query = query_embeddings[j]
    #         start,end=0,0
    #         if k==0:
    #             start=0
    #             if k+1<len(unique_queries):
    #                 end=unique_queries[k+1]
    #             else:
    #                 end = len(query_embeddings)
    #         elif k==len(unique_queries)-1:
    #             start,end = unique_queries[k], len(query_embeddings)
    #         else:
    #             # print(len(unique_queries), k)
    #             start,end = unique_queries[k], unique_queries[k+1]
            
    #         query_mask = torch.zeros(len(dataset), dtype=torch.int32).to(0)
    #         query_mask[start:end]=1
    #         query_mask = query_mask * all_rel_labels

            

    #         product_similarities = F.cosine_similarity(query.unsqueeze(0), torch.stack(product_embeddings), dim=1)

    #         ranked_index = torch.argsort(product_similarities, descending=True)
    #         ranked_indices.append((j, ranked_index))
    #         print(query_mask.size())
    #         print(product_similarities.size())
    #         hits = ndcg_score([query_mask.cpu().tolist()], [product_similarities.cpu().tolist()])
            
    #         ndcg += hits
    #         total_queries += 1
        
    #     with open('ranked_indicies.pkl', 'wb') as f:
    #         pickle.dump(ranked_indices, f)
        
    #     print("NDCG is :", ndcg / total_queries)
    dist.destroy_process_group()



def main():
    world_size = torch.cuda.device_count()  # Use all available GPUs
    mp.spawn(compute_recall, args=(world_size, data), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

# # def compute_recall_val(query_embeddings, product_embeddings, dataset):
#     """
#         After the above function of compute_Recall is done, we can assume that query_embeddings and product_embeddings
#         have embbedings for each row of CLiP dataset. Now each row of th dataset has a query and relevant product pair,
#         we want to recall@10 when we retrieve with maximum cosine similarity search all products with each query.
#         Assuming the order of the qi, pi in dataset and query_embeddings[i], product_embeddings[i] is the same,
#         first group query_embeddings[i] by unique queries
#         second for each unique query, get the list of relevant product ids from dataset (query_tokens, text_tokens, image_tensor.squeeze(0)) k1, k2, ..kr now, compare similarity
#         with all products in the dataset product_embeddings, then compute recall @10
#     """
#     query_embeddings.sort(key=lambda x:x[1])
#     product_embeddings.sort(key=lambda x:x[1])
#     query_embeddings = [x[0] for x in query_embeddings]
#     product_embeddings = [x[0] for x in product_embeddings]

#     unique_queries = [0]

#     for i, (q_tokens, p_tokens, _, _, _) in enumerate(dataset):
#         # print(dataset.queries[i])
#         if i > 0 and dataset.queries[i][0]!=dataset.queries[i-1][0]:
#             unique_queries.append(i)

#     recall_at_10 = 0.0
#     total_queries = 0
#     ranked_lists = []
#     print("Queries: ", len(unique_queries))
#     for k, j in tqdm(enumerate(range(len(query_embeddings)))):
#         query = query_embeddings[j]
#         start,end=0,0
#         if k==0:
#             start=0
#             if k+1<len(unique_queries):
#                 end=unique_queries[k+1]
#             else:
#                 end = len(query_embeddings)
#         elif k==len(unique_queries)-1:
#             start,end = unique_queries[k], len(query_embeddings)
#         else:
#             # print(len(unique_queries), k)
#             start,end = unique_queries[k], unique_queries[k+1]
#         relevant_products = [i for i in range(start, end) if dataset.tokenized_data[i][-1]!='I']
#         # print(relevant_products)
#         # relevant_products = [i for i in query_index]

#         product_similarities = F.cosine_similarity(query.unsqueeze(0), torch.stack(product_embeddings), dim=1)
#         ranked_products = sorted(enumerate(product_similarities), key=lambda x: x[1], reverse=True)

#         hits_at_10 = sum(1 for i, _ in ranked_products[:100] if i in relevant_products)
#         recall_at_10 += hits_at_10 / len(relevant_products)
#         total_queries += 1
#         ranked_lists.append((j, ranked_products[:100]))
    
#     with open('ranked_data.pkl', 'wb') as f:
#         pickle.dump(ranked_lists, f)
#     return recall_at_10 / total_queries
        # print(quw)
    # Calculate recall only on rank 0
    # if rank == 0:
    #     # print(query_embeddings)
    #     # print(query_embeddings[0].shape)
    #     recalls = []
    #     for q in query_embeddings:
    #         similarities = []
    #         for j, p in enumerate(product_embeddings):
    #             # print(product_embedding.size())
    #             # print(query_embedding.size())
    #             # p, q = product_embedding/product_embedding.norm(dim=1, keepdim=True), query_embedding/query_embedding.norm(dim=1, keepdim=True)
    #             # similarity = torch.mm(q, p.t())
    #             # print(p.size())
    #             # print(q.size())
    #             similarity = F.cosine_similarity(q.unsqueeze(0), p.unsqueeze(0))
    #             print(similarity.size())
    #             similarities.append((j, similarity.mean().item()))
            
    #         similarities.sort(key=lambda x: x[1], reverse=True)
    #         recalls.append(similarities)
        
    #     avg_recall = sum([sim[0][1] for sim in recalls]) / len(recalls)
    #     print(f"Average Recall: {avg_recall}")