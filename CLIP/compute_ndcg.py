import torch
import tqdm
import pandas as pd
from sklearn.metrics import ndcg_score
import torch.nn.functional as F
import torch.multiprocessing as mp
import pickle
import pytrec_eval
from tqdm import tqdm

# df = pd.read_csv('df_Examples_Products_IMG_URLS_test.csv').fillna('')
# df = df[df['product_locale']=='us']
# df = df[df['small_version']==1]
# df.to_csv('df_Examples_Products_IMG_URLS_test_final.csv', index=False)
# exit(0)
df = pd.read_csv("final_esci_all_final_test_38286377-ca48-4bd0-9029-a4e43499daca.csv").fillna('')


product_embeddings = torch.load('product_embeddings_text_title_imageinclude_20241124_035416.pt')
query_embeddings = torch.load('query_embeddings_text_only_20241123_190337.pt')
print(len(product_embeddings))
# exit(0)
print("Inference completed, computing metrics")
# query_embeddings.sort(key=lambda x:x[1])
# print(product_embeddings[0])
# product_embeddings.sort(key=lambda x:x[1])
# print(query_embeddings.size())
# print(product_embeddings.size())
# query_embeddings = [x[0] for x in query_embeddings]
# product_embeddings = [x[0] for x in product_embeddings]
# exit(0)
unique_queries = [0]

for i in range(len(df)):
    # print(dataset.queries[i])
    if i > 0 and df.iloc[i]['query']!=df.iloc[i-1]['query']:
        unique_queries.append(i)

ndcg = 0.0
total_queries = 0
ranked_lists = []
print("Queries: ", len(unique_queries))
esci_label2relevance_pos = {
    "E" : 4,
    "S" : 3,
    "C" : 2,
    "I" : 1,
}
all_rel_labels = [esci_label2relevance_pos[df.iloc[i]['esci_label']] for i in range(len(df))]


queries=[]

for k, j in tqdm(enumerate(unique_queries)):
    query = query_embeddings[j]
    # start,end=0,0
    # if k==0:
    #     start=0
    #     if k+1<len(unique_queries):
    #         end=unique_queries[k+1]
    #     else:
    #         end = len(query_embeddings)
    # elif k==len(unique_queries)-1:
    #     start,end = unique_queries[k], len(query_embeddings)
    # else:
    #     # print(len(unique_queries), k)
    #     start,end = unique_queries[k], unique_queries[k+1]
    
    # query_mask = torch.zeros(len(df), dtype=torch.int32).to(0)
    # query_mask[start:end]=1
    # query_mask = query_mask * all_rel_labels
    queries.append(query)

    
# queries=

# queries *= queries_masks
# Compute the product similarities
queries = torch.stack(queries)
product_embeddings = torch.stack(product_embeddings)
# print(query_embeddings.size())
# print(product_embeddings.size())
queries = F.normalize(queries, p=2, dim=1)
product_embeddings = F.normalize(product_embeddings, p=2, dim=1)

product_similarities = torch.mm(queries, product_embeddings.t())
print(product_similarities.size())
print(product_similarities)
# exit(0)
# Prepare the relevance judgements and scores for pytrec eval
q_rels={}
q_rels["0"]={"0": all_rel_labels[0]}

cnt=0
for i in tqdm(range(len(df))):
    # print(dataset.queries[i])
    if i > 0 and df.iloc[i]['query_id']!=df.iloc[i-1]['query_id']:
        cnt+=1
        q_rels[str(cnt)]={}
        q_rels[str(cnt)][str(i)]=all_rel_labels[i]
    else:
        q_rels[str(cnt)][str(i)]=all_rel_labels[i]

print("Computed qrels")
# Compute similarity score
predicted_rels={}
for i in tqdm(range(len(unique_queries))):
    predicted_rels[str(i)]={}

    # argsort
    sorted_ps = torch.argsort(product_similarities[i], descending=True)[:8000]
    for j in sorted_ps:
        predicted_rels[str(i)][str(int(j.cpu()))]=float(product_similarities[i][j].cpu())+1
    # full
    # for j in range(len(product_embeddings)):
    #     predicted_rels[str(i)][str(j)]=float(product_similarities[i][j].cpu())+1
# import json
# pretty_json = json.dumps(predicted_rels, indent=4)

# print(pretty_json)
evaluator = pytrec_eval.RelevanceEvaluator(
        q_rels, {'map', 'ndcg'}
    )
    
# Compute all metrics
results = evaluator.evaluate(predicted_rels)
avg = {'map': 0, 'ndcg': 0}
# print(results)
for key in avg.keys():
  avg[key]=sum([results[i][key] for i in results.keys()])* 1.00/len(results.keys())
import json
print(json.dumps(avg, indent=1))