import pandas as pd
import pytrec_eval
from tqdm import tqdm
import json

def compute_metrics_from_ranked_indices(qids,ranked_indices,data_file,data_labels_file):
    q_rels={}
            
    with open(data_file, "r") as file:
        data = json.load(file)
    with open(data_labels_file, "r") as file:
        data_labels = json.load(file)

    ndcg_gains = {
    "E" : 100,
    "S" : 10,
    "C" : 1,
    "I" : 0,
    }

    for qid,qid_rels in data_labels.items():
        q_rels[qid] = {}
        for product_id,esci_label in qid_rels.items(): 
            if esci_label == 'I': continue
            q_rels[qid][product_id] = ndcg_gains[esci_label]

    evaluator = pytrec_eval.RelevanceEvaluator(
        q_rels, {'ndcg','ndcg_cut_10', 'ndcg_cut_100','P_5','P_10','recall_5','recall_10'}
    )
  
    q_preds = {str(qids[i]):{str(data[docid]['product_id']): len(ranked_index)-doc_index for doc_index,docid in enumerate(ranked_index)} for i,ranked_index in enumerate(ranked_indices)}
    results = evaluator.evaluate(q_preds)
    avg = {'ndcg':0,'ndcg_cut_10': 0, 'ndcg_cut_100': 0,'P_5':0,'P_10':0,'recall_5':0,'recall_10':0}

    for key in avg.keys():
        avg[key]=sum([results[i][key] for i in results.keys()])* 1.00/len(results.keys())

    return {'P_5':avg['P_5'],'P_10':avg['P_10']},{'recall_5':avg['recall_5'],'recall_10':avg['recall_10']},{'ndcg':avg['ndcg'],'ndcg_cut_10':avg['ndcg_cut_10'],'ndcg_cut_100':avg['ndcg_cut_100']}


def compute_metrics_from_ranked_indices_v2(qids,ranked_indices,data_file,data_labels_file,df_path = "df_Examples_Products_IMG_URLS_test.csv"):
    df = pd.read_csv(df_path).fillna('')
   
    ndcg_gains = {
    "E" : 100,
    "S" : 10,
    "C" : 1,
    "I" : 0,
    }
    all_rel_labels = {df.iloc[i]['product_id']:ndcg_gains[df.iloc[i]['esci_label']] for i in range(len(df))}

    q_rels={}
            
    with open(data_file, "r") as file:
        data = json.load(file)
    with open(data_labels_file, "r") as file:
        data_labels = json.load(file)
    
    for qid,qid_rels in data_labels.items():
        q_rels[qid] = {}
        for product_id in qid_rels: 
            if all_rel_labels[product_id] == 'I': continue
            q_rels[qid][product_id] = all_rel_labels[product_id]

    evaluator = pytrec_eval.RelevanceEvaluator(
        q_rels, {'ndcg_cut_10', 'ndcg_cut_100','P_5','P_10','recall_5','recall_10'}
    )
  
    q_preds = {str(qids[i]):{str(data[docid]['product_id']): len(ranked_index)-doc_index for doc_index,docid in enumerate(ranked_index)} for i,ranked_index in enumerate(ranked_indices)}
    results = evaluator.evaluate(q_preds)
    avg = {'ndcg_cut_10': 0, 'ndcg_cut_100': 0,'P_5':0,'P_10':0,'recall_5':0,'recall_10':0}

    for key in avg.keys():
        avg[key]=sum([results[i][key] for i in results.keys()])* 1.00/len(results.keys())

    return {'P_5':avg['P_5'],'P_10':avg['P_10']},{'recall_5':avg['recall_5'],'recall_10':avg['recall_10']},{'ndcg_cut_10':avg['ndcg_cut_10'],'ndcg_cut_100':avg['ndcg_cut_100']}

