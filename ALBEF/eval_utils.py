import pandas as pd
import pytrec_eval
from tqdm import tqdm


def compute_ndcg_from_ranked_indices(ranked_indices,df_path = "final_esci_all_final_test_38286377-ca48-4bd0-9029-a4e43499daca.csv"):
    df = pd.read_csv(df_path).fillna('')
   
    ndcg_gains = {
    "E" : 100,
    "S" : 10,
    "C" : 1,
    "I" : 0,
    }
    all_rel_labels = [ndcg_gains[df.iloc[i]['esci_label']] for i in range(len(df))]

    q_rels={}
    q_rels["0"]={"0": all_rel_labels[0]}
    cnt=0
    for i in tqdm(range(len(df)),desc='computing ndcg'):
        if i > 0 and df.iloc[i]['query_id']!=df.iloc[i-1]['query_id']:
            cnt+=1
            q_rels[str(cnt)]={}
            q_rels[str(cnt)][str(i)]=all_rel_labels[i]
        else:
            q_rels[str(cnt)][str(i)]=all_rel_labels[i]

    evaluator = pytrec_eval.RelevanceEvaluator(
        q_rels, {'ndcg_cut_10', 'ndcg_cut_100'}
    )
  
    q_preds = {str(i):{str(docid): len(ranked_index)-doc_index for doc_index,docid in enumerate(ranked_index)} for i,ranked_index in enumerate(ranked_indices)}
    results = evaluator.evaluate(q_preds)
    avg = {'ndcg_cut_10': 0, 'ndcg_cut_100': 0}

    for key in avg.keys():
        avg[key]=sum([results[i][key] for i in results.keys()])* 1.00/len(results.keys())

    return avg