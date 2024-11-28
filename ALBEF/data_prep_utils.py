import json
import pandas as pd
from concat_item_metadata import clean_text, concat_item_metadata_esci
import argparse

def labels_to_rel_json(labels_path,data_csv_path,out_path):
    with open(labels_path, "r") as file:
        relevant_labels = json.load(file)

    df = pd.read_csv(data_csv_path).fillna('')

    product_esci_labels  = {df.iloc[i]['product_id']:df.iloc[i]['esci_label'] for i in range(len(df))}
    q_rels = {}
    for qid,product_ids in relevant_labels.items():
        q_rels[qid] = {}
        for product_id in product_ids: q_rels[qid][product_id] = product_esci_labels[product_id]

    with open(out_path,'w') as f:
        json.dump(q_rels,f)

def get_json_entries(df,img_dir_path):
    entries = []
    for index,row in df.iterrows():
        img_id = str(row['product_id'])+'_'+str(0)
        img_path = img_dir_path+'/'+str(row['product_id'])+'/'+str(0)+'.jpg'
        entries.append({'image':img_path,'caption':concat_item_metadata_esci(row),'image_id':img_id,'query':clean_text(row['query']),'product_id':row['product_id'],'query_id':row['query_id'],'esci':row['esci_label']})
    return entries


def clip_to_albef_data_dump(args):
    df = pd.read_csv(args.data_csv_path).fillna('')
    queries = df['query_id'].unique()
    split = 8*len(queries)//10
    train_queries = queries[:split]
    val_queries = queries[split:]

    labels = {str(qid):{} for qid in queries}
    for i,row in df.iterrows():
       labels[str(row['query_id'])][row['product_id']] = row['esci_label']
 
    train_df = df[df['query_id'].isin(train_queries)]
    train_entries = get_json_entries(train_df,args.img_dir_path)
    with open(args.data_out_path, 'w') as file:
        file.write(json.dumps(train_entries))
    
    train_data_labels = {str(qid):labels[str(qid)] for qid in train_queries}
    with open(args.labels_out_path, 'w') as file:
        json.dump(train_data_labels,file)

    
    val_df = df[df['query_id'].isin(val_queries)]
    val_entries = get_json_entries(val_df,args.img_dir_path)
    with open(args.val_data_out_path, 'w') as file:
        file.write(json.dumps(val_entries))
    
    val_data_labels = {str(qid):labels[str(qid)] for qid in val_queries}
    with open(args.val_labels_out_path, 'w') as file:
        json.dump(val_data_labels,file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv_path', default='../../train_all.csv')    
    parser.add_argument('--data_out_path', default='./data/train_data.json')
    parser.add_argument('--labels_out_path', default='./data/train_labels.json')
    parser.add_argument('--val_data_out_path', default='./data/val_data.json')
    parser.add_argument('--val_labels_out_path', default ='./data/val_labels.json')
    parser.add_argument('--img_dir_path',default='AmazonFullImageCache')
    args = parser.parse_args()
    clip_to_albef_data_dump(args)
    