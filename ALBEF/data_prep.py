import pandas as pd
import requests
from PIL import Image
import io
import json
import time
from concat_item_metadata import concat_item_metadata_esci
import argparse

def prep_test_data(args):
    samples_needed = args.sample_count
    data = []
    df = pd.read_csv(args.data_path).fillna('')
    processed_row_indices = []
    for index,row in df.iterrows():
        product_image_fetched = False
        if len(row['image_urls']) <= 0 or row['product_locale'] != 'us' : continue
        for img_index,img_url in enumerate(row['image_urls'].split('[P_IMG]')[1:]):
            if product_image_fetched : continue
            retries = 3
            for attempt in range(retries):
                try:
                    response = requests.get(img_url.strip())
                    if response.status_code == 200:
                        product_image_fetched = True
                        img = Image.open(io.BytesIO(response.content))
                        img_id = str(row['product_id'])+'_'+str(img_index)
                        img_path = args.img_root_path+'/'+img_id+'.PNG'
                        img.save(img_path)
                        data.append({'image':img_path,'caption':concat_item_metadata_esci(row),'image_id':img_id,'query':row['query'],'product_id':row['product_id'],'query_id':row['query_id']})
                        processed_row_indices.append(index)
                        samples_needed -= 1
                        if samples_needed <= 0 : 
                            with open(args.out_path, 'w') as file:
                                file.write(json.dumps(data))

                            data_labels = {}
                            data_df = df.iloc[processed_row_indices]
                            for qid in data_df['query_id'].unique():
                                qid = int(qid)
                                qid_relevant_products = data_df[data_df['query_id'] == qid]['product_id'].tolist()
                                data_labels[qid] = qid_relevant_products
                            
                            with open(args.labels_out_path, 'w') as file:
                                json.dump(data_labels,file)

                            return
                        break
                except:
                    pass
                
                time.sleep(1)  # Wait for a seconds before retrying


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../df_Examples_Products_IMG_URLS_test.csv')    
    parser.add_argument('--out_path', default='./data/test_data.json')
    parser.add_argument('--labels_out_path', default='./data/test_labels.json')
    parser.add_argument('--img_root_path', default='./data/images')
    parser.add_argument('--sample_count', default = 10,type=int)
    args = parser.parse_args()
    prep_test_data(args)