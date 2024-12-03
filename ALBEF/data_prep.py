import pandas as pd
import requests
from PIL import Image
import io
import json
import time
import ast
from concat_item_metadata import concat_item_metadata_esci
import argparse

def prep_test_data(args):
    data = []
    df = None 
    if args.sample_count > 0 : 
        df = pd.read_csv(args.data_path,nrows = args.sample_count).fillna('')
    else:
        df = pd.read_csv(args.data_path).fillna('')

    for index,row in df.iterrows():
        
        if row['product_locale'] != 'us' : continue

        img_id = str(row['product_id'])+'_'+str(0)
        img_path = args.img_root_path+'/'+img_id+'.jpg'
        data_info = {'image':img_path,'caption':concat_item_metadata_esci(row),'image_id':img_id,'query':row['query'],'product_id':row['product_id'],'query_id':row['query_id'],'esci':row['esci_label']}

        if row['image0'] != '':
            image = Image.open(io.BytesIO(ast.literal_eval(row['image0'])))
            image.save(img_path)
            data.append(data_info)

        elif row['image_urls'] == '':
            data_info['image'] = ""
            data.append(data_info)


        else:
            product_image_fetched = False
            if args.fetch_image:
                print('fetching_image')
                for img_index,img_url in enumerate(row['image_urls'].split('[P_IMG]')[1:]):
                    if product_image_fetched : break
                    retries = 3
                    for _ in range(retries):
                        try:
                            response = requests.get(img_url.strip())
                            if response.status_code == 200:
                                product_image_fetched = True
                                img = Image.open(io.BytesIO(response.content))
                                img.save(img_path)
                                data.append(data_info)
                                break
                            time.sleep(1)  # Wait for 2 seconds before retrying
                        except:
                            time.sleep(1)
            if product_image_fetched == False: 
                 data_info['image'] = ""
                 data.append(data_info)

        
    with open(args.out_path, 'w') as file:
        file.write(json.dumps(data))
    
    data_labels = {}
    for qid in df['query_id'].unique():
        data_labels[str(qid)] = {}
        for i,row in df[df['query_id'] == qid].iterrows():
            data_labels[str(qid)][row['product_id']] = row['esci_label']
    
    with open(args.labels_out_path, 'w') as file:
        json.dump(data_labels,file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../esci_rand_test_10k.csv')    
    parser.add_argument('--out_path', default='./data/test_data.json')
    parser.add_argument('--labels_out_path', default='./data/test_labels.json')
    parser.add_argument('--img_root_path', default='./data/images')
    parser.add_argument('--sample_count', default = 10,type=int)
    parser.add_argument('--fetch_image', default=False)
    args = parser.parse_args()
    prep_test_data(args)