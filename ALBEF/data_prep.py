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
    p = pd.read_csv(args.data_path)
    for index,row in p.iterrows():
        if type(row['image_urls']) != str : continue
        for img_index,img_url in enumerate(row['image_urls'].split('[P_IMG]')[1:]):
            retries = 3
            for attempt in range(retries):
                try:
                    response = requests.get(img_url.strip())
                    print(img_url)
                    if response.status_code == 200:
                        print('fetched')
                        img = Image.open(io.BytesIO(response.content))
                        img_extension = img_url.split('.')[-1]
                        img_id = str(row['product_id'])+'_'+str(img_index)
                        img_path = args.img_root_path+'/'+img_id+'.'+'.PNG'
                        img.save(img_path)
                        data.append({'image':img_path,'caption':concat_item_metadata_esci(row),'image_id':img_id,'query':row['query'],'product_id':row['product_id'],'query_id':row['query_id']})
                        samples_needed -= 1
                        if samples_needed <= 0 : 
                            with open(args.out_path, 'w') as file:
                                file.write(json.dumps(data))
                            return
                        break
                    if attempt < retries - 1:
                        time.sleep(1)  # Wait for 2 seconds before retrying
                except:
                    time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../df_Examples_Products_IMG_URLS_test.csv')    
    parser.add_argument('--out_path', default='./data/test_data.json')
    parser.add_argument('--img_root_path', default='./data/images')
    parser.add_argument('--sample_count', default = 10,type=int)
    args = parser.parse_args()
    prep_test_data(args)