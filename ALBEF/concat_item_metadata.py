import random
import pandas as pd
from huggingface_hub import hf_hub_download
import gc


def load_all_categories():
    category_filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename='all_categories.txt',
        repo_type='dataset'
    )
    with open(category_filepath, 'r') as file:
        all_categories = [_.strip() for _ in file.readlines()]
    return all_categories

def concat_item_metadata(dp):
    meta = ''
    flag = False
    if dp['title'] is not None:
        meta += dp['title']
        flag = True
    if len(dp['features']) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(dp['features'])
        flag = True
    if len(dp['description']) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(dp['description'])
    dp['cleaned_metadata'] = meta \
        .replace('\t', ' ') \
        .replace('\n', ' ') \
        .replace('\r', '') \
        .strip()
    return dp

def clean_text(text):
    return text \
        .replace('\t', ' ') \
        .replace('\n', ' ') \
        .replace('\r', '') \
        .strip()

def concat_item_metadata_esci(dp):
    meta = ''
    flag = False
    if dp['product_title'] is not None:
        meta += dp['product_title']
        flag = True
    # if len(dp['features']) > 0:
    #     if flag:
    #         meta += ' '
    #     meta += ' '.join(dp['features'])
    #     flag = True
    if len(dp['product_description']) > 0:
        if flag:
            meta += ' '
        meta += dp['product_description']
    return clean_text(meta)
    # return dp