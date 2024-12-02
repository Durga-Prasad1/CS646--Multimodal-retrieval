import json
import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root,setting='Q_PI', max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        self.setting = setting
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        text = ''
        if self.setting == 'Q_PI': text = pre_caption(ann['query'],self.max_words)
        else: text = pre_caption(ann['caption'], self.max_words) 

        return image, text, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
    
class re_product_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.queries = {}
        
        for ann in self.ann:
            self.image.append(ann['image'])
            caption = ann['caption']
            self.text.append(pre_caption(caption,self.max_words))
            if ann['query'] not in self.queries.keys():
                self.queries[ann['query_id']] = ann['query']
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = ""
        image = None
        if self.ann[index]['image'] != '' : 
            image_path = os.path.join(self.image_root, self.ann[index]['image']) 
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image) 
        else:
            image = torch.zeros((3,384,384))
         

        return image, index
    
class re_product_image_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, ann_file_indices):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.image = []
        
        for ann_index in ann_file_indices:
            self.image.append(self.ann[ann_index]['image'])
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = ""
        image = None
        if self.ann[index]['image'] != '' : 
            image_path = os.path.join(self.image_root, self.ann[index]['image']) 
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image) 
        else:
            image = torch.zeros((3,384,384))  

        return image, index
    
class re_product_inbatch_dataset(Dataset):
    def __init__(self, ann_file,labels_file,transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.labels = json.load(open(labels_file,'r'))
        self.qid_product = {qid:list(self.labels[qid].keys()) for qid in self.labels.keys()}

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.product_ids_index = {}
        self.queries = {}

        for index,ann_sample in enumerate(self.ann):
            self.text.append(pre_caption(ann_sample['caption'],self.max_words))
            self.image.append(ann_sample['image'])
            self.product_ids_index[ann_sample['product_id']] = index 
            if ann_sample['query'] not in self.queries.keys():
                self.queries[ann_sample['query_id']] = ann_sample['query']
        
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        image_path = ""
        image = None
        if self.image[index] != '' : 
            image_path = os.path.join(self.image_root, self.image[index]) 
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image) 
        else:
            image = torch.zeros((3,384,384))   

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

    
        
def getProductDataloaderForImageIndices(config,image_indices):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    return re_product_image_dataset(config['test_file'], test_transform, config['image_root'],image_indices) 