import argparse
import os
# import ruamel_yaml as yaml
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from data_prep_utils import clip_to_albef_data_dump, labels_to_rel_json
from dataset.caption_dataset import getProductDataloaderForImageIndices
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from tqdm import tqdm
from eval_utils import compute_metrics_from_ranked_indices, compute_metrics_from_ranked_productIds


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)  
            
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ita, loss_itm = model(image, text_input,alpha=alpha, idx=idx)                  
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def compute_product_metrics(qids,ranked_indices,data_file,data_labels_file):
    with open(data_file, "r") as file:
        data = json.load(file)
    with open(data_labels_file, "r") as file:
        data_labels = json.load(file)

    precision = {1:0,3:0,5:0,10:0}
    recall = {1:0,3:0,5:0,10:0}
    
    for index,qid in enumerate(qids):
        relevant_products = data_labels[str(qid)]
        ranked_product_ids = [data[i]['product_id'] for i in ranked_indices[index]]
        for top_k in [1,3,5,10]:
            ranked_top_k = ranked_product_ids[:top_k]
            # Calculate recall
            relevant_in_top_k = len(list(set(ranked_top_k) & set(relevant_products)))
            recall[top_k] += relevant_in_top_k / len(relevant_products)
            
            # Calculate precision
            precision[top_k] += relevant_in_top_k / top_k

    precision = {key:val/len(qids) for key,val in precision.items()}
    recall = {key:val/len(qids) for key,val in recall.items()}
    # ndcg = compute_ndcg_from_ranked_indices(ranked_indices)

    # return  precision,recall,ndcg 
    return  precision,recall

@torch.no_grad()
def evaluation_product_mean_fusion_inbatch(model,data_loader,tokenizer, device, config, setting='text'):
     # test
    model.eval()   
    
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=75, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state 
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
   
    queries_info = data_loader.dataset.queries
    qids = [] 
    queries = []
    for qid,query_text in queries_info.items():
        qids.append(qid)
        queries.append(query_text)
    num_queries = len(queries)
    query_bs = 256
    query_feats = []
    query_embeds = []
    query_atts = []
    for i in tqdm(range(0, num_queries, query_bs)):
        text = queries[i: min(num_queries, i+query_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=75, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state  
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        query_embeds.append(text_embed)  
        query_feats.append(text_feat)
        query_atts.append(text_input.attention_mask)
    query_embeds = torch.cat(query_embeds,dim=0)
    query_feats = torch.cat(query_feats,dim=0)
    query_atts = torch.cat(query_atts,dim=0)

    image_feats = []
    image_embeds = []
    image_feats = []
    image_embeds = []
    if setting == 'fusion':
        for image, img_id in tqdm(data_loader): 
            image = image.to(device) 
            image_feat = model.visual_encoder(image)          
            image_feats.append(image_feat)
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)   
            image_embeds.append(image_embed)
    
        image_feats = torch.cat(image_feats,dim=0)
        image_embeds = torch.cat(image_embeds,dim=0)

    ranked_indices = []
    for qid_index,qid in enumerate(qids):
        qid_products = data_loader.dataset.qid_product[str(qid)]
        qid_product_embeds = []
        for qid_product in qid_products:
            index = data_loader.dataset.product_ids_index[qid_product]
            product_embed = text_embeds[index]
            if setting == 'fusion':
                product_embed = image_embeds[index] + text_embeds[index]
            qid_product_embeds.append(product_embed.unsqueeze(0))
        qid_sims = torch.mm(query_embeds[qid_index].unsqueeze(0),torch.cat(qid_product_embeds,dim=0).T)
        sorted_indices = torch.argsort(qid_sims, descending=True,dim=1).squeeze(0)
        ranked_indices.append([qid_products[sorted_index.item()] for sorted_index in sorted_indices])

    return qids,ranked_indices
    
@torch.no_grad()
def evaluation_product_mean_fusion(model,data_loader,tokenizer, device, config, setting='text'):
     # test
    model.eval()   
    
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=75, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state 
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
   
    queries_info = data_loader.dataset.queries
    qids = [] 
    queries = []
    for qid,query_text in queries_info.items():
        qids.append(qid)
        queries.append(query_text)
    num_queries = len(queries)
    query_bs = 256
    query_feats = []
    query_embeds = []
    query_atts = []
    for i in tqdm(range(0, num_queries, query_bs)):
        text = queries[i: min(num_queries, i+query_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=75, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state  
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        query_embeds.append(text_embed)  
        query_feats.append(text_feat)
        query_atts.append(text_input.attention_mask)
    query_embeds = torch.cat(query_embeds,dim=0)
    query_feats = torch.cat(query_feats,dim=0)
    query_atts = torch.cat(query_atts,dim=0)

    similarities_t = torch.mm(query_embeds,text_embeds.T)
    similarities = similarities_t
    if setting == 'fusion':

        image_feats = []
        image_embeds = []
        for image, img_id in tqdm(data_loader): 
            image = image.to(device) 
            image_feat = model.visual_encoder(image)          
            image_feats.append(image_feat)
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)   
            image_embeds.append(image_embed)
        
        image_feats = torch.cat(image_feats,dim=0)
        image_embeds = torch.cat(image_embeds,dim=0)

        similarities_i =  torch.mm(query_embeds,image_embeds.T)
        similarities = similarities_i + similarities_t

    ranked_indices = torch.argsort(similarities, descending=True,dim=1).cpu().tolist()
    return qids,ranked_indices

@torch.no_grad()
def evaluation_product_t2i(model, data_loader, tokenizer, device, config, only_t2i_retrieval = False):
    # test
    model.eval() 

    start_time = time.time()  
    # No. of items to retrieve for stage 1
    retrieval_count = min(len(data_loader.dataset),config['k_test'])
    # Queries
    queries_info = data_loader.dataset.queries
    qids = [] 
    queries = []
    for qid,query_text in queries_info.items():
        qids.append(qid)
        queries.append(query_text)
    num_queries = len(queries)
    query_bs = 256
    query_feats = []
    query_embeds = []
    query_atts = []
    for i in tqdm(range(0, num_queries, query_bs),desc='encoding_queries'):
        text = queries[i: min(num_queries, i+query_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=75, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state  
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        query_embeds.append(text_embed)  
        query_feats.append(text_feat)
        query_atts.append(text_input.attention_mask)
    query_embeds = torch.cat(query_embeds,dim=0)
    query_feats = torch.cat(query_feats,dim=0)
    query_atts = torch.cat(query_atts,dim=0)
    # Product texts
    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs),desc='encoding_product_descriptions'):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=75, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)

    # ranking
    sims_matrix_text_only = query_embeds @ text_embeds.t()
    ranked_indices = torch.argsort(sims_matrix_text_only, descending=True,dim=1)
    ranked_indices = ranked_indices[:,:retrieval_count]
    
    # re-ranking
    score_matrix_t2i = torch.full((num_queries,retrieval_count),-100.0).to(device)
    for query_index, text_only_ranked_indices in enumerate(tqdm(ranked_indices,desc='reranking_topk')):
        image_loader = create_loader([getProductDataloaderForImageIndices(config,text_only_ranked_indices)],[None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    is_trains=[False], 
                                    collate_fns=[None])[0]

        image_feats = []
        image_embeds = []
        for image, _ in image_loader: 
            image = image.to(device) 
            image_feat = model.visual_encoder(image)        
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)      
            image_feats.append(image_feat)
            image_embeds.append(image_embed)
        image_feats = torch.cat(image_feats,dim=0)
        image_embeds = torch.cat(image_embeds,dim=0)
        image_att = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = query_feats[query_index].repeat(retrieval_count,1,1), 
                                    attention_mask = query_atts[query_index].repeat(retrieval_count,1),
                                    encoder_hidden_states = image_feats,
                                    encoder_attention_mask = image_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[query_index] = score

    score_matrix_t2i = torch.argsort(score_matrix_t2i, descending=True,dim=1).cpu()
    reranked_indices = [[0 for _ in range(score_matrix_t2i.shape[1])] for _ in range(score_matrix_t2i.shape[0])]
    for index,stage2_ranked_indices in enumerate(score_matrix_t2i):
        reranked_indices[index] = [ranked_indices[index][i].item() for i in stage2_ranked_indices]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 
    return qids , reranked_indices

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result




def main(args, config):
    #utils.init_distributed_mode(args)    
    device = torch.device(args.device)
    args.distributed=False
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('pre', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint
        # state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        # pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        # state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        # m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        # state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    
    model_without_ddp = model 
    print(args.distributed)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module  
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  

        # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
       

        if utils.is_main_process():  
      
            # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)  

            if args.evaluate:      

                # qids,ranked_indices = evaluation_product_mean_fusion_inbatch(model_without_ddp, test_loader, tokenizer, device, config, 'fusion')
                # precision, recall, ndcg = compute_metrics_from_ranked_productIds(qids,ranked_indices,config['test_file'],config['test_labels'])

                # qids,ranked_indices = evaluation_product_mean_fusion(model_without_ddp, test_loader, tokenizer, device, config, 'fusion')
                qids,ranked_indices = evaluation_product_t2i(model_without_ddp, test_loader, tokenizer, device, config)
                precision, recall, ndcg = compute_metrics_from_ranked_indices(qids,ranked_indices,config['test_file'],config['test_labels'])
                print(precision,recall,ndcg)          
                # log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                #              **{f'test_{k}': v for k, v in test_result.items()},                  
                #              'epoch': epoch,
                #             }
                # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")     
            else:
                qids,ranked_indices = evaluation_product_mean_fusion(model_without_ddp, val_loader, tokenizer, device, config, 'fusion')
                val_precision, val_recall, val_ndcg = compute_metrics_from_ranked_indices(qids,ranked_indices,config['val_file'],config['val_labels'])
                # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                #              **{f'val_{k}': v for k, v in val_result.items()},
                #              **{f'test_{k}': v for k, v in test_result.items()},                  
                #              'epoch': epoch,
                #             }
                # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")   
                    
                # if val_result['r_mean']>best:
                # val_recall_mean = sum(val_recall.values())/len(val_recall.items())
                val_ndcg_100 = val_ndcg['ndcg_cut_100']
                if val_ndcg_100 > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    # best = val_result['r_mean']    
                    best = val_ndcg_100
                    best_epoch = epoch
                    print('best_val',val_precision,val_recall,val_ndcg)
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    yaml = YAML(typ='rt')

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = yaml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  

    
    main(args, config)
