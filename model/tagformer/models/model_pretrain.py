import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import numpy as np
import random, json
from transformers import BertTokenizer

from models.xbert import BertConfig, BertForMaskedLM
from models.gt import Graphormer, MLP_dec
from models.loss_fn import TripletLoss

def load_config_from_json_file(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def all_to_device(lst, device):
    return (x.to(device) for x in lst)

def setup_loss_fn(loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        # elif loss_fn == "sce":
        #     criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "nll":
            criterion = nn.NLLLoss()
        elif loss_fn == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == "cos":
            criterion = nn.CosineSimilarity()
        elif loss_fn == "mae":
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError
        return criterion

class RTL_Fusion(nn.Module):
    def __init__(self,   
                 config,
                 device,
                 accelerator=None
                 ):
        super(RTL_Fusion, self).__init__()

        self.device = device
        self.config = config
        if accelerator:
            self.accelerator = accelerator

        ### loss function  initialization ###
        margin_num = 1.0
        self.loss_cl = TripletLoss(margin = margin_num)
        self.loss_gtmae = setup_loss_fn("mse")
        
        ### BERT initialization ###
        bert_model = config['bert_model_name']
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.summary_enc = BertForMaskedLM.from_pretrained(bert_model, config=bert_config)

        self.embed_dim = config['embed_dim']
        summary_embed_dim = self.summary_enc.config.hidden_size
        self.sumamry_proj = nn.Linear(summary_embed_dim, self.embed_dim)
        self.mlm_probability = config['mlm_probability']

        ### Graphormer initialization ###
        self.gt_config = load_config_from_json_file(config['gt_config'])
        self.graphormer = Graphormer(self.gt_config)
        self.graph_proj = nn.Linear(self.gt_config['embed_dim'], self.embed_dim)
        self.gtmae_decoder = MLP_dec(
            input_dim=self.embed_dim,
            hidden_dim=512,
            num_layers=3,
            output_dim=self.gt_config['feat_dim'],
            activation="relu",
            norm="batchnorm"
        )
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.gsm_head = nn.Linear(self.embed_dim, 2)  

        ### Text embedding initialization ###
        self.text_proj = nn.Linear(config['text_width'], self.embed_dim)   
    
    def load_net_enc(self, net_enc):
        self.net_enc = net_enc
        

    def forward(self, data, mode = 'pretrain'):
        self.mode = mode
        if self.mode == 'pretrain':
            return self.pretrain_forward(data)
        elif self.mode == 'infer':
            return self.finetune_forward(data)
        else:
            raise ValueError("Invalid mode")

    def summary_encode(self, summary_data):
        if self.mode == 'infer':
            enc_mode = "multi_modal"
        elif self.mode == 'pretrain':
            enc_mode = "text"
        else:
            enc_mode = "text"
        input = self.tokenizer(summary_data, return_tensors="pt", padding='longest', truncation=True, max_length=512).to(self.device)
        output = self.summary_enc.bert(input.input_ids, attention_mask=input.attention_mask, return_dict=True, mode=enc_mode)
        embeds = F.normalize(self.sumamry_proj(output.last_hidden_state[:, 0, :]),dim=-1)

        return embeds, input, output

    def graph_encode(self, graph_data):
        (attn_mask,node_feat,in_degree,out_degree,path_data,dist) = all_to_device(graph_data, self.device)
        batched_data = (attn_mask,node_feat,in_degree,out_degree,path_data,dist)

        node_rep, graph_rep = self.graphormer(batched_data)
        embeds = F.normalize(self.graph_proj(graph_rep),dim=-1)

        del (attn_mask,node_feat,in_degree,out_degree,path_data,dist,batched_data)
        torch.cuda.empty_cache()

        return node_rep, embeds

    def finetune_forward(self, data):
        graph_data, summary_data, text_data =  data[0], data[1], data[2]
        (attn_mask,node_feat,in_degree,out_degree,path_data,dist) = all_to_device(graph_data, self.device)
        batched_data = (attn_mask,node_feat,in_degree,out_degree,path_data,dist)

        node_embeds, _ = self.graphormer(batched_data)
        node_atts = torch.ones(node_embeds.size()[:-1],dtype=torch.long).to(self.device)

        token_embeds, _ = text_data
        token_embeds = token_embeds.clone().to(self.device).to(torch.float32)
        token_embeds = self.text_proj(token_embeds)
        token_atts = torch.ones(token_embeds.size()[:-1],dtype=torch.long).to(self.device)

        alpha = 0.5
        alpha = 0.9
        alpha = 0

        if alpha == 0:
            mixup_embeds = node_embeds
            mixup_attn_mask = node_atts
        elif alpha == 1:
            mixup_embeds = token_embeds
            mixup_attn_mask = token_atts
        else:

            if node_embeds.size(1) < token_embeds.size(1):
                node_embeds = F.pad(node_embeds, (0, 0, 0, token_embeds.size(1) - node_embeds.size(1)), "constant", 0)
            else:
                token_embeds = F.pad(token_embeds, (0, 0, 0, node_embeds.size(1) - token_embeds.size(1)), "constant", 0)
            mixup_embeds = alpha*token_embeds + (1-alpha)*node_embeds
            mixup_attn_mask = torch.ones(mixup_embeds.size()[:-1],dtype=torch.long).to(self.device)

        summary_in = self.tokenizer(summary_data, return_tensors="pt", padding='longest', truncation=True, max_length=512).to(self.device)
        fusion_output = self.summary_enc.bert(input_ids=summary_in.input_ids,
                                            attention_mask=summary_in.attention_mask,
                                            encoder_hidden_states=mixup_embeds,
                                            encoder_attention_mask=mixup_attn_mask,
                                            return_dict=True,
                                            mode='multi_modal')
        fusion_embeds = F.normalize(self.sumamry_proj(fusion_output.last_hidden_state[:, 0, :]),dim=-1)
        return fusion_embeds
    
    def pretrain_forward(self, data):
        rtl_data, net_data = data
        graph_ori, graph_pos, graph_neg, batched_summary, batched_text_ori, batched_text_neg = rtl_data
        summary_ori, summary_pos, summary_neg = batched_summary[0], batched_summary[1], batched_summary[2]

        net_ori, net_neg = net_data

        summary_embeds_ori, summary_in_ori, summay_out_ori = self.summary_encode(summary_ori)
        summary_embeds_pos, _, _ = self.summary_encode(summary_pos)
        summary_embeds_neg, summary_in_neg, summay_out_neg = self.summary_encode(summary_neg)

        node_embeds_ori, graph_embeds_ori = self.graph_encode(graph_ori)
        _, graph_embeds_pos = self.graph_encode(graph_pos)
        node_embeds_neg, graph_embeds_neg = self.graph_encode(graph_neg)

        loss_gtmae = self.pretrain_task_gtmae(graph_ori)

        loss_cl = self.pretrain_task_cl(summary_embeds_ori, summary_embeds_pos, summary_embeds_neg,\
                                        graph_embeds_ori, graph_embeds_pos, graph_embeds_neg)

        # loss_mlm_graph = self.pretrain_task_mlm_graph(node_embeds_ori, summary_ori)

        # loss_mlm_text = self.pretrain_task_mlm_text(batched_text_ori, summary_ori)

        loss_mlm_mixup = self.pretrain_task_mlm_mixup(node_embeds_ori, batched_text_ori, summary_ori)

        loss_match, rtl_emb = self.pretrain_task_match(node_embeds_ori, node_embeds_neg,\
                                            batched_text_ori, batched_text_neg,\
                                          summary_in_ori, summary_in_neg,\
                                          summay_out_ori, summay_out_neg)

        loss_align = self.pretrain_task_align(rtl_emb, net_ori, net_neg)

        loss = 1.0*loss_cl + 0.1*loss_gtmae + 1.0*loss_mlm_mixup + 1.0*loss_match + 0.2*loss_align

        # self.accelerator.print(f"Loss: {loss.item()}, Loss_CL: {loss_cl.item()}, Loss_GTMAE: {loss_gtmae.item()}, Loss_MLM: {loss_mlm.item()}")

        return loss, loss_cl, loss_gtmae, loss_mlm_mixup, loss_match, loss_align


    def pretrain_task_cl(self, summary_embeds_ori, summary_embeds_pos, summary_embeds_neg,\
                          graph_embeds_ori, graph_embeds_pos, graph_embeds_neg):
        summary_embeds_ori, summary_embeds_pos, summary_embeds_neg = summary_embeds_ori.clone(), summary_embeds_pos.clone(), summary_embeds_neg.clone()
        graph_embeds_ori, graph_embeds_pos, graph_embeds_neg = graph_embeds_ori.clone(), graph_embeds_pos.clone(), graph_embeds_neg.clone()
        loss_cl_ss = self.loss_cl(summary_embeds_ori, summary_embeds_pos, summary_embeds_neg)
        loss_cl_gg = self.loss_cl(graph_embeds_ori, graph_embeds_pos, graph_embeds_neg)
        loss_cl_gs = self.loss_cl(graph_embeds_ori, summary_embeds_pos, summary_embeds_neg)
        loss_cl_sg = self.loss_cl(summary_embeds_ori, graph_embeds_pos, graph_embeds_neg)

        lambda_ss = 1.0
        lambda_gg = 1.0
        lambda_cm = 0.5

        loss_cl = lambda_ss * loss_cl_ss + lambda_gg * loss_cl_gg + lambda_cm *0.5*(loss_cl_gs + loss_cl_sg)

        return loss_cl

    def pretrain_task_gsm(self, node_embeds_ori, node_embeds_neg,\
                           summary_in_ori, summary_in_neg,\
                          summay_out_ori, summay_out_neg):

        bs = node_embeds_ori.size(0)

        # forward the positve graph-summary pair
        graph_node_embeds_ori = node_embeds_ori.clone()
        graph_node_atts_ori = torch.ones(node_embeds_ori.size()[:-1],dtype=torch.long).to(self.device)

        graph_node_embeds_neg = node_embeds_neg.clone()
        graph_node_atts_neg = torch.ones(node_embeds_neg.size()[:-1],dtype=torch.long).to(self.device)

        output_pos = self.summary_enc.bert(encoder_embeds = summay_out_ori.last_hidden_state, 
                                        attention_mask = summary_in_ori.attention_mask,
                                        encoder_hidden_states = graph_node_embeds_ori,
                                        encoder_attention_mask = graph_node_atts_ori,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        output_neg_sg = self.summary_enc.bert(encoder_embeds = summay_out_ori.last_hidden_state, 
                                        attention_mask = summary_in_ori.attention_mask,
                                        encoder_hidden_states = graph_node_embeds_neg,
                                        encoder_attention_mask = graph_node_atts_neg,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        output_neg_gs = self.summary_enc.bert(encoder_embeds = summay_out_neg.last_hidden_state, 
                                        attention_mask = summary_in_neg.attention_mask,
                                        encoder_hidden_states = graph_node_embeds_ori,
                                        encoder_attention_mask = graph_node_atts_ori,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg_sg.last_hidden_state[:,0,:], output_neg_gs.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.gsm_head(vl_embeddings)   
        gsm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(self.device) 

        loss_gsm = F.cross_entropy(vl_output, gsm_labels)   
        return loss_gsm

    def pretrain_task_match(self, node_embeds_ori, node_embeds_neg,\
                            batched_text_ori, batched_text_neg,\
                           summary_in_ori, summary_in_neg,\
                          summay_out_ori, summay_out_neg):

        bs = node_embeds_ori.size(0)

        # forward the positve mixup-summary pair
        graph_node_embeds_ori = node_embeds_ori.clone()
        token_embeds_ori, _ = batched_text_ori
        token_embeds_ori = token_embeds_ori.clone().to(self.device).to(torch.float32)
        token_embeds_ori = self.text_proj(token_embeds_ori)
        mixup_embeds_ori, mixup_atts_ori = self.mixup_embeds(0.5, graph_node_embeds_ori, token_embeds_ori)

        graph_node_embeds_neg = node_embeds_neg.clone()
        token_embeds_neg, _ = batched_text_neg
        token_embeds_neg = token_embeds_neg.clone().to(self.device).to(torch.float32)
        token_embeds_neg = self.text_proj(token_embeds_neg)
        mixup_embeds_neg, mixup_atts_neg = self.mixup_embeds(0.5, graph_node_embeds_neg, token_embeds_neg)

        

        output_pos = self.summary_enc.bert(encoder_embeds = summay_out_ori.last_hidden_state, 
                                        attention_mask = summary_in_ori.attention_mask,
                                        encoder_hidden_states = mixup_embeds_ori,
                                        encoder_attention_mask = mixup_atts_ori,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        output_neg_sg = self.summary_enc.bert(encoder_embeds = summay_out_ori.last_hidden_state, 
                                        attention_mask = summary_in_ori.attention_mask,
                                        encoder_hidden_states = mixup_embeds_neg,
                                        encoder_attention_mask = mixup_atts_neg,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        output_neg_gs = self.summary_enc.bert(encoder_embeds = summay_out_neg.last_hidden_state, 
                                        attention_mask = summary_in_neg.attention_mask,
                                        encoder_hidden_states = mixup_embeds_ori,
                                        encoder_attention_mask = mixup_atts_ori,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg_sg.last_hidden_state[:,0,:], output_neg_gs.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.gsm_head(vl_embeddings)   
        gsm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(self.device) 

        loss_gsm = F.cross_entropy(vl_output, gsm_labels)   
        return loss_gsm, output_pos.last_hidden_state[:, -1]

    def pretrain_task_gtmae(self, graph_data):
        num_nodes = graph_data[1].size(1)
        mask_rate = self.gt_config['mask_rate']
        num_mask = int(mask_rate * num_nodes)
        mask = np.hstack([
            np.zeros(num_nodes - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask = torch.Tensor(mask).bool()

        ### add Fasle at the beginning of mask tensor
        mask = torch.cat([torch.tensor([False]), mask])

        
        x_rec_masked, x_init_mased, graph_emb = self.mask_attr_prediction(graph_data, mask)
        x_init = x_init_mased.view(-1)
        x_rec = x_rec_masked.contiguous().view(-1)  # [n graphs, n nodes, n feature]->[n graphs * n nodes,  n feature]
        
        if x_init.shape[0] == 0:
            loss = torch.tensor(0.0)
            loss_item = {"loss": loss.item()}
            print("No masked nodes")
            input()
            return loss, loss_item, graph_emb
        
        loss_gtmae = self.loss_gtmae(x_rec, x_init)

        return loss_gtmae

    def mask_attr_prediction(self, batched_data, mask):

        (attn_mask,node_feat,in_degree,out_degree,path_data,dist) = all_to_device(batched_data, self.device)
        batched_data_encoder = (attn_mask,node_feat,in_degree,out_degree,path_data,dist)

        output, graph_emb = self.graphormer(batched_data_encoder, mask)
        output = self.encoder_to_decoder(output)

        mask = mask[1:] ## remove the first bit of mask tensor

        gold = node_feat[:, mask]
        num_masked = mask.nonzero().view(-1).shape[0]
        mask_token = nn.Parameter(torch.zeros(output.shape[0], num_masked, output.shape[2]))
        dec_input = torch.cat([output, mask_token.to(self.device)], dim=1)
        ### enumerate based on the output.shape[0]
        for idx in range(output.shape[0]):
            dec_output = self.gtmae_decoder(dec_input[idx])
            if idx == 0:
                dec_outputs = dec_output.unsqueeze(0)
            else:
                dec_outputs = torch.cat([dec_outputs, dec_output.unsqueeze(0)], dim=0)

        # dec_output = self.decoder(dec_input)
        dec_outputs = dec_outputs[:, -num_masked:, :]

        return dec_outputs, gold, graph_emb

    def pretrain_task_mlm_graph(self, node_embeds, summary_data):
        graph_node_embeds = node_embeds.clone()
        graph_node_atts = torch.ones(node_embeds.size()[:-1],dtype=torch.long).to(self.device)
        
        summary_in = self.tokenizer(summary_data, return_tensors="pt", padding='longest', truncation=True, max_length=512).to(self.device)
        summary_ids = summary_in.input_ids.clone()
        labels = summary_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(summary_ids, self.summary_enc.config.vocab_size, self.device, targets=labels,
                                    probability_matrix = probability_matrix)  

        mlm_output = self.summary_enc(input_ids, 
                                    attention_mask = summary_in.attention_mask,
                                    encoder_hidden_states = graph_node_embeds,
                                    encoder_attention_mask = graph_node_atts,      
                                    return_dict = True,
                                    labels = labels,   
                                    )                           
        loss_mlm = mlm_output.loss

        return loss_mlm
    

    def pretrain_task_mlm_text(self, batched_text, summary_data):
        token_embeds, cone_embeds = batched_text
        token_embeds = token_embeds.clone().to(self.device).to(torch.float32)
        token_embeds = self.text_proj(token_embeds)
        token_atts = torch.ones(token_embeds.size()[:-1],dtype=torch.long).to(self.device)
        
        summary_in = self.tokenizer(summary_data, return_tensors="pt", padding='longest', truncation=True, max_length=512).to(self.device)
        summary_ids = summary_in.input_ids.clone()
        labels = summary_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(summary_ids, self.summary_enc.config.vocab_size, self.device, targets=labels,
                                    probability_matrix = probability_matrix)  


        mlm_output = self.summary_enc(input_ids, 
                                    attention_mask = summary_in.attention_mask,
                                    encoder_hidden_states = token_embeds,
                                    encoder_attention_mask = token_atts,      
                                    return_dict = True,
                                    labels = labels,   
                                    )                           
        loss_mlm = mlm_output.loss

        return loss_mlm
    
    def pretrain_task_mlm_mixup(self, node_embeds, batched_text, summary_data):
        token_embeds, cone_embeds = batched_text
        token_embeds = token_embeds.clone().to(self.device).to(torch.float32)
        token_embeds = self.text_proj(token_embeds)
        node_embeds = node_embeds.clone()

        alpha = 0
        if alpha == 0:
            mixup_embeds = node_embeds
        elif alpha == 1:
            mixup_embeds = token_embeds
        else:
            if node_embeds.size(1) < token_embeds.size(1):
                node_embeds = F.pad(node_embeds, (0, 0, 0, token_embeds.size(1) - node_embeds.size(1)), "constant", 0)
            else:
                token_embeds = F.pad(token_embeds, (0, 0, 0, node_embeds.size(1) - token_embeds.size(1)), "constant", 0)
            mixup_embeds = alpha*token_embeds + (1-alpha)*node_embeds
        mixup_attn_mask = torch.ones(mixup_embeds.size()[:-1],dtype=torch.long).to(self.device)

        summary_in = self.tokenizer(summary_data, return_tensors="pt", padding='longest', truncation=True, max_length=512).to(self.device)
        summary_ids = summary_in.input_ids.clone()
        labels = summary_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(summary_ids, self.summary_enc.config.vocab_size, self.device, targets=labels,
                                    probability_matrix = probability_matrix)  
 
        mlm_output = self.summary_enc(input_ids, 
                                    attention_mask = summary_in.attention_mask,
                                    encoder_hidden_states = mixup_embeds,
                                    encoder_attention_mask = mixup_attn_mask,      
                                    return_dict = True,
                                    labels = labels,   
                                    )                           
        loss_mlm = mlm_output.loss

        return loss_mlm


    
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        
    
    def mixup_embeds(self, alpha, graph_embeds, text_embeds):

        if graph_embeds.size(1) < text_embeds.size(1):
            graph_embeds = F.pad(graph_embeds, (0, 0, 0, text_embeds.size(1) - graph_embeds.size(1)), "constant", 0)
        else:
            text_embeds = F.pad(text_embeds, (0, 0, 0, graph_embeds.size(1) - text_embeds.size(1)), "constant", 0)

        if alpha == 0:
            mixup_embeds = graph_embeds
        elif alpha == 1:
            mixup_embeds = text_embeds
        else:
            mixup_embeds = alpha*text_embeds + (1-alpha)*graph_embeds
        
        mixup_attn_mask = torch.ones(mixup_embeds.size()[:-1],dtype=torch.long).to(self.device)

        return mixup_embeds, mixup_attn_mask
    
    def pretrain_task_align(self, rtl_embed, net_graph_ori, net_graph_neg):

        net_embeds_pos = self.net_enc(net_graph_ori, mode='infer')
        net_embeds_neg = self.net_enc(net_graph_neg, mode='infer')
        loss = self.loss_cl(rtl_embed, net_embeds_pos, net_embeds_neg)
        return loss