import os
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_normal_



class MoELayer(nn.Module):
    def __init__(self, input_size=128, num_experts=5, hidden_size=64, dropout=0.2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        
        self.total_input_size = input_size * 8 + 16   # 8个embedding拼接后的总维度, 加上职类向量
        self.gate = nn.Sequential(
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(8, num_experts),
                    nn.Softmax(dim=1)
        )
        
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),  
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            ) for _ in range(num_experts)
        ])
        
        
       
    def forward(self, user_emb, item_emb, f_i_1, g_i_1, f_i_2, g_i_2, f_i_3, g_i_3, cate_emb):
        combined = torch.cat([user_emb, item_emb, f_i_1, g_i_1, f_i_2, g_i_2, f_i_3, g_i_3, cate_emb], dim=1)
        gates = self.gate(cate_emb)     
        expert_outputs = torch.zeros(combined.shape[0], 1).to(combined.device)
        for i, expert in enumerate(self.experts):
            expert_out = expert(combined)
            expert_outputs += gates[:, i:i+1] * expert_out
            
        return expert_outputs.squeeze()
    
    
class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.attention2 = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 融合层
        self.fusion = nn.Linear(input_dim * 2, input_dim)
        
    def forward(self, origin, target):
        attn_out, _ = self.attention(origin, target, target)
        out = self.norm(attn_out + target) # 残差连接
        
        attn_out_2, _ = self.attention2(target, origin, origin)
        out_2 = self.norm2(attn_out_2 + origin) # 残差连接
        
        concat_out = torch.cat([out, out_2], dim=-1)
        
        concat_out = self.fusion(concat_out)
        
        out = self.dropout(concat_out)
        
        out = self.fc(out)

        out = torch.mean(out, dim=1) 
        return out

    
class OurPJF(nn.Module):
    def __init__(self):
        super(OurPJF, self).__init__()
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
        # BGE-1024
        self.embedding_size = 1024

        self.job_bert_lr = nn.Linear(1024, 128)
        self.geek_bert_lr = nn.Linear(1024, 128)
        
        self.job_hist_bert_lr = nn.Linear(1024, 128)
        self.geek_hist_bert_lr = nn.Linear(1024, 128)
        
        self.cate_emb = nn.Embedding(10000, 8)
        
        self.job_attn_1 = AttentionAggregator(
            input_dim=128,
            hidden_dim=self.embedding_size
        )
        
        self.job_attn_2 = AttentionAggregator(
            input_dim=128,
            hidden_dim=self.embedding_size
        )
        
        self.job_attn_3 = AttentionAggregator(
            input_dim=128,
            hidden_dim=self.embedding_size
        )
        
        self.job_attn_4 = AttentionAggregator(
            input_dim=128,
            hidden_dim=self.embedding_size
        )
    
        self.job_attn_5 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
        
        self.job_attn_6 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
            
        self.geek_attn_1 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
    
        self.geek_attn_2 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
        
        self.geek_attn_3 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
        
        self.geek_attn_4 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
    
        self.geek_attn_5 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
        
        self.geek_attn_6 = AttentionAggregator(
            input_dim=128 ,
            hidden_dim=self.embedding_size
        )
        
        self.job_layer_1 = nn.Linear(self.embedding_size , 128)
        self.job_layer_2 = nn.Linear(self.embedding_size , 128)
        self.job_layer_3 = nn.Linear(self.embedding_size , 128)
        
        self.job_layer_4 = nn.Linear(self.embedding_size , 128)
        self.job_layer_5 = nn.Linear(self.embedding_size , 128)
        self.job_layer_6 = nn.Linear(self.embedding_size , 128)
        
        self.geek_layer_1 = nn.Linear(self.embedding_size , 128)
        self.geek_layer_2 = nn.Linear(self.embedding_size , 128)
        self.geek_layer_3 = nn.Linear(self.embedding_size , 128)
        
        self.geek_layer_4 = nn.Linear(self.embedding_size , 128)
        self.geek_layer_5 = nn.Linear(self.embedding_size , 128)
        self.geek_layer_6 = nn.Linear(self.embedding_size , 128)
        
        self.sigmoid = nn.Sigmoid()
        self._init_weights(nn.Embedding)
        
        self.job_hist_1_hidden = nn.Linear(128*2, 128)
        self.job_hist_2_hidden = nn.Linear(128*2, 128)
        self.job_hist_3_hidden = nn.Linear(128*2, 128)
        
        self.geek_hist_1_hidden = nn.Linear(128*2, 128)
        self.geek_hist_2_hidden = nn.Linear(128*2, 128)
        self.geek_hist_3_hidden = nn.Linear(128*2, 128)
                
        self.moe_layer = MoELayer(input_size=128, num_experts=5, hidden_size=64)
         
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def _get_fg_E_user(self, user_vec):
        
        f_e = self.geek_bert_lr(user_vec)
        
        return f_e

    def _get_fg_E_item(self, item_vec):

        g_e = self.job_bert_lr(item_vec)
      
        return g_e

    def _forward_E(self, user_vec, item_vec):
        f_e = self._get_fg_E_user(user_vec)
        g_e = self._get_fg_E_item(item_vec)
        score_E = torch.mul(f_e, g_e).sum(dim=1)

        return score_E

    def _get_fg_I_user_1(self, item_vec, his_arr_eval_item_vec):
        f_e = self._get_fg_E_item(item_vec) 
        his_g_e = self.geek_hist_bert_lr(his_arr_eval_item_vec)  
        f_i = self.geek_attn_1(f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1), his_g_e)
        f_i = self.geek_layer_1(f_i)

        return f_i
    
    def _get_fg_I_user_2(self, item_vec, his_eval_pass_item_vec):
        f_e = self._get_fg_E_item(item_vec)  
        his_g_e = self.geek_hist_bert_lr(his_eval_pass_item_vec)  
        f_i = self.geek_attn_2(f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1), his_g_e)
        f_i = self.geek_layer_2(f_i)

        return f_i
    
    def _get_fg_I_user_3(self, item_vec, his_intv_pass_item_vec):
        f_e = self._get_fg_E_item(item_vec) 
        his_g_e = self.geek_hist_bert_lr(his_intv_pass_item_vec) 
        f_i = self.geek_attn_3(f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1), his_g_e)
        f_i = self.geek_layer_3(f_i)
        
        return f_i
    
    def _get_fg_I_user_4(self, user_vec, his_arr_eval_item_vec):
        f_e = self._get_fg_E_user(user_vec) 
        his_g_e = self.geek_hist_bert_lr(his_arr_eval_item_vec) 
        f_i = self.geek_attn_4(f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1), his_g_e)
        f_i = self.geek_layer_4(f_i)

        return f_i
    
    def _get_fg_I_user_5(self, user_vec, his_eval_pass_item_vec):
        f_e = self._get_fg_E_user(user_vec)  
        his_g_e = self.geek_hist_bert_lr(his_eval_pass_item_vec)  
        f_i = self.geek_attn_5(f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1), his_g_e)
        f_i = self.geek_layer_5(f_i)

        return f_i
    
    def _get_fg_I_user_6(self, user_vec, his_intv_pass_item_vec):
        f_e = self._get_fg_E_user(user_vec)  
        his_g_e = self.geek_hist_bert_lr(his_intv_pass_item_vec)  
        f_i = self.geek_attn_6(f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1), his_g_e)
        f_i = self.geek_layer_6(f_i)
        
        return f_i
        
        

    def _get_fg_I_item_1(self, user_vec, his_arr_eval_users_vec):
        g_e = self._get_fg_E_user(user_vec)
        his_f_e = self.job_hist_bert_lr(his_arr_eval_users_vec)  
        g_i = self.job_attn_1(g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1), his_f_e)
        g_i = self.job_layer_1(g_i)

        return g_i
    
    def _get_fg_I_item_2(self, user_vec, his_eval_pass_users_vec):
        g_e = self._get_fg_E_user(user_vec)
        his_f_e = self.job_hist_bert_lr(his_eval_pass_users_vec)  
        g_i = self.job_attn_2(g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1), his_f_e)
        g_i = self.job_layer_2(g_i)

        return g_i
    
    def _get_fg_I_item_3(self, user_vec, his_intv_pass_users_vec):
        g_e = self._get_fg_E_user(user_vec)
        his_f_e = self.job_hist_bert_lr(his_intv_pass_users_vec)  
        g_i = self.job_attn_3(g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1), his_f_e)
        g_i = self.job_layer_3(g_i)

        return g_i
    
    def _get_fg_I_item_4(self, item_vec, his_arr_eval_users_vec):
        g_e = self._get_fg_E_item(item_vec)
        his_f_e = self.job_hist_bert_lr(his_arr_eval_users_vec)  
        g_i = self.job_attn_4(g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1), his_f_e)
        g_i = self.job_layer_4(g_i)

        return g_i
    
    def _get_fg_I_item_5(self, item_vec, his_eval_pass_users_vec):
        g_e = self._get_fg_E_item(item_vec)
        his_f_e = self.job_hist_bert_lr(his_eval_pass_users_vec)  
        g_i = self.job_attn_5(g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1), his_f_e)
        g_i = self.job_layer_5(g_i)

        return g_i
    
    def _get_fg_I_item_6(self, item_vec, his_intv_pass_users_vec):
        g_e = self._get_fg_E_item(item_vec)
        his_f_e = self.job_hist_bert_lr(his_intv_pass_users_vec)  
        g_i = self.job_attn_6(g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1), his_f_e)
        g_i = self.job_layer_6(g_i)

        return g_i

        
    def _forward_I_1(self, user_vec, item_vec, his_arr_eval_item_vec, his_arr_eval_users_vec):
        
        f_i_1 = self._get_fg_I_user_1(item_vec, his_arr_eval_item_vec) 
        g_i_1 = self._get_fg_I_item_1(user_vec, his_arr_eval_users_vec) 
        
        g_g_1 = self._get_fg_I_user_4(user_vec, his_arr_eval_item_vec)
        f_f_1 = self._get_fg_I_item_4(item_vec, his_arr_eval_users_vec)
        
        f_hist_1 = torch.cat([f_i_1, f_f_1], dim=-1)
        f_i_1 = self.job_hist_1_hidden(f_hist_1)
        
        g_hist_1 = torch.cat([g_i_1, g_g_1], dim=-1)
        g_i_1 = self.geek_hist_1_hidden(g_hist_1)

        return f_i_1, g_i_1
    
    def _forward_I_2(self, user_vec, item_vec, his_eval_pass_item_vec, his_eval_pass_users_vec):
        
        f_i_2 = self._get_fg_I_user_2(item_vec, his_eval_pass_item_vec) 
        g_i_2 = self._get_fg_I_item_2(user_vec, his_eval_pass_users_vec)
        
        g_g_2 = self._get_fg_I_user_5(user_vec, his_eval_pass_item_vec)
        f_f_2 = self._get_fg_I_item_5(item_vec, his_eval_pass_users_vec)
        
        f_hist_2 = torch.cat([f_i_2, f_f_2], dim=-1)
        f_i_2 = self.job_hist_2_hidden(f_hist_2)
        
        g_hist_2 = torch.cat([g_i_2, g_g_2], dim=-1)
        g_i_2 = self.geek_hist_2_hidden(g_hist_2)

        return f_i_2, g_i_2
    
    def _forward_I_3(self, user_vec, item_vec, his_intv_pass_item_vec, his_intv_pass_users_vec):
        
        f_i_3 = self._get_fg_I_user_3(item_vec, his_intv_pass_item_vec) 
        g_i_3 = self._get_fg_I_item_3(user_vec, his_intv_pass_users_vec) 
        
        g_g_3 = self._get_fg_I_user_6(user_vec, his_intv_pass_item_vec)
        f_f_3 = self._get_fg_I_item_6(item_vec, his_intv_pass_users_vec)
        
        f_hist_3 = torch.cat([f_i_3, f_f_3], dim=-1)
        f_i_3 = self.job_hist_3_hidden(f_hist_3)
        
        g_hist_3 = torch.cat([g_i_3, g_g_3], dim=-1)
        g_i_3 = self.geek_hist_3_hidden(g_hist_3)
        return f_i_3, g_i_3
    

    def calculate_loss(self, user_vec, item_vec, his_arr_eval_item_vec, his_eval_pass_item_vec, his_intv_pass_item_vec,\
                       his_arr_eval_users_vec, his_eval_pass_users_vec, his_intv_pass_users_vec,\
                        neg_user_vec, neg_his_arr_eval_item_vec, neg_his_eval_pass_item_vec, neg_his_intv_pass_item_vec,\
                      user_cate, item_cate, neg_user_cate):
        
        # 职类向量获取
        user_cate_emb = self.cate_emb(user_cate).squeeze(1)
        
        item_cate_emb = self.cate_emb(item_cate).squeeze(1)
        
        cate_emb = torch.cat([user_cate_emb, item_cate_emb], dim=1)
        
        f_i_1, g_i_1 = self._forward_I_1(user_vec, item_vec, his_arr_eval_item_vec, his_arr_eval_users_vec)
        f_i_2, g_i_2 = self._forward_I_2(user_vec, item_vec, his_eval_pass_item_vec, his_eval_pass_users_vec)
        f_i_3, g_i_3 = self._forward_I_3(user_vec, item_vec, his_intv_pass_item_vec, his_intv_pass_users_vec)
        
        user_emb = self._get_fg_E_user(user_vec)
        item_emb = self._get_fg_E_item(item_vec)
        
        pos_score = self.moe_layer(user_emb, item_emb, f_i_1, g_i_1, f_i_2, g_i_2, f_i_3, g_i_3, cate_emb)
       
                
        neg_user_cate_emb = self.cate_emb(neg_user_cate).squeeze(1)
        
        neg_cate_emb = torch.cat([neg_user_cate_emb, item_cate_emb], dim=1)
        
        neg_score_E = self._forward_E(neg_user_vec, item_vec)
        neg_f_i_1, neg_g_i_1  = self._forward_I_1(neg_user_vec, item_vec, neg_his_arr_eval_item_vec, his_arr_eval_users_vec)
        neg_f_i_2, neg_g_i_2 = self._forward_I_2(neg_user_vec, item_vec, neg_his_eval_pass_item_vec, his_eval_pass_users_vec)
        neg_f_i_3, neg_g_i_3 = self._forward_I_3(neg_user_vec, item_vec, neg_his_intv_pass_item_vec, his_intv_pass_users_vec)
        
        neg_user_emb = self._get_fg_E_user(neg_user_vec)
            
        neg_score = self.moe_layer(neg_user_emb, item_emb, neg_f_i_1, neg_g_i_1, neg_f_i_2, neg_g_i_2, neg_f_i_3, neg_g_i_3, neg_cate_emb)
        
        # 计算BPR损失
        
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean() + \
       0.1 * (torch.pow(pos_score, 2).mean() + torch.pow(neg_score, 2).mean())

        return loss


    def predict(self, user_vec, item_vec, his_arr_eval_item_vec, his_eval_pass_item_vec, his_intv_pass_item_vec,\
                       his_arr_eval_users_vec, his_eval_pass_users_vec, his_intv_pass_users_vec, \
                        user_cate, item_cate, type='dev'):
        if type == 'dev':
            
            # 计算_forward_E
            score_E = self._forward_E(user_vec, item_vec)
            
            user_cate_emb = self.cate_emb(user_cate).squeeze(1)
            
            item_cate_emb = self.cate_emb(item_cate).squeeze(1)
            
            cate_emb = torch.cat([user_cate_emb, item_cate_emb], dim=1)
            
            f_i_1, g_i_1 = self._forward_I_1(user_vec, item_vec, his_arr_eval_item_vec, his_arr_eval_users_vec)
            f_i_2, g_i_2 = self._forward_I_2(user_vec, item_vec, his_eval_pass_item_vec, his_eval_pass_users_vec)
            f_i_3, g_i_3 = self._forward_I_3(user_vec, item_vec, his_intv_pass_item_vec, his_intv_pass_users_vec)

            user_emb = self._get_fg_E_user(user_vec)
            item_emb = self._get_fg_E_item(item_vec)
            
            score = self.moe_layer(user_emb, item_emb, f_i_1, g_i_1, f_i_2, g_i_2, f_i_3, g_i_3, cate_emb)
            
            return self.sigmoid(score)

        elif type == 'talent':
            
            f_e = self._get_fg_E_user(user_vec)

            return f_e
        elif type == 'post':

            g_e = self._get_fg_E_item(item_vec)

            return g_e
