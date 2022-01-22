import copy
import math
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def future_mask(seq_length, k=1):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=k).astype('bool')
    return torch.from_numpy(future_mask)


def clone(module, num):
    """Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.weight[:, :x.size(1), :]
        return self.dropout(x)



class CosinePositionalEmbedding(nn.Module):
    """Implement the PE function.
    """
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(CosinePositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.weight[:, :x.size(1), :]
        return self.dropout(x)


class YKT(nn.Module):
    def __init__(self, num_items, num_skills, max_length, embed_size, num_attn_layers, num_heads,
                 rel_pos, max_pos, drop_prob, no_bert, dataset):
        """Relation-aware Self-attentive Mechanism
        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            max_length (int): max sequence length
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            rel_pos (bool): if True, predict performance using relative position embeddings
            max_pos (int): max distance
            drop_prob (float): dropout probability
            use_bert: if True, utilize textual content embedding by Bert
        """
        super(YKT, self).__init__()
        self.num_items = num_items
        self.max_length = max_length
        self.embed_size = embed_size
        self.rel_pos = rel_pos
        self.no_bert = no_bert

        if self.no_bert:
            self.item_embeds = nn.Embedding(num_items + 1, embed_size, padding_idx=0)
        else:
            # The parameter is written, not flexible enough, follow-up improvement
            bert_file = './data/' + dataset + '_question_bert.json'
            # eanalyst_question_bert.json、junyi、poj_question_bert.json
            # question_bert_dict = open(bert_file, 'r', encoding='utf-8')
            weight = []
            weight.append(np.zeros(768))
            with open(bert_file, 'r') as f:
                question_bert_dict = json.load(f)
                for value in question_bert_dict.values():
                    weight.append(np.array(value))
            big_weight = torch.FloatTensor(weight)
            self.item_embeds = nn.Embedding.from_pretrained(embeddings=big_weight, padding_idx=0)
            self.after_qembed = nn.Linear(768, embed_size)

        self.skill_embeds = nn.Embedding(num_skills+1, embed_size, padding_idx=0)
        self.skill_embed_diff = nn.Embedding(num_skills+1, embed_size, padding_idx=0)
        self.difficult_param = nn.Embedding(num_items+1, 1)

        # Absolute position encoding
        self.pe = CosinePositionalEmbedding(embed_size, drop_prob)

        # Learnable position encoding
        # self.pe = LearnablePositionalEmbedding(embed_size, drop_prob)

        # relative position encoding
        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        # Cognitive feature fusers: memory (forgetting+learning), praxis (knowledge state), language (exercise-relation)
        self.smfe = SMFE(max_length, embed_size, num_heads, drop_prob)
        self.erfe = ERFE(embed_size, num_heads, drop_prob)
        self.krfe = KRFE(num_items, num_skills, embed_size, num_heads, drop_prob)
        # ……other feature encoder

        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.lin_in = clone(nn.Linear(2*embed_size, embed_size), 4)
        self.lin_in2 = nn.Linear(3*embed_size, embed_size)

        # feed forward layer
        self.ffn = FFN(embed_size, 2*embed_size, drop_prob)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)

        self.layer_norm = nn.LayerNorm(embed_size)
        self.reset()

    def reset(self):
        for p in self.parameters():
            if (p.size(0) == self.num_items+1) and (p.size(1) == 1):
                torch.nn.init.constant_(p, 0.)

    def get_query(self, item_ids, skill_ids):
        item_embed = self.item_embeds(item_ids)
        if not self.no_bert:
            item_embed = self.after_qembed(item_embed)
        item_query = torch.cat([item_embed], dim=-1)

        skill_embed = self.skill_embeds(skill_ids)
        skill_diff = self.skill_embed_diff(skill_ids)
        difficult = self.difficult_param(item_ids)
        skill_embed = skill_embed + difficult * skill_diff
        skill_query = torch.cat([skill_embed], dim=-1)

        return item_query, skill_query

    def get_inputs(self, item_ids, skill_ids, labels):
        labels = labels.unsqueeze(-1).float()

        item_inputs_embed = self.item_embeds(item_ids)
        if not self.no_bert:
            item_inputs_embed = self.after_qembed(item_inputs_embed)
        item_inputs_embed = self.pe(item_inputs_embed)

        skill_inputs_embed = self.skill_embeds(skill_ids)
        skill_inputs_diff = self.skill_embed_diff(skill_ids)
        difficult = self.difficult_param(item_ids)
        skill_inputs_embed = skill_inputs_embed + difficult * skill_inputs_diff
        skill_inputs_embed = self.pe(skill_inputs_embed)

        item_inputs = torch.cat([item_inputs_embed, item_inputs_embed], dim=-1)
        item_inputs[..., :self.embed_size] *= labels
        item_inputs[..., self.embed_size:] *= 1 - labels

        skill_inputs = torch.cat([skill_inputs_embed, skill_inputs_embed], dim=-1)
        skill_inputs[..., :self.embed_size] *= labels
        skill_inputs[..., self.embed_size:] *= 1 - labels

        return item_inputs, skill_inputs


    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels, rel, ren_mat, timestamp):
        query, skill_query = self.get_query(item_ids, skill_ids)
        batch_size, embed_size = query.size(0), query.size(2)
        inputs, skill_inputs = self.get_inputs(item_ids, skill_ids, labels)
        inputs = F.relu(self.lin_in[0](inputs))
        skill_inputs = F.relu(self.lin_in[1](skill_inputs))

        mask = future_mask(inputs.size(-2), k=1)
        if inputs.is_cuda:
            mask = mask.cuda()

        smfe = self.smfe(query, inputs, ren_mat, timestamp, mask)
        erfe = self.erfe(query, inputs, rel, mask)
        krfe = self.krfe(skill_query, skill_inputs, mask)


        query = self.lin_in[2](torch.cat([query, skill_query], dim=-1))

        inputs = self.lin_in2(torch.cat([smfe, erfe, krfe], dim=-1))
        inputs = torch.cat((torch.zeros(batch_size, 1, embed_size, dtype=torch.float).cuda(), inputs), dim=1)[:, :-1, :]

        outputs, attn = self.attn_layers[0](query, inputs, inputs, mask=mask, rel_pos=self.rel_pos,
                                            pos_key_embeds=self.pos_key_embeds, pos_value_embeds=self.pos_value_embeds)
        outputs = self.dropout(outputs)

        # Layer norm and Residual connection
        for l in self.attn_layers[1:]:
            residual, attn = l(query, outputs, outputs, mask=mask, rel_pos=self.rel_pos,
                               pos_key_embeds=self.pos_key_embeds, pos_value_embeds=self.pos_value_embeds)
            outputs = self.layer_norm(outputs + self.ffn(residual))
        return self.lin_out(outputs), attn


class SMFE(nn.Module):
    def __init__(self, max_len, in_dim, n_head, dropout):
        super(SMFE, self).__init__()
        assert in_dim % n_head == 0
        self.n_head = n_head
        self.attention = MultiHeadedAttention(in_dim, n_head, dropout)
        self.lin_out1 = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(in_dim)

        self.l1 = nn.Parameter(torch.rand(1))
        self.gammas = nn.Parameter(torch.zeros(n_head, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, query, inputs, ren_mat, timestamp, mask):
        batch_size, seq_length = query.size(0), query.size(1)
        gamma = self.gammas.unsqueeze(0).repeat(batch_size, 1, seq_length, 1)
        out, attn = self.attention(inputs, inputs, inputs, mask=mask, l=self.l1, ren_mat=ren_mat, timestamp=timestamp,
                                   gamma=gamma)
        out = out + self.dropout(self.lin_out1(out))
        out = self.layernorm(out)
        return out

class KRFE(nn.Module):
    """
    Followed by layer norm and postion-wise feed forward net and dropout layer.
    """
    def __init__(self, num_items, num_skills, in_dim, n_head, dropout):
        super(KRFE, self).__init__()
        assert in_dim % n_head == 0
        self.n_head = n_head
        self.attention = MultiHeadedAttention(in_dim, n_head, dropout)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.lin_out = nn.Linear(in_dim, in_dim)

    def forward(self, skill_query, skill_inputs, mask):
        """
        Input:
            query: skill embeded; inputs: skill-answer embeded; mask: peek only current and pas values
        Output:
            knowledge state feature encoded
        """
        out, attn = self.attention(skill_inputs, skill_inputs, skill_inputs, mask=mask, kr=True)
        out = out + self.dropout(self.lin_out(out))
        out = self.layer_norm(out)
        return out


class ERFE(nn.Module):
    def __init__(self, in_dim, n_head, dropout):
        super(ERFE, self).__init__()
        assert in_dim % n_head == 0
        self.n_head = n_head
        self.attention = MultiHeadedAttention(in_dim, n_head, dropout)
        self.lin_out = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(in_dim)

        self.l2 = nn.Parameter(torch.rand(1))

    def forward(self, query, inputs, rel, mask):
        out, attn = self.attention(inputs, inputs, inputs, mask=mask, l=self.l2, rel=rel)
        out = out + self.dropout(self.lin_out(out))
        out = self.layernorm(out)
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

        self.m = nn.Softplus()
        self.lin_out = nn.Linear(total_size, total_size)

    def forward(self, query, key, value, mask=None, qk_same=False, l=None, ren_mat=None, timestamp=None, gamma=None,
                kr=False, rel=None, rel_pos=False, pos_key_embeds=None, pos_value_embeds=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        if qk_same:
            query = self.linear_layers[0](query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
            key = self.linear_layers[0](key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
            value = self.linear_layers[1](value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        else:
            query, key, value = [linear(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                                 for linear, x in zip(self.linear_layers,
                                                 (query, key,
                                                  value))]
        if gamma is not None:
            gamma = self.m(gamma)
            timestamp = timestamp.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            ren_mat = ren_mat.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            out, self.prob_attn = attention(query, key, value, mask=mask, dropout=self.dropout, l=l, ren_mat=ren_mat,
                                            timestamp=timestamp, memory_gamma=gamma)

        if kr:
            out, self.prob_attn = attention(query, key, value, mask=mask, dropout=self.dropout, kr=kr)

        if rel is not None:
            rel = rel.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            out, self.prob_attn = attention(query, key, value, mask=mask, dropout=self.dropout, l=l, rel=rel)

        # Apply relation-aware self-attention
        if rel_pos:
            out, self.prob_attn = relative_attention(query, key, value, mask=mask, dropout=self.dropout,
                                                     pos_key_embeds=pos_key_embeds, pos_value_embeds=pos_value_embeds)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)

        out = self.lin_out(out)
        return out, self.prob_attn


def attention(query, key, value, mask=None, dropout=None, l=None, ren_mat=None, timestamp=None, memory_gamma=None,
              kr=False, rel=None):
    """Compute scaled dot product attention.
       query, key, value：(BS,num_heads,SL,head_size)
       mask: (1,1,SL,SL)
       l1: Memory coefficient (1,)
       timestamp: (BS,num_heads,SL,SL)
       memory_gamma: (BS,num_heads,SL,1)
       rel: (Batch_size,1,SL,SL)
    """
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
        prob_attn = F.softmax(scores, dim=-1)
    if memory_gamma is not None:
        ren_feat = torch.log10(ren_mat + 1)
        ren_feat = ren_feat.masked_fill(mask, -1e9)
        time_stamp = torch.exp(-torch.abs(timestamp.float()))
        time_stamp = time_stamp.masked_fill(mask, -1e9)

        mem_feat = time_stamp + ren_feat
        mem_attn = F.softmax(mem_feat, dim=-1)
        prob_attn = (1 - l) * prob_attn + l * mem_attn

    if kr:
        prob_attn = prob_attn

    if rel is not None:
        rel = rel.masked_fill(mask, -1e9)
        rel = rel.masked_fill(rel == 0, -1e9)
        rel_attn = nn.Softmax(dim=-1)(rel)
        prob_attn = (1 - l) * prob_attn + l * rel_attn

    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, mask=None, dropout=None, pos_key_embeds=None, pos_value_embeds=None):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    Parameters
        pos_key_embeds, pos_value_embeds: ( max_pos, embed_size//num_heads->(i.e. head_size) )
    """
    batch_size, seq_length = query.size(0), query.size(2)
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings

    scores = torch.matmul(query, key.transpose(-2, -1))

    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)

    prob_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        prob_attn = dropout(prob_attn)
    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)

    prob_attn = prob_attn.contiguous().view(batch_size, -1, seq_length, seq_length)
    return output, prob_attn

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super(FFN, self).__init__()
        self.lin_out1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU()
        self.lin_out2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin_out1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin_out2(x)
        return x
