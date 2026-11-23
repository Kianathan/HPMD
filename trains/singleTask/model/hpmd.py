import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from ...singleTask.model.router import LogitRouter

class HPMD(nn.Module):
    def __init__(self, args):
        super(HPMD, self).__init__()
        # if args.use_bert:
        #     self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
        #                                     pretrained=args.pretrained)
        self.use_bert = args.use_bert               #true
        dst_feature_dims, nheads = args.dst_feature_dim_nheads     # [256, 8]
        if args.dataset_name == 'abide':     #true
            self.len_pheno = self.len_anat = self.len_func  = self.len_corr = 1
            
            
        self.aligned = args.need_data_aligned
        self.orig_d_pheno, self.orig_d_anat, self.orig_d_func, self.orig_d_corr = args.feature_dims     #48,6,10,512
        self.d_pheno = self.d_anat = self.d_func = self.d_corr = dst_feature_dims                   #256
        self.num_heads = nheads                                             #8
        self.layers = args.nlevels                                          #4
        self.attn_dropout_p   = args.attn_dropout_p      #0.2,attn_dropout
        self.attn_dropout_a = args.attn_dropout_a    #0.2
        self.attn_dropout_f = args.attn_dropout_f    #0.2args.attn_dropout_v
        self.attn_dropout_c = args.attn_dropout_c    #0.3
        self.relu_dropout = args.relu_dropout                               #0.0
        self.embed_dropout = args.embed_dropout                             #0.2
        self.res_dropout = args.res_dropout                                 #0.0
        self.output_dropout = args.output_dropout                           #0.5
        self.text_dropout = args.text_dropout                               #0.5
        self.attn_mask = args.attn_mask                                     #TRUE
        self.fusion_method = args.fusion_method                             #sum
        output_dim = 2
        self.args = args

        self.proj_pheno = nn.Conv1d(self.orig_d_pheno, self.d_pheno, kernel_size=args.conv1d_kernel_size_p, padding=0, bias=False)
        self.proj_anat = nn.Conv1d(self.orig_d_anat, self.d_anat, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_func = nn.Conv1d(self.orig_d_func, self.d_func, kernel_size=args.conv1d_kernel_size_f, padding=0, bias=False)
        self.proj_corr = nn.Conv1d(self.orig_d_corr, self.d_corr, kernel_size=args.conv1d_kernel_size_c, padding=0, bias=False)
        
        self.encoder_c = nn.Conv1d(self.d_pheno, self.d_pheno, kernel_size=1, padding=0, bias=False)

        self.encoder_pheno = nn.Conv1d(self.d_pheno, self.d_pheno, kernel_size=1, padding=0, bias=False)
        self.encoder_anat = nn.Conv1d(self.d_anat, self.d_anat, kernel_size=1, padding=0, bias=False)
        self.encoder_func = nn.Conv1d(self.d_func, self.d_func, kernel_size=1, padding=0, bias=False)
        self.encoder_corr = nn.Conv1d(self.d_corr, self.d_corr, kernel_size=1, padding=0, bias=False)

        self.self_attentions_pheno = self.get_network(self_type='p')
        self.self_attentions_anat = self.get_network(self_type='a')
        self.self_attentions_func = self.get_network(self_type='f')
        self.self_attentions_corr = self.get_network(self_type='c')
       

        self.proj1_pheno = nn.Linear(self.d_pheno, self.d_pheno)
        self.proj2_pheno = nn.Linear(self.d_pheno, self.d_pheno)
        self.out_layer_pheno = nn.Linear(self.d_pheno, output_dim)
        self.proj1_anat = nn.Linear(self.d_anat, self.d_anat)
        self.proj2_anat = nn.Linear(self.d_anat, self.d_anat)
        self.out_layer_anat = nn.Linear(self.d_anat, output_dim)
        self.proj1_func = nn.Linear(self.d_func, self.d_func)
        self.proj2_func = nn.Linear(self.d_func, self.d_func)
        self.out_layer_func = nn.Linear(self.d_func, output_dim)
        self.proj1_corr = nn.Linear(self.d_corr, self.d_corr)
        self.proj2_corr = nn.Linear(self.d_corr, self.d_corr)
        self.out_layer_corr = nn.Linear(self.d_corr, output_dim)
       

        if self.fusion_method == "sum":     
            self.proj1_c = nn.Linear(self.d_anat, self.d_anat)
            self.proj2_c = nn.Linear(self.d_anat, self.d_anat)
            self.out_layer_c = nn.Linear(self.d_anat, output_dim)
        elif self.fusion_method == "concat":
            self.proj1_c = nn.Linear(self.d_anat*4, self.d_anat*4)
            self.proj2_c = nn.Linear(self.d_anat*4, self.d_anat*4)
            self.out_layer_c = nn.Linear(self.d_anat*4, output_dim)

        self.Router = LogitRouter(output_dim, self.args.alpha, 4, self.args.temperature)

    def get_network(self, self_type='p', layers=-1):
        if self_type == 'p':
            embed_dim, attn_dropout = self.d_pheno, self.attn_dropout_p
        elif self_type == 'a':
            embed_dim, attn_dropout = self.d_anat, self.attn_dropout_a
        elif self_type == 'f':
            embed_dim, attn_dropout = self.d_func, self.attn_dropout_f
        elif self_type == 'c':
            embed_dim, attn_dropout = self.d_corr, self.attn_dropout_c
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def get_net(self, name):
        return getattr(self, name)
      
    def forward(self, pheno, anat, func, corr, y_true=None):
        # if self.use_bert:
        #     text = self.text_model(text)
        # x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_pheno = pheno.transpose(1, 2)
        x_anat = anat.transpose(1, 2)
        x_func = func.transpose(1, 2)
        x_corr = corr.transpose(1, 2)

        proj_x_pheno = x_pheno if self.orig_d_pheno == self.d_pheno else self.proj_pheno(x_pheno)
        proj_x_anat = x_anat if self.orig_d_anat == self.d_anat else self.proj_anat(x_anat)
        proj_x_func = x_func if self.orig_d_func == self.d_func else self.proj_func(x_func)
        proj_x_corr = x_corr if self.orig_d_corr == self.d_corr else self.proj_corr(x_corr)

        c_pheno = self.encoder_pheno(proj_x_pheno)
        c_anat = self.encoder_anat(proj_x_anat)
        c_func = self.encoder_func(proj_x_func)
        c_corr = self.encoder_corr(proj_x_corr)

        c_pheno = c_pheno.permute(2, 0, 1)
        c_anat = c_anat.permute(2, 0, 1)
        c_func = c_func.permute(2, 0, 1)
        c_corr = c_corr.permute(2, 0, 1)

        c_pheno_att = self.self_attentions_pheno(c_pheno)
        if type(c_pheno_att) == tuple:
            c_pheno_att = c_pheno_att[0]
        c_pheno_att = c_pheno_att[-1]
        c_anat_att = self.self_attentions_anat(c_anat)
        if type(c_anat_att) == tuple:
            c_anat_att = c_anat_att[0]
        c_anat_att = c_anat_att[-1]
        c_func_att = self.self_attentions_func(c_func)
        if type(c_func_att) == tuple:
            c_func_att = c_func_att[0]
        c_func_att = c_func_att[-1]
        c_corr_att = self.self_attentions_corr(c_corr)
        if type(c_corr_att) == tuple:
            c_corr_att = c_corr_att[0]
        c_corr_att = c_corr_att[-1]

        pheno_proj = self.proj2_pheno(
            F.dropout(F.relu(self.proj1_pheno(c_pheno_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        pheno_proj += c_pheno_att
        logits_pheno = self.out_layer_pheno(pheno_proj)
        anat_proj = self.proj2_anat(
            F.dropout(F.relu(self.proj1_anat(c_anat_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        anat_proj += c_anat_att
        logits_anat = self.out_layer_anat(anat_proj)
        func_proj = self.proj2_func(
            F.dropout(F.relu(self.proj1_func(c_func_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        func_proj += c_func_att
        logits_func = self.out_layer_func(func_proj)
        corr_proj = self.proj2_corr(
            F.dropout(F.relu(self.proj1_corr(c_corr_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        corr_proj += c_corr_att
        logits_corr = self.out_layer_corr(corr_proj)
        
        if self.training:
            m_w, L_loss = self.Router(logits_pheno, logits_anat, logits_func, logits_corr, y_true)
        else:
            router_out = self.Router(logits_pheno, logits_anat, logits_func, logits_corr)
            m_w = router_out
            L_loss = torch.tensor(0.0, device=logits_pheno.device, dtype=logits_pheno.dtype)
        
        if self.fusion_method == "sum": 
            for i in range(m_w.shape[0]):
                c_f = c_pheno_att[i] * m_w[i][0] + c_anat_att[i] * m_w[i][1] + c_func_att[i] * m_w[i][2] + c_corr_att[i] * m_w[i][3]
                # c_f = c_anat_att[i] * m_w[i][0] + c_func_att[i] * m_w[i][1] + c_corr_att[i] * m_w[i][2]
                if i == 0: c_fusion = c_f.unsqueeze(0)
                else: c_fusion = torch.cat([c_fusion, c_f.unsqueeze(0)], dim=0)   
        elif self.fusion_method == "concat":        
            for i in range(m_w.shape[0]):
                c_f = torch.cat([c_pheno_att[i] * m_w[i][0], c_anat_att[i] * m_w[i][1], c_func_att[i] * m_w[i][2], c_corr_att[i] * m_w[i][3]], dim=0) * 4
                if i == 0: c_fusion = c_f.unsqueeze(0)
                else: c_fusion = torch.cat([c_fusion, c_f.unsqueeze(0)], dim=0)   


        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)
         
        res = {
            'logits_c': logits_c,    
            'logits_pheno': logits_pheno,    
            'logits_anat': logits_anat,   
            'logits_func': logits_func,   
            'logits_corr': logits_corr,    
            'channel_weight': m_w,   
            'c_proj': c_proj,        
            'pheno_proj': pheno_proj,        
            'anat_proj': anat_proj,         
            'func_proj': func_proj,       
            'corr_proj': corr_proj,       
            'c_fea': c_fusion,      
            'L_loss':L_loss
        }
        return res