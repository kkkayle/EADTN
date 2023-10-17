import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightNetwork(nn.Module):
    def __init__(self, filed_size, reduction_ratio=3):
        super(WeightNetwork, self).__init__()
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, filed_size, bias=False),
            nn.ReLU(),
        )

    def forward(self, inputs):
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=-1))
        return V

class EADTN(nn.Module):
    def __init__(self, args,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(EADTN, self).__init__()
        self.dim = args.char_dim
        self.conv = args.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = args.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = args.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = args.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = 5

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Re_Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        
        self.Re_Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        
        
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)


        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

        self.drug_weightnet=WeightNetwork(85)
        self.protein_weightnet=WeightNetwork(979)

    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)
        reweight_drug=self.drug_weightnet(drugConv.permute(0, 2, 1))
        reweight_protein=self.protein_weightnet(proteinConv.permute(0, 2, 1))
        
        drug_feats = self.Drug_max_pool(drugConv).squeeze(2)
        protein_feats = self.Protein_max_pool(proteinConv).squeeze(2)
        feats=torch.cat([drug_feats,protein_feats],dim=1)
        
        re_drug_feats = self.Re_Drug_max_pool(reweight_drug.permute(0, 2, 1)).squeeze(2)
        re_protein_feats = self.Re_Protein_max_pool(reweight_protein.permute(0, 2, 1)).squeeze(2)
        re_feats=torch.cat([re_drug_feats,re_protein_feats],dim=1)
        
        pair=feats+re_feats
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict