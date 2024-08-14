import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch

#A total sample of 928 mutations of endolysin along with their corresponding effect on protein stability
dataset=pd.read_csv(os.getcwd()+"\\Endolysin_Data.csv")[['sequence','ddG']].dropna()

seqs=dataset['sequence'].to_list()

#initialization of the Tokenizer // i.e. The method by which we encode the protein sequence to machine readable form
tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model=AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

tokenized_seqs=tokenizer(seqs,return_tensors="pt", padding=True)

#Formulation of the embeddings
with torch.no_grad():
    outputs = model(**tokenized_seqs)
embeddings = outputs.last_hidden_state

#This could be used to create the pooled embeddings
torch.mean(embeddings,dim=1)