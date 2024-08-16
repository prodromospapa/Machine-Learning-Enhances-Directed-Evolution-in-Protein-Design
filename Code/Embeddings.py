import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA

#A total sample of 928 mutations of endolysin along with their corresponding effect on protein stability
#dataset=pd.read_csv(os.getcwd()+"\\Endolysin_Data.csv")[['sequence','ddG']].dropna()
dataset=pd.read_csv(os.getcwd()+"\\Endolysin_Data.csv")
dataset["sequence"]=dataset.apply(lambda x: x['sequence'][:x['position']-1]+ x["mutation"].upper() + x["sequence"][x["position"]:], axis=1)



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
pooled_embeddings=torch.mean(embeddings,dim=2)


pca=PCA(n_components=2)
pca.fit(tokenized_seqs["input_ids"])
transformed_seqs=pca.transform(tokenized_seqs["input_ids"])

import matplotlib.pyplot as plt
import seaborn as sns


#Temporary code to retrieve the embeddings that i calculated
with open(os.getcwd()+"\\embeddings.npy",'rb') as f:
    a=np.load(f)

fig, ax=plt.subplots()
ax.scatter(transformed_seqs[:,0],transformed_seqs[:,1])
plt.show()