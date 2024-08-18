import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA

#A total sample of 928 mutations of endolysin along with their corresponding effect on protein stability
#dataset=pd.read_csv(os.getcwd()+"\\Endolysin_Data.csv")[['sequence','ddG']].dropna()
dataset=pd.read_csv(os.getcwd()+"\\Endolysin_Data.csv")


#Filtering my data based on arbitrary criterions set by us // That is, curated data, with existing Tm effects and pH conditions of 6<=pH<=8
#There still exist some duplicate values, thus one must filter the data further
filtered_dataset=dataset[(dataset["is_curated"]==True) & (~dataset["dTm"].isna())  & (6<=dataset["pH"]) & (dataset["pH"]<=8)]

#Drop the duplicate entries that exist based on the relevant columns of my data
filtered_dataset.drop_duplicates(subset=["wild_type","position","mutation","pH","tm"],keep='first',inplace=True)

wt_seq=dataset.iloc[0]["sequence"]

filtered_dataset["sequence"]=filtered_dataset.apply(lambda x: x['sequence'][:x['position']-1]+ x["mutation"].upper() + x["sequence"][x["position"]:], axis=1)


#Create a random seq
rand_seq=wt_seq
indices=[]
for _ in range(2):
    index=np.random.randint(filtered_dataset.shape[0])
    indices.append((index,filtered_dataset.iloc[index]["position"]))
    params=filtered_dataset.iloc[index][["position","mutation"]].tolist()
    rand_seq=rand_seq[:params[0]]+params[1]+rand_seq[params[0]+1:]



seqs=filtered_dataset['sequence'].to_list()

#initialization of the Tokenizer // i.e. The method by which we encode the protein sequence to machine readable form
tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model=AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

tokenized_seqs=tokenizer(seqs,return_tensors="pt", padding=True)

tok_seq=tokenizer(rand_seq,return_tensors="pt", padding=True)


#Formulation of the embeddings
with torch.no_grad():
    outputs = model(**tok_seq)
    #outputs = model(**tokenized_seqs)
embeddings = outputs.last_hidden_state

#This could be used to create the pooled embeddings
pooled_embeddings=torch.mean(embeddings,dim=2)


#From here on out, it is just code for some fancy visualizations

pca=PCA(n_components=2)
pca.fit(a)
transformed_seqs=pca.transform(a)
rand_transform=pca.transform(pooled_embeddings)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

#Temporary code to retrieve the embeddings that I calculated
with open(os.getcwd()+"\\embeddings.npy",'rb') as f:
    a=np.load(f)

t=transformed_seqs[transformed_seqs[:,0]<0.014]

relevant=filtered_dataset.index.tolist()
fig, ax=plt.subplots()


#Setting up a color ramp
colors = ['#A3EBB1', '#21B6A8','#116530']
positions = [0,0.5,1]
cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)))

scatter=ax.scatter(x=t[relevant,0],y=t[relevant,1],c=filtered_dataset["position"].tolist(), cmap=cmap, edgecolors='k', linewidths=0.5,)

#This is the code to plot a random synthetic sequence embedding on the two-dimensional plain
rand_rep=ax.scatter(x=rand_transform[:,0],y=rand_transform[:,1],marker='x',c='k')

#Some code to plot the original sequence representations from which the synthetic sequence arose
sub=np.array(relevant)[list(dict(indices).keys())]
parent_seqs=ax.scatter(x=t[sub,0],y=t[sub,1],linewidths=0.5, c=["#ECF87F","#ECF87F"],edgecolors='k')
plt.xticks(rotation=45)
ax.set_xlabel(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.3f}%)")
ax.set_ylabel(ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.3f}%)")

plt.tight_layout()
cbar=fig.colorbar(mappable=scatter)
cbar.set_label("Mutation Position")
fig.show()
plt.close()