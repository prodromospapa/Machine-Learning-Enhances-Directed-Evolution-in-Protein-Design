import torch
from transformers import AutoTokenizer, AutoModel


tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model=AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

tokenized_seqs=tokenizer(seqs,return_tensors="pt", padding=True)

#Formulation of the embeddings
with torch.no_grad():
    outputs = model(**tokenized_seqs)
    #outputs = model(**tokenized_seqs)
embeddings = outputs.last_hidden_state

#This could be used to create the pooled embeddings
pooled_embeddings=torch.mean(embeddings,dim=2)
token_seqs=tokenized_seqs['input_ids']



import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

fig, ax=plt.subplots()

colors = ['#A3EBB1', '#21B6A8','#116530']
positions = [0,0.5,1]
cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)))

scatter=ax.scatter(x=x_train[:,0],y=x_train[:,1],c=y_train, cmap=cmap, edgecolors='k', linewidths=0.5,)

plt.xticks(rotation=45)
ax.set_xlabel(xlabel=f"PC1 ({tokenize.pca.explained_variance_ratio_[0]*100:.3f}%)")
ax.set_ylabel(ylabel=f"PC2 ({tokenize.pca.explained_variance_ratio_[1]*100:.3f}%)")

ax.set_xlim([x_train[:,0].mean() - x_train[:,0].var(), x_train[:,0].mean() + x_train[:,0].var()])
ax.set_ylim([x_train[:,1].mean() - x_train[:,1].var(), x_train[:,1].mean() + x_train[:,1].var()])

plt.tight_layout()
cbar=fig.colorbar(mappable=scatter)
cbar.set_label("Mutation Position")

fig.savefig("wtf")




#Setting up a color ramp
colors = ['#A3EBB1', '#21B6A8','#116530']
positions = [0,0.5,1]
cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)))

#scatter=ax.scatter(x=t[relevant,0],y=t[relevant,1],c=filtered_dataset["position"].tolist(), cmap=cmap, edgecolors='k', linewidths=0.5,)

scatter=ax.scatter(x=transformed_seqs[:,0],y=transformed_seqs[:,1],c=filtered_dataset["position"].tolist(), cmap=cmap, edgecolors='k', linewidths=0.5,)


#This is the code to plot a random synthetic sequence embedding on the two-dimensional plain
rand_rep=ax.scatter(x=rand_transform[:,0],y=rand_transform[:,1],marker='x',c='k')

#Some code to plot the original sequence representations from which the synthetic sequence arose
#sub=np.array(relevant)[list(dict(indices).keys())]

#This is the implementation of sub for token sequences
sub=list(dict(indices).keys())
parent_seqs=ax.scatter(x=transformed_seqs[sub,0],y=transformed_seqs[sub,1],linewidths=0.5, c=["#ECF87F","#ECF87F"],edgecolors='k')

#General stuff
plt.xticks(rotation=45)
ax.set_xlabel(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.3f}%)")
ax.set_ylabel(ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.3f}%)")
ax.set_xlim([mean_x - 0.8 * var_x/10, -0.5])
ax.set_ylim([-1.25, 0.25])
plt.tight_layout()
cbar=fig.colorbar(mappable=scatter)
cbar.set_label("Mutation Position")
fig.show()

mean_x, mean_y=np.median(transformed_seqs, axis=0)
var_x, var_y=np.std(transformed_seqs, axis=0)

plt.close()