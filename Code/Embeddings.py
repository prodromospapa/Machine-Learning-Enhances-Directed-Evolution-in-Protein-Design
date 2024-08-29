from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModel

class Tokenization:
    
    def __init__(self):
        self.pca=PCA(n_components=2)
        self._tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    def generate_token(self,seq):
        return self._tokenizer(seq, padding=True)["input_ids"]
    
    def project(self,data):
        self.pca.fit(data)
        return self.pca.transform(data)

    def generate_seq(self,projection):
        gen=np.round(self.pca.inverse_transform(projection))

        return self._tokenizer.decode(gen,skip_special_tokens=True).replace(' ','')
    
    def sequence_compare(self, seq1, seq2):
        seq1, seq2 =list(seq1), list(seq2)
        check=zip(seq1, seq2)
        
        results=dict(filter(lambda x: x[1][0]!=x[1][1],enumerate(check)))

        return list(results.keys()), list(map(lambda x: x[1], results.values()))
        