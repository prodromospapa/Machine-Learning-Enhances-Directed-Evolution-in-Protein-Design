import numpy as np
from transformers import AutoTokenizer, AutoModel

class Tokenization:

    aminos={'M': 0, 'N': 1, 'I': 2, 'F': 3, 'E': 4, 'L': 5, 'R': 6, 'D': 7, 'K': 8,'Y': 9, 
     'T': 10, 'G': 11, 'H': 12, 'S': 13, 'P': 14, 'A': 15, 'C': 16,  'V': 17, 'Q': 18, 'W': 19}
    
    def __init__(self,wild_type):
        self._tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.wild_type=wild_type

    def generate_token(self,seq):
        return self._tokenizer(seq, padding=True)["input_ids"]
        #return list(map(lambda x: Tokenization.aminos[x],seq))
    
    
    def generate_seq(self,token):
        
        #inverse_dict=dict(map(lambda x: (x[1],x[0]),Tokenization.aminos.items()))
        return self._tokenizer.decode(token,skip_special_tokens=True).replace(' ','')
        #return "".join(map(lambda x: inverse_dict[x], token))
    
    
    def _mutate_seq(self,arr, position, mutation):
        base = arr.copy()
        base[position]=mutation

        return base
    
    
    def generate_test_data(self,number):

        mutations=np.random.randint(low=0,high=20, size=(number,))
        positions=np.random.randint(low=0,high=len(self.wild_type), size=(number,))
        baseline=np.array(self.generate_token(self.wild_type))
        
        return np.array([self._mutate_seq(baseline,pos,mut) for pos,mut in zip(list(positions),list(mutations))])

    
    def sequence_compare(self, seq1, seq2):
        seq1, seq2 =list(seq1), list(seq2)
        check=zip(seq1, seq2)
        
        results=dict(filter(lambda x: x[1][0]!=x[1][1],enumerate(check)))

        return list(results.keys()), list(map(lambda x: x[1], results.values()))