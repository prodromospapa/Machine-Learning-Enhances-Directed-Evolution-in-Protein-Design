#export PATH="/home/pipis/foldx:$PATH"
import pandas as pd
from io import StringIO
import os

rep = 10 # Number of runs for each mutation

mut_df = pd.read_csv('../Endolysin_Data.csv')

pos = mut_df['position']
wild = mut_df['wild_type']
mut = mut_df['mutation']
chain = mut_df['chain']
ddg = mut_df['ddG']
dtm = mut_df['dTm']

positions = [str(w + c + str(p) + m) for w, c, p, m in zip(wild, chain, pos, mut)]


#remove HA48N as it is not a valid mutation
i_remove = [i for i, position in enumerate(positions) if position == "HA48N"]

pos = pos.drop(i_remove)
wild = wild.drop(i_remove)
mut = mut.drop(i_remove)
chain = chain.drop(i_remove)
ddg = ddg.drop(i_remove)
dtm = dtm.drop(i_remove)
positions = [str(w + c + str(p) + m) for w, c, p, m in zip(wild, chain, pos, mut)]

#correct the pdb file
with open("individual_list.txt", "w") as f:
    f.write(f"GA12R,RA137I;")
if not os.path.exists("2lzm+correct.pdb"):        
    os.system(f"foldx --command=BuildModel --pdb=2lzm.pdb --mutant-file=individual_list.txt --numberOfRuns=1")
    os.system(f"mv 2lzm_1.pdb 2lzm+correct.pdb")

with open("individual_list.txt", "w") as f:
    for position in positions:
        f.write(f"{position};\n")
        #os.system(f'./foldx_20241231  --command=PositionScan --pdb=2lzm.pdb --positions={position} --output-file={position}') 

#os.system(f"foldx  --command=BuildModel --pdb=2lzm+correct.pdb --mutant-file=individual_list.txt --numberOfRuns={rep}")


with open("Average_2lzm+correct.fxout", "r") as f:
    file = f.read()
    df = pd.read_csv(StringIO(file), sep="\t", skiprows=8)
    calculated_ddg = df['total energy']

final_df = pd.DataFrame({'position': list(positions),'experimental_dTm':list(dtm), 'experimental_ddg': ddg,'calculated_ddg': list(calculated_ddg)})
final_df.to_csv("calculated_ddg.csv", index=False)


os.system(f"rm *_*.pdb")
os.system(f"rm *.fxout")
os.system("find . -type f -name '*.txt' ! -name 'rotabase.txt' -exec rm {} +")


