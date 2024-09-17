import pandas as pd
import os


#Apply the existing mutations to the sequence column 
def set_mutation(data):
    wild_type=data.iloc[0]["sequence"]
    data["sequence"]=data.apply(lambda x: x['sequence'][:x['position']-1]+ x["mutation"].upper() + x["sequence"][x["position"]:], axis=1)
    return wild_type, data

def get_data(filename):
    dataset=pd.read_csv(os.getcwd()+f"/{filename}.csv")

#Filtering my data based on arbitrary criterions set by us // That is, curated data, with existing Tm effects and pH conditions of 6<=pH<=8
#There still exist some duplicate values, thus one must filter the data further
    #filtered_dataset=dataset[(dataset["is_curated"]==True) & (~dataset["ddG"].isna())  & (6<=dataset["pH"]) & (dataset["pH"]<=8)]
    
    dataset.dropna(subset=['pH'],inplace=True)
    filtered_dataset=dataset[dataset['pH'] == dataset['pH'].value_counts().idxmax()]

#Drop the duplicate entries that exist based on the relevant columns of my data
    filtered_dataset.drop_duplicates(subset=["wild_type","position","mutation","pH"],keep='first',inplace=True)
    return filtered_dataset


