from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from Data_Import import get_data, set_mutation
from Embeddings import Tokenization
from DDG_Calculation import Energy_Calculation
#Generation of the appropriate data
data=get_data("Endolysin_data")
wild_type, data=set_mutation(data)


#Initialize the tokenization process
tokenize=Tokenization(wild_type=wild_type)

#Initialization of the DDG calculation procedure
Gibbs=Energy_Calculation()

#Excluded indices 83,84 because they do not correspond to the same sequence
x_train=np.array(tokenize.generate_token(data["sequence"].tolist()))

y_train=np.array(data["ddG"].tolist())


x_test=tokenize.generate_test_data(number=1000)



#Defining the selection criteria that will be used for optimization
def UCB(mean: np.array, std: np.array, coef: float)-> np.array:
    return mean + coef * std


def calculate_state(model,test_data, ucb_coef=8):
    y_mean,y_std=model.predict(X=test_data,return_std=True)
    return UCB(y_mean,y_std,ucb_coef)

#This is the original trained GP
def training(train_data:np.array, known_values: np.array, x_test:np.array, cycles: int = 0)-> tuple:
    #Defining the RBF kernel 
    kernel=RBF(length_scale=1)
    #Initialization of the Gaussian Process Regressor 
    gp=GP(kernel=kernel)

    #Fitting the known data
    gp.fit(X=train_data,y=known_values)
    ucb=calculate_state(gp,x_test,ucb_coef=8)
    
    #Finding the maximum point to be considered based on the UCB criterion
    next_point=x_test[np.argmin(ucb)]

    if cycles!=0:

        for cycle in range(cycles):
            
            print(f"Cycle: {cycle}")
    
            
            #Update the training data
            train_data=np.append(train_data,next_point.reshape(1,-1),axis=0)
            seq=tokenize.generate_seq(next_point)
            positions, mutations=tokenize.sequence_compare(wild_type, seq)
   
            mut=Gibbs.set_mutations(position=positions, mutation=mutations)
            ddG=[Gibbs.get_Gibbs(mut)-Gibbs.base]
            
            if known_values.min()<=ddG:
                break

            known_values=np.concatenate((known_values,np.array(ddG)))


            #Fit the new GP
            gp.fit(X=train_data,y=known_values.reshape(-1,1))

            #Set a new test set
            x_test=tokenize.generate_test_data(1000)

            #Calculate the conditional parameters for my all points that belong to my test space
            ucb=calculate_state(gp,x_test,ucb_coef=8)

            next_point=x_test[np.argmin(ucb)]
        
        return tokenize.generate_seq(train_data[np.argmin(known_values)]),known_values[np.argmin(known_values)]
    
    else:
        ddG=-np.inf
        #If cycles are not specified, then just check for convergence
        while known_values.min()>ddG:
            
            print("Entered")
            train_data=np.append(train_data,next_point.reshape(1,-1),axis=0)
            
            seq=tokenize.generate_seq(next_point)
            positions, mutations=tokenize.sequence_compare(wild_type, seq)

            mut=Gibbs.set_mutations(position=positions, mutation=mutations)
            ddG=[Gibbs.get_Gibbs(mut)-Gibbs.base]

            known_values=np.concatenate((known_values,np.array(ddG)))

            gp.fit(X=train_data,y=known_values.reshape(-1,1))

            x_test=tokenize.generate_test_data(1000)

            ucb=calculate_state(gp,x_test,ucb_coef=8)

            next_point=x_test[np.argmin(ucb)]
        
        return tokenize.generate_seq(train_data[np.argmin(known_values)]),known_values[np.argmin(known_values)]


opt,stabilization=training(x_train,y_train,x_test,cycles=3)

with open("result.txt","w") as f:
    f.write(f"Sequence: {opt}\t {stabilization}")