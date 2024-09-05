from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from Code.Data_Import import get_data, set_mutation
from Code.Embeddings import Tokenization
from Code.DDG_Calculation import Energy_Calculation
#Generation of the appropriate data
data=get_data("Endolysin_data")
wild_type, data=set_mutation(data)


#Initialize the tokenization process
tokenize=Tokenization(wild_type=wild_type)

#Initialization of the DDG calculation procedure
Gibbs=Energy_Calculation()

#Excluded indices 83,84 because they do not correspond to the same sequence
#x_train=np.array([tokenize.generate_token(data.iloc[index]["sequence"]) for index in range(data.shape[0]) if not index in [83,84]])
x_train=np.array(tokenize.generate_token(data["sequence"].tolist()))

y_train=np.array(data["ddG"].tolist())
#y_train=np.delete(y_train, np.array([83,84]))

#def true_function(x: int)-> int:
    #return 5/(1+np.exp(-x))

#Definition of the training samples// i.e. The data that we have information about
#x_train=np.random.uniform(low=-6,high=6,size=2)
#y_train=np.apply_along_axis(true_function,axis=0,arr=x_train)


#Definition of the test points // i.e. The total interval of points where we will test the function

#x = np.random.uniform(low=x_train[:,0].min(),
                      #high=x_train[:,0].max(),
                      #size=(10_000,)).reshape(-1,1)

#y = np.random.uniform(low=x_train[:,1].min(),
                      #high=x_train[:,1].max(),
                      #size=(10_000,)).reshape(-1,1)

#x_test=np.column_stack((x,y))

x_test=tokenize.generate_test_data(number=1000)

#This could be used to sample from each conditional for every test point and define a random function 
#Based on the current GP //I need not use that
#y_samples = gp.sample_y(x, n_samples=5)

#Defining the selection criteria that will be used for optimization
def UCB(mean: np.array, std: np.array, coef: float)-> np.array:
    return mean + coef * std

def get_snapshot(gp,y_mean,y_std,ucb,x_train,y_train,trained=False):

    #This predicts the mean and variance of each conditional distribution for all test points

    fig,ax=plt.subplots(nrows=2,ncols=1)

    if trained:
        ax[0].scatter(x_train[:,0],y_train,zorder=1)

    ax[0].plot(x_test[:,0], y_mean, color="black", label="Mean",zorder=2)

    ax[0].fill_between(
            x_test[:,0],
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.1,
            color="black",
        )  
    
    #This is the acquisition function based on the fitted model
    ax[0].set_ylim([np.min(y_mean)-3,np.max(y_mean)+3])

    #This is the True function in the local landscape
    #ax[0].plot(x,list(map(true_function,x.reshape(-1,))),'b--')
    sns.despine()

    #This is the code for the visualization of the UCB acquisition function
    ax[1].plot(x_test[:,0],ucb,color='green')
    ax[1].fill_between(
        x_test[:,0],
        np.zeros((x_test.shape[0])),
        ucb,
        alpha=0.1,
        color="green",
        interpolate=True  
    )
    #This is the next point for which the UCB function is maximized and will be tested for desired function value
    ax[1].axvline(x=x_test[np.argmin(ucb),0],ymin=0,ymax=np.min(ucb),linestyle='--',color='k')
    ax[1].scatter(x_test[np.argmin(ucb),0],np.min(ucb),marker='x',color='k')
    
    #Could build some code to save the checkpoint images somewhere
    plt.savefig("snap")

def calculate_state(model,test_data, ucb_coef=8):
    y_mean,y_std=model.predict(X=test_data,return_std=True)
    ucb=UCB(y_mean,y_std,ucb_coef)
    return y_mean, y_std, ucb

#This is the original trained GP
def training(train_data:np.array, known_values: np.array, x_test:np.array, cycles: int = 0)-> tuple:
    #Defining the RBF kernel 
    kernel=RBF(length_scale=1)
    #Initialization of the Gaussian Process Regressor 
    gp=GP(kernel=kernel)
    y_mean,y_std,ucb=calculate_state(gp,x_test,ucb_coef=8)
    #get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=train_data,y_train=known_values)

    #Fitting the known data
    gp.fit(X=train_data,y=known_values)
    y_mean,y_std,ucb=calculate_state(gp,x_test,ucb_coef=8)
    #get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=train_data,y_train=known_values,trained=True)
    
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
            y_mean,y_std,ucb=calculate_state(gp,x_test,ucb_coef=8)
            #get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=train_data,y_train=known_values,trained=True)

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

            y_mean,y_std,ucb=calculate_state(gp,x_test,ucb_coef=8)
            #get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=x_train,y_train=y_train,trained=True)

            next_point=x_test[np.argmin(ucb)]
        
        return tokenize.generate_seq(train_data[np.argmin(known_values)]),known_values[np.argmin(known_values)]

import time
start=time.time()
opt,stabilization=training(x_train,y_train,x_test,cycles=3)
end=time.time()
print(f"Elapsed time {end-start}")


exit()

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

plt.close()
fig.savefig("m")