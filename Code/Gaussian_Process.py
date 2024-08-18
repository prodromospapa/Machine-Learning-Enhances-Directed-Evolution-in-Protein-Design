from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#This is a dummy function that will be used instead of the oracle for the time being
#This is just a basic sigmoid function

def true_function(x: int)-> int:
    return 5/(1+np.exp(-x))


#Definition of the training samples// i.e. The data that we have information about
x_train=np.random.uniform(low=-6,high=6,size=2)
y_train=np.apply_along_axis(true_function,axis=0,arr=x_train)


#Definition of the test points // i.e. The total interval of points where we will test the function
x = np.linspace(start=-6, 
                stop=6, 
                num=1000).reshape(-1,1)


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
        ax[0].scatter(x_train,y_train,zorder=1)

    ax[0].plot(x.reshape(-1,), y_mean, color="black", label="Mean",zorder=2)

    ax[0].fill_between(
            x.reshape(-1,),
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.1,
            color="black",
        )  
    
    #This is the acquisition function based on the fitted model
    ax[0].set_ylim([np.min(y_mean)-3,np.max(y_mean)+3])

    #This is the True function in the local landscape
    ax[0].plot(x,list(map(true_function,x.reshape(-1,))),'b--')
    sns.despine()

    #This is the code for the visualization of the UCB acquisition function
    ax[1].plot(x,ucb,color='green')
    ax[1].fill_between(
      x.reshape(-1,),
        np.zeros((x.shape[0])),
        ucb,
        alpha=0.1,
        color="green",
        interpolate=True  
    )
    #This is the next point for which the UCB function is maximized and will be tested for desired function value
    ax[1].axvline(x=x[np.argmax(ucb)],ymin=0,ymax=np.max(ucb),linestyle='--',color='k')
    ax[1].scatter(x[np.argmax(ucb)],np.max(ucb),marker='x',color='k')
    fig.show()
    #Could build some code to save the checkpoint images somewhere
    #plt.savefig()

def calculate_state(model,test_data, ucb_coef=8):
    y_mean,y_std=model.predict(X=test_data,return_std=True)
    ucb=UCB(y_mean,y_std,ucb_coef)
    return y_mean, y_std, ucb

#This is the original trained GP
def training(train_data:np.array, known_values: np.array, cycles: int= 0)->float:
    #Defining the RBF kernel 
    kernel=RBF(length_scale=1)

    #Initialization of the Gaussian Process Regressor 
    gp=GP(kernel=kernel)
    y_mean,y_std,ucb=calculate_state(gp,x)
    get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=train_data,y_train=known_values)

    #Fitting the known data
    gp.fit(X=train_data.reshape(-1,1),y=known_values.reshape(-1,1))
    y_mean,y_std,ucb=calculate_state(gp,x)
    get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=train_data,y_train=known_values,trained=True)
    
    #Finding the maximum point to be considered based on the UCB criterion
    next_point=x[np.argmax(ucb)]
    if cycles!=0:
        for cycle in range(cycles):
            #If the value converges before the cycles end then break out of the loop
            if next_point==train_data[-1]:
                break
            
            #Update the training data
            train_data=np.concatenate((train_data,next_point))
            known_values=np.concatenate((known_values,true_function(next_point)))

            #Fit the new GP
            gp.fit(X=train_data.reshape(-1,1),y=known_values.reshape(-1,1))
            #Calculate the conditional parameters for my all points that belong to my test space
            y_mean,y_std,ucb=calculate_state(gp,x)
            get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=train_data,y_train=known_values,trained=True)

            next_point=x[np.argmax(ucb)]
        
        return next_point[0]
    
    #If cycles are not specified, then just check for convergence
    while next_point!=train_data[-1]:
        train_data=np.concatenate((train_data,next_point))
        known_values=np.concatenate((known_values,true_function(next_point)))

        gp.fit(X=train_data.reshape(-1,1),y=known_values.reshape(-1,1))
        y_mean,y_std,ucb=calculate_state(gp,x)
        get_snapshot(gp=gp,y_mean=y_mean,y_std=y_std,ucb=ucb,x_train=train_data,y_train=known_values,trained=True)

        next_point=x[np.argmax(ucb)]
        
        return next_point[0]
    
training(x_train,y_train,5)