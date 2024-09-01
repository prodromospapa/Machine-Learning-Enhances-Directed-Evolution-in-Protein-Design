from pyrosetta import *
import numpy as np
from pyrosetta.rosetta import *
from pyrosetta.toolbox import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.protocols.relax import FastRelax

class Energy_Calculation:
    def __init__(self):
        
        #Initilization of PyRosetta and setting up
        init()
        #Setting up the score function for Gibbs Energy Calculation
        scorefxnDDG=get_fa_scorefxn()

        self._score = scorefxnDDG

        #Setting up the relaxation parameters
        scorefxnRelax = pyrosetta.create_score_function("ref2015_cart")
        relax = pyrosetta.rosetta.protocols.relax.FastRelax()
        relax.constrain_relax_to_start_coords(True)
        relax.coord_constrain_sidechains(True)
        relax.ramp_down_constraints(False)
        relax.cartesian(True)
        relax.min_type("dfpmin_armijo_nonmonotone")
        relax.set_scorefxn(scorefxnRelax)

        self._relax=relax

        wild_type=Pose()
        wild_type=pose_from_pdb("Code_server/2lzm.pdb")
        mutate_residue(wild_type,12,"R") #Defining the mutations made to the crystal structure, so that it matches the wt-seq in our data
        mutate_residue(wild_type,137,"I")
        
        self._wild_type=wild_type

        #Get the Gibbs score for the wt-sequence
        relax.apply(wild_type)
        self._minimize_energy(wild_type)
        self.base=self.get_Gibbs(wild_type)
    
    def set_mutations(self,position,mutation):
        #Initialize the sequence, as the non-mutated wild type sequence
        mutated=Pose()
        mutated.assign(self._wild_type)

        #Define the single/multiple mutation profile for said sequence
        if type(position)==list:
            for pos, mut in zip(position,mutation):
                mutate_residue(mutated,pos+1,mut) #Addition of +1 since the index starts from one
        
        else:
            mutate_residue(mutated,position+1,mutation)
        
        #Relax the generated sequence
        self._relax.apply(mutated)

        return mutated


    def _minimize_energy(self,pose,n_cycles=100):

        #initialize MC energy minimization procedure
         min_mover = MinMover() 
         mm=MoveMap()
         mm.set_bb(True)
         min_mover.movemap(mm)
         min_mover.score_function(self._score)
         min_mover.min_type("dfpmin")
         min_mover.tolerance(0.01)

         MC=MonteCarlo(pose,self._score,1) #1 stands for kT i.e. the Manhattan Criterion threshold
         
         for _ in range(n_cycles):
             min_mover.apply(pose)
             MC.boltzmann(pose)
        
         MC.recover_low(pose)
        
         return None


    def get_Gibbs(self,pose):
        #Apply the MC energy minimization for all structures 
        self._minimize_energy(pose)

        #Score the resulting structures
        return self._score(pose)



#Gibbs=Energy_Calculation()
#mut=Gibbs.set_mutations(filtered_dataset.iloc[0]["position"]-1, filtered_dataset.iloc[0]["mutation"])
#mutated_gibbs=Gibbs.get_Gibbs(mut)



