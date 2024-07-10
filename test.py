import subprocess
import numpy as np
import argparse
import os
import datetime
import rpy2.robjects as robjects
import networkx as nx
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.score import Scoresheet
import utils as f
from sklearn.linear_model import LogisticRegressionCV 
from evalne.evaluation.split import EvalSplit
import pandas as pd
import networkx as nx
import multiprocessing
from sklearn.ensemble import RandomForestClassifier


def main(args=None):
          
    cpu_number = multiprocessing.cpu_count()  
    
    parser = argparse.ArgumentParser(description='Path of networks')
    parser.add_argument('-n', type=str, help='Multiplex 1')
    parser.add_argument('-m', type=str, help='Multiplex 2')    
    parser.add_argument('-b', type=str, help='Bipartite')        
    
    args = parser.parse_args(args)
    print(args)

    ########################################################################
    # Parameters multiverse and train/test
    ########################################################################
    EMBED_DIMENSION = 128
    CLOSEST_NODES = np.int64(300)
    NUM_SAMPLED = np.int64(10)
    LEARNING_RATE = np.float64(0.01)
    KL = False
    NB_CHUNK = np.int64(1)
    CHUNK_SIZE = np.int64(100)
    NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)


    train_frac =0.7
    solver = 'lbfgs'
    max_iter= 1000
    split_alg = 'random'
    lp_model = RandomForestClassifier(n_estimators=400, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
                                                      max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,\
                                                      oob_score=True, n_jobs=cpu_number, random_state=777, verbose=0, warm_start=False) 

    graph_name = 'Test_Eval'

     ##################################################################################
    # !! Careful !! 
    # Check if nodes in the bipartite have the same nodes in the multiplex
    # networks. If not you have to remove the nodes in the multiplexes not included in the  
    # bipartites
    ##################################################################################

    ###################################################################################
    # EvalNE Link prediction processing
    ###################################################################################

    data_bipartite = pd.read_csv(args.b, delimiter = ' ', header = None) 
    data_bipartite = data_bipartite.drop(columns = [0,3])
    data_bipartite.to_csv('bipartite_2colformat.csv', header=None, index= None, sep = ' ')

    G_hetereogeneous = f.preprocess('bipartite_2colformat.csv', '.', ' ', False,  False, True)
    print('Preprocessing done')
    G_hetereogeneous_traintest_split = EvalSplit()
    G_hetereogeneous_traintest_split.compute_splits(G_hetereogeneous, split_alg=split_alg, train_frac=train_frac, owa=False)
    nee = LPEvaluator(G_hetereogeneous_traintest_split, dim=EMBED_DIMENSION, lp_model=lp_model)
    G_heterogeneous_split = (G_hetereogeneous_traintest_split.TG)
    os.replace('bipartite_2colformat.csv', './Generated_graphs/'+ 'bipartite_2colformat.csv')
    print('Splitting done')

    # Write the bipartite training graph for multiverse in extended edgelist format 'layer n1 n2 weight'
    file_multi = open('bipartite_training_graph_'  + '_'+ graph_name, 'w+')  
    tmp_array_het = []
    tmp_array_het = np.asarray(G_heterogeneous_split.edges)
    for i in range(len(tmp_array_het[:,0])):
        if tmp_array_het[i,0] in list(data_bipartite[2]):
            tmp = tmp_array_het[i,0]
            tmp_array_het[i,0] = tmp_array_het[i,1]
            tmp_array_het[i,1] = tmp

    tmp_array_het = np.hstack((tmp_array_het, np.ones((len(tmp_array_het),1))))
    tmp_array_het = np.hstack((np.ones((len(tmp_array_het),1)), tmp_array_het))
    tmp_array_het = np.vstack(tmp_array_het)
    tmp_array_het = np.int_(tmp_array_het)

    np.savetxt(file_multi, tmp_array_het, fmt='%s', delimiter=' ', newline=os.linesep)
    
    file_multi.close()
    os.replace('bipartite_training_graph_'  + '_'+ graph_name, './Generated_graphs/'+ 'bipartite_training_graph_'  + '_'+ graph_name+'.txt')

    ###################################################################################
    # MULTIVERSE
    ###################################################################################
    r_readRDS = robjects.r['readRDS']
    
    print('RWR-MH')
    proc = subprocess.Popen(['Rscript',  './GSM_MH_test.R', \
              '-n', '.' + args.n,  \
              '-m', '.' + args.m,  \
              '-b', '../Generated_graphs/'+ 'bipartite_training_graph_'  + '_'+ graph_name+'.txt', 
              '-o', '../ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name, '-c', str(cpu_number)])

    proc.wait() 
    proc.kill()
    print('RWR done')
    
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name +'.rds') 




main()
