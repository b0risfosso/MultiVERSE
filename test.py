# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:09:00 2019

@author: Léo Pio-Lopez
"""


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
    
    # If toy example
    #EMBED_DIMENSION = 128
    #CLOSEST_NODES = np.int64(2)
    #NUM_SAMPLED = np.int64(10)
    #LEARNING_RATE = np.float64(0.01)
    #KL = False
    #NB_CHUNK = np.int64(1)
    #CHUNK_SIZE = np.int64(2)
    #NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)

    graph_name = 'Test_Eval'
    
    
        
    ##################################################################################
    # !! Careful !! 
    # Check if nodes in the bipartite have the same nodes in the multiplex
    # networks. If not you have to remove the nodes in the multiplexes not included in the  
    # bipartites
      
    ###################################################################################
    # MULTIVERSE
    ###################################################################################
    r_readRDS = robjects.r['readRDS']
    
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name +'.rds') 

    import gc
    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, rawdata_DistancematrixPPI, neighborhood, nodesstr \
     = f.netpreprocess(r_DistancematrixPPI, CLOSEST_NODES)
    

        ########################################################################
        # Initialization
        ######################################################################## 

    embeddings = np.random.normal(0, 1, [np.size(nodes), EMBED_DIMENSION])

        ########################################################################
        # Training and saving best embeddings   
        ######################################################################## 
    # Train and test during training
    neighborhood = np.asarray(neighborhood)
    nodes= np.asarray(nodes)

    print("Data types of the parameters before calling train function:")
    print(f"neighborhood: {type(neighborhood)}")
    print(f"nodes: {type(nodes)}")
    print(f"list_neighbours: {type(list_neighbours)}")
    print(f"NUM_STEPS_1: {type(NUM_STEPS_1)}")
    print(f"NUM_SAMPLED: {type(NUM_SAMPLED)}")
    print(f"LEARNING_RATE: {type(LEARNING_RATE)}")
    print(f"CLOSEST_NODES: {type(CLOSEST_NODES)}")
    print(f"CHUNK_SIZE: {type(CHUNK_SIZE)}")
    print(f"NB_CHUNK: {type(NB_CHUNK)}")
    print(f"embeddings: {type(embeddings)}")
    print(f"reverse_data_DistancematrixPPI: {type(reverse_data_DistancematrixPPI)}")


    
    embeddings = f.train(neighborhood, nodes, list_neighbours, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, \
                         CLOSEST_NODES, CHUNK_SIZE, NB_CHUNK, embeddings, reverse_data_DistancematrixPPI)


    X = dict(zip(range(embeddings.shape[0]), embeddings))
    X = {str(int(nodes[key])): X[key] for key in X}

    # Convert nodes to numpy array if not already
    nodes = np.asarray(nodes)

    # Check for missing nodes
    missing_nodes = [str(node) for node in nodes if str(node) not in X]
    if missing_nodes:
        print(f"Missing embeddings for nodes: {missing_nodes}")
    else:
        print("No missing node.")
        
    np.save('embeddings_MH', X)
    date = datetime.datetime.now()
    os.replace('embeddings_MH.npy', './ResultsMultiVERSE/'+ 'embeddings_MH.npy')

             
        ########################################################################
        # Link prediction for evaluation of MH
        ######################################################################## 

    edge_emb = ['hadamard', 'weighted_l1', 'weighted_l2', 'average', 'cosine']
    results_embeddings_methods = dict()

    for i in range (len(edge_emb)):
        tmp_result_multiverse = nee.evaluate_ne(data_split=nee.traintest_split, X=X, method="Multiverse", edge_embed_method=edge_emb[i], label_binarizer=lp_model)
        results_embeddings_methods[tmp_result_multiverse.method +'_'  + str(edge_emb[i])] = tmp_result_multiverse.get_all()[1][4]


    ########################################################################
    # Analysis and saving of the results
    ######################################################################## 
    
    Result_file = 'Result_LinkpredMultiplexHet_'+graph_name+'_'+str(date)+'.txt'
    with open(Result_file,"w+") as overall_result:
       print("%s: \n\
                EMBED_DIMENSION: %s \n\
                CLOSEST_NODES: %s  \n\
                NUM_STEPS_1: %s  \n\
                NUM_SAMPLED: %s  \n\
                LEARNING_RATE: %s  \n\
                CHUNK_SIZE: %s  \n\
                NB_CHUNK: %s  \n\
                train_frac: %s \n\
                solver: %s \n\
                max_iter: %s  \n\
                split_alg: %s  \n\
                "% (str(date), EMBED_DIMENSION, CLOSEST_NODES, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, CHUNK_SIZE, NB_CHUNK, train_frac, solver, max_iter, split_alg), file=overall_result)
             
       print('Overall MULTIVERSE AUC hadamard:', results_embeddings_methods['Multiverse_hadamard'], file=overall_result)
       print('Overall MULTIVERSE AUC weighted_l1:', results_embeddings_methods['Multiverse_weighted_l1'], file=overall_result)
       print('Overall MULTIVERSE AUC weighted_l2:', results_embeddings_methods['Multiverse_weighted_l2'], file=overall_result)
       print('Overall MULTIVERSE AUC average:', results_embeddings_methods['Multiverse_average'], file=overall_result)
       print('Overall MULTIVERSE AUC cosine:', results_embeddings_methods['Multiverse_cosine'], file=overall_result)
       
  
    overall_result.close() 
    os.replace(Result_file, './ResultsMultiVERSE/'+ Result_file)
    
    print('End')

if __name__ == "__main__":
    main()
