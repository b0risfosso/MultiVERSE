rm(list=ls());cat('\014');if(length(dev.list()>0)){dev.off()}

install.packages("unix") 
library(unix)
rlimit_as(1e12) 
options(vsize = 64000)


setwd("./RWR/")

## We load the R file containing the associated RWR functions.
source("Functions_RWRMH.R")

## Installation and load of the required R Packages
packages <- c("igraph", "mclust","Matrix","kernlab", "R.matlab","bc3net",
              "optparse","parallel","tidyverse")
ipak(packages)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Reading and checking the input arguments
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
## Reading input arguments

#!/usr/bin/env Rscript
print("Reading arguments...")
option_list = list(
  make_option(c("-n", "--network1"), type="character", default=NULL, 
              help="Path to the first multiplex network to be used as Input. 
              It should be a space separated four column file containing 
              the fields: edge_type, source, target and weight.", 
              metavar="character"),
  make_option(c("-m", "--network2"), type="character", default=NULL, 
              help="Path to the second multiplex network to be used as Input. 
              It should be a space separated four column file containing 
              the fields: edge_type, source, target and weight.", 
              metavar="character"),
  make_option(c("-b", "--bipartite"), type="character", default=NULL, 
              help="Path to the bipartite network to be used as Input. 
              It should be a space separated four column file containing 
              the fields: edge_type, source, target and weight. Source Nodes
              should be the ones from the first multiplex and target nodes
              from the second.", 
              metavar="character"),
  make_option(c("-r", "--restart"), type="double", default=0.7, 
              help="Value of the restart parameter ranging from 0 to 1. 
              [default= %default]", metavar="numeric"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
              help="Name for the output file. SimMatrix will be added before 
              this argument", metavar="character"),
  make_option(c("-c", "--cores"), type="integer", default=1, 
              help="Number of cores to be used for the Random Walk Calculation. 
              [default= %default]", metavar="integer")
);

print("Done. like a donut, dufus.")


opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

# opt$network1 <- "Networks/m1_toy.txt"
# opt$network2 <- "Networks/m2_toyMulti.txt"
# opt$bipartite <-"Networks/bipartite_toy.txt" 

multiplex1 <- checkNetworks(opt$network1)
multiplex2 <- checkNetworks(opt$network2)
bipartite <- checkNetworks(opt$bipartite)

if (opt$restart > 1 || opt$restart < 0){
  print_help(opt_parser)
  stop("Restart parameter should range between 0 and 1", call.=FALSE)
}

if (is.null(opt$out)){
  print_help(opt_parser)
  stop("You need to specify a name to be used to generate the output files.", 
       call.=FALSE)
}

MachineCores <- detectCores()

if (opt$cores < 1 || opt$cores > MachineCores){
  print_help(opt_parser)
  stop("The number of cores should be between 1 and the total number of cores of 
       your machine", call.=FALSE)
}


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Data Transformation and associated calculations to apply RWR
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## We transform the input multiplex format to a L-length list of igraphs objects.
## We also scale the weigths of every layer between 1 and the minimun 
## divided by the maximun weight. 

LayersMultiplex1 <- MultiplexToList(multiplex1)
LayersMultiplex2 <- MultiplexToList(multiplex2)

## We create the multiplex objects and multiplex heterogeneous.
MultiplexObject1 <- create.multiplex(LayersMultiplex1)
MultiplexObject2 <- create.multiplex(LayersMultiplex2)
Allnodes1 <- MultiplexObject1$Pool_of_Nodes
Allnodes2 <- MultiplexObject2$Pool_of_Nodes

multiHetObject <- 
      create.multiplexHet(MultiplexObject1,MultiplexObject2,bipartite)


## We have now to compute the transition Matrix to be able to apply RWR

MultiHetTranMatrix <- compute.transition.matrix(multiHetObject)