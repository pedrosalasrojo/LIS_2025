#         Author: Pedro Salas Rojo 
#         Date: 07/2025
#         Dataset: LWS data
#         Name of project: Predict Financial Behavior in Spain

rm(list = ls(all.names = TRUE))  
library(tidyverse) 
library(haven) 
library(partykit)

# Get and clean data ---- 

# Define function to get household-related variables 
# Include in "hhold" all variables  
seth <- function(data_file) {     
  hhold <- c('hid', 'hpopwgt', 'hhtype', 'inum')
  data1 <- read.LIS(data_file, labels = FALSE, vars = hhold)   
  return(data1)   
} 

# Define function to get personal-related variables 
# Include in "hhold" all variables  

setp <- function(data_file){ 
  pers <- c('hid', 'pid', 'sex', 'relation', 'inum', 'age', 'marital',
            'health_c', 'educlev', 'status1', 'ind1_c', 'occ1_c',
            'basb', 'bafr1_c') 
  data2 <- read.LIS(data_file, labels = FALSE, vars = pers)    
  return(data2) 
} 

# Store names of the datasets 
ccyy <- c('es21')  
results <- NA 
  
  # Print name of dataset 
  print(ccyy) 
  
  # Get data with functions previouisly defined 
  data1 <- seth(paste0(ccyy,'h'))  
  data2 <- setp(paste0(ccyy,'p')) 

  # To simplify the analysis now, simply take respondents (relation == 1000) and first
  # imputation (inum == 1). In "real-life" applications please use all imputations and a proper
  # cleaning of the data.

  data1 <- data1 %>%
    filter(inum == 1)     
 
  data2 <- data2 %>%
    filter(age>=25 & age<=75) %>%
    filter(relation == 1000) %>%  
    filter(inum == 1)             
  
  # Merge by ID. Define age and select household heads
  data <- merge(data1, data2, by=c("hid"), all = TRUE, sort=TRUE) 

  # Get na.omit information 
  print(summary(is.na(data))) 
  data <- na.omit(data)
  print(summary(is.na(data))) 

  # See variables
  
  for(i in c("sex", "age", "marital", "health_c", "educlev", "status1", "ind1_c", 
             "occ1_c", "basb", "bafr1_c")){
    print(i)
    print(table(data[[i]]))
  }
  
  # Make basb binary, 1 = saves, 0 = does not save
  data$saves <- ifelse(data$basb==20, 1, 0)
  
  #################################
  #
  # Get Tree ----
  #
  #################################
  
  model <- saves ~ age + sex + factor(marital) + factor(health_c) + 
    factor(educlev) + factor(status1) + factor(ind1_c) + factor(occ1_c) 
  
  # This way you get the tree. Play with the parameters and get different trees. Try to understand
  # why different parameters give different trees (and why sometimes give the same tree).

  tree <- partykit::ctree(model,
                          data = data, 
                          control = ctree_control(testtype = "Bonferroni", 
                                                  teststat = "quad", 
                                                  alpha = 0.01,
                                                  minbucket = 100,
                                                  minsplit = 300,
                                                  maxdepth = 3))
  
  # Predict income and groups
  data$y_tilde <- predict(tree, type="response")
  data$groups <- predict(tree, type="node")
  table(data$groups)
  
  #################################
  #
  # Plot Tree ---- This codes is much nicer than the default. Modify at will!
  #
  #################################
  
  ct_node <- as.list(tree$node)
  data$groups <- predict(tree, type = "node")           # This line predicts "nodes". Predict "anything" from the tree and the random forest
  
  pred <- data %>%
    group_by(groups) %>%
    mutate(x = stats::weighted.mean(saves)) %>%
    summarise_all(funs(mean), na.rm = TRUE)    %>%
    ungroup() %>%
    dplyr::select(groups, x) 
  
  a <- data %>%
    mutate(m = stats::weighted.mean(x = saves))
  
  mean_pop <- round(mean(a$m),3)
  
  pred <- as.data.frame(pred)
  qi <- pred
  
  for (t in 1:length(qi[,1])){
    typ<-as.numeric(names(table(data$groups)[t]))
    qi[t,2]<-length(data$groups[data$groups==typ])/length(data$groups) 
  }
  
  for(t in 1:nrow(pred)) {
    ct_node[[pred[t,1]]]$info$prediction <- as.numeric(paste(format(round(pred[t, -1], 
                                                                          digits = 3), nsmall = 2)))
    ct_node[[pred[t,1]]]$info$nobs       <- as.numeric(paste(format(round(100*qi[t, -1]  , 
                                                                          digits = 2), nsmall = 2)))
  }
  
  tree$node <- as.partynode(ct_node)
  
  print(plot(tree,  terminal_panel=node_terminal, 
             tp_args = list(FUN = function(node) 
               c("Exp. outcome",node$prediction, "Pop. Share (%)", node$nobs))))
  
  #################################
  #
  # Get Random Forest and Variable importance ----
  #
  #################################
  
  forest <- partykit::cforest(model,
                              data = data,
                              ntree = 100,
                              mtry = 5,
                              trace = FALSE,
                              control = ctree_control(testtype = "Bonferroni",
                                                      teststat = "quad",
                                                      mincriterion = 0,
                                                      minbucket = 10))
  
  imp <- partykit::varimp(forest)
  relimp <- round(100*imp/max(imp), 2)
  relimp <- relimp[order(-relimp)]
  print(relimp)
