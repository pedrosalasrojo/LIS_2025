#         Author: Pedro Salas Rojo 
#         Date: 07/2025
#         Dataset: LIS data
#         Name of project: COMPARE OLS VS LASSO PERFORMANCE

rm(list = ls(all.names = TRUE))  
library(tidyverse) 
library(haven) 
library(caret)
library(xtable) 
library(glmnet)

# Get and clean data ---- 

# Define function to get household-related variables 
# Include in "hhold" all variables needed from household files

seth <- function(data_file) {     
  hhold <- c('hid', 'hpopwgt', 'hilabour')
  data1 <- read.LIS(data_file, labels = FALSE, vars = hhold)   
  return(data1)   
} 

# Define function to get personal-related variables 
# Include in "pers" all variables from personal files

setp <- function(data_file){ 
  pers <- c('hid', 'pid', 'pilabour', 'sex', 'age', 'marital', 'status1',
            'disabled', 'educlev', 'lfs', 'ind1_c', 'occ1_c') 
  data2 <- read.LIS(data_file, labels = FALSE, vars = pers)    
  return(data2) 
} 

# Store dataset names, as an example we use Germany 2020 from LIS
ccyy <- c('de20')  
  
# Get data LWS with functions previouisly defined 
data1 <- seth(paste0(ccyy,'h'))  
data2 <- setp(paste0(ccyy,'p')) 
  
# Merge by ID 
data <- merge(data1, data2, by=c("hid"), sort=TRUE) 
  
# Arrange the data, just fix age, get positive pilabour and adjust to USD2017 with PPP

data <- data %>% 
    dplyr::filter(age>=30 & age<=60 & pilabour > 0) %>%
    dplyr::mutate(pilabour = pilabour/0.799178,
                  age5 = cut(age, breaks = seq(min(age), max(age), 5), 
                             right = FALSE, ordered_result = TRUE),
                  age5num = as.numeric(age5)) 
  
# Get missing information. Drop item non responses.
print(summary(is.na(data))) 
data <- na.omit(data)
print(summary(is.na(data))) 
  
# Get subsample (the algorithm runs faster with a smaller sample)
set.seed(3)
data <- data[sample(1:nrow(data), 4000, replace = FALSE),]  
  
# See variables, check categories and levels make sense with METIS documentation
  
for(i in c("sex", "age5num", "marital", "disabled", "status1", "educlev", "ind1_c", "occ1_c")){
    print(i)
    print(table(data[[i]]))
}
  
  #################################
  #
  # OLS output ----
  #
  #################################
  
  # Define model, run OLS, print results and store values of coefficients
  model <- pilabour ~ sex + factor(marital) + factor(educlev) + factor(age5num) +
                    factor(ind1_c) + factor(occ1_c) + disabled + factor(status1)
  
  reg <- lm(model, data = data)
  print(summary(reg))
  coef_df <- summary(reg)$coefficients
  
  # Run Cross Validation (alpha 1, lambda 0, OLS). This way, we easily get a simple measure of OOS RMSE
  ols_tr <- caret::train(model,
                         data = data,
                         method = "glmnet",
                         trControl = trainControl(method = "cv", number = 3, 
                                                    verboseIter = TRUE,   savePredictions = "all"),
                         tuneGrid = expand.grid(alpha = 1,         
                                                lambda = 0))          # Lambda 0 is equivalent to OLS
  
  results <- ols_tr[["results"]]
  lambda <- ols_tr[["bestTune"]][["lambda"]]         
  rmse_ols <- round(mean(ols_tr[["resample"]][["RMSE"]]), 2) 
  
  print(paste0("The RMSE of the OLS model is: ", rmse_ols))

  #################################
  #
  # LASSO regularization ----
  #
  #################################

  set.seed(3)
 
  # Define model and lambda
  model <- pilabour ~ sex + factor(marital) + factor(educlev) + factor(age5num) +
                    factor(ind1_c) + factor(occ1_c) + disabled + factor(status1)

  # Set lambda to 225, as an example. Use "any" lambda you like, just to try. 
  lambda_try <- 225       
  
  # Use caret package to check RMSE associated with your lambda_try
  lasso_tr <- caret::train(model,
                           data = data,
                           method = "glmnet",
                           trControl = trainControl(method = "cv", number = 2, 
                                                    verboseIter = TRUE,   savePredictions = "all"),
                           tuneGrid = expand.grid(alpha = 1,         
                                                  lambda = lambda_try))
  
  print(paste0("The RMSE of this model, with a lambda of ",lambda_try,", is: ", round(mean(lasso_tr[["resample"]][["RMSE"]]), 2)))
  
  # Train LASSO to select opt* lambda (the one leading to the "best" OOS RMSE, the best prediction)
  # Define LASSO setting 
  dep <- data$pilabour
  vec <- model.matrix(~ sex + factor(marital) + factor(ind1_c) + factor(occ1_c) + disabled  + factor(status1) + 
                      factor(educlev) + factor(age5num), data)

  # Define optimum lambda grid (you can also explore by trying your own values... this is faster)
  exploremodel <- glmnet::cv.glmnet(x = vec, y = dep, alpha = 1)
  range(exploremodel$lambda)
  lambda_range <- exp(seq(log(min(exploremodel$lambda)), log(max(exploremodel$lambda)), length.out = 50))
  print(lambda_range)

  # Define model and run tuning
  model <- pilabour ~ sex + factor(marital) + factor(educlev) + factor(age5num) + factor(status1) +
                      factor(ind1_c) + factor(occ1_c) + disabled

  lasso_tr <- caret::train(model,
                           data = data,
                           method = "glmnet",
                           trControl = trainControl(method = "cv", number = 3, 
                                                    verboseIter = TRUE,   savePredictions = "all"),
                           tuneGrid = expand.grid(alpha = 1,         
                                                  lambda = lambda_range))
  
  # Plot tuning. Check that the minimum RMSE corresponds to the lambda value the algorithm has selected
  print(plot(y = lasso_tr$results$RMSE, x = lasso_tr$results$lambda,
             main = "RMSE by lambda value", xlab = "Lambda", ylab = "RMSE"))
  print(abline(v = lasso_tr[["bestTune"]][["lambda"]]))  
  lambda <- lasso_tr[["bestTune"]][["lambda"]]         
  rmse_lasso <- round(mean(lasso_tr[["resample"]][["RMSE"]]), 2) 
    
  # Plot LASSO results. 
  lasso_mod <- glmnet(vec, dep, alpha=1)
  plot(lasso_mod, xvar = "lambda")
  abline(v=log(lambda), lty="dashed", col="black")
  coeff2 <- lasso_mod$beta 
  
  print(paste0("Out of sample RMSE from OLS is: ", rmse_ols))
  print(paste0("Out of sample RMSE from LASSO is: ", rmse_lasso))
  print(paste0("LASSO improves RMSE by ", round((rmse_ols - rmse_lasso)/rmse_ols*100, 2), "%"))

  # Get LASSO and get coefficients. You can use the lasso_mod object to predict and work as you need.
  lasso_mod <- glmnet(vec, dep, alpha=1, lambda = lambda)
  coeff_lasso <- lasso_mod$beta 
