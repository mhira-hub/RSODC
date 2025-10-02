###############################################################
#function for cross-validation

#generate the data list for cross validation based on Kappa
#and calculate Kappa
###############################################################


source("datamake_e_mat.R")
source("clust_kappa.R")
sourceCpp("weight_calculation.cpp")
sourceCpp("RSODC_implement.cpp")

rsodc_cv <- function(
  X, #data
  d, #(k-1)dimension
  n_seed, ## for set.seed()
  n_cv, #number of calculation of kappa
  eta, #tuning parameter for group lasso in SODC
  v, #threshold for update W
  rho, #parameter for augmented Lagrangian (>0)
  gamma, #tuning parameter for update V
  mu_w, #parameter for w_vec
  mu_dd, #parameter for w_vec, # of nearest neighbors
  thre, #threshold for update objective function
  kuri, #number of repetition for the function
  kuri_w, #number of repetition for update W
  kuri_y, #number of repetition for update Y
  kuri_admm, #number of repetition for update admm
  eps_primal, #threshold for primal convergence
  eps_dual   #threshold for secondary convergence
){
  
  #vec for result of each kappa
  res_kappa_vec <- rep(NA, n_cv) 
  
  res_W_list <- rep(list(NA), n_cv)
  
  res_mat_list <- res_W_list
  
  res_V_list <- res_W_list
  
  for(kk_cv in 1:n_cv){  
    #1.split X data and datamake for calculate Kappa
    #list for split 2 data
    X_list <- rep(list(NA),2)
    Z_list <- X_list
    W_list <- X_list
    Y_list <- X_list
    H_list <- X_list
    e_mat_list <- X_list
    V_list <- X_list
    Lambda_list <- X_list
    e_mat_list <- X_list
    I_n_list <- X_list
    
    set.seed(n_seed+kk_cv)
    
    #split X
    if(nrow(X)%%2==0){
      n_split <- sample(1:nrow(X),nrow(X)/2,replace = FALSE)
    }else{
      n_split <- sample(1:nrow(X),(nrow(X)/2+1),replace = FALSE)
      
    }
    
    X_list[[1]] <- X[n_split,] 
    X_list[[2]] <- X[-n_split,]
    
    X_n_vec <- c(nrow(X_list[[1]]),nrow(X_list[[2]]))
    
    
    p <- ncol(X)
  
    #2. calculate SODC with split data
    #list for calculated W
    W_list_res <- rep(list(NA),2)
    
    W_taisho <- rnorm((p*d),0,1)
   
    W_taisho <- matrix(W_taisho,p,d)
    
    
    list_res <-  W_list_res
    
    V_list_res <- W_list_res
    
    #vec for caluclate twice
    TRUE_vec <- rep(NA,2)
    
    
    for(o in 1:2){
      
      X_taisho <- as.matrix(X_list[[o]])
      X_taisho <- scale(X_taisho )
      
      
      #w_vec
      w_vec <- weight_function_rcpp(X_taisho,mu=mu_w, dd = mu_dd)
      w_vec <- as.vector(w_vec)
      
      #Y
      ei <- eigen(X_taisho%*%t(X_taisho))
      Y_taisho<- as.matrix(ei$vectors[,1:d])
      
     
      #generate Z
      Z_taisho <- matrix(0,X_n_vec[o]*d,p*d)
      
      for(i in 1:d){
        
        Z_taisho[(((i-1)*X_n_vec[o])+1):(i*X_n_vec[o]),(((i-1)*p)+1):(i*p)] <- X_taisho
        
        
      }
  
      
      Z_2 <- sqrt(1/2)*Z_taisho
      
      
      #H
      H_taisho  <- matrix(0,X_n_vec[o],X_n_vec[o])
      diag( H_taisho ) <- 1
      
      #numuber of the difference of y_i and y_j
      row_Y_diff <- (X_n_vec[[o]]*(X_n_vec[[o]]-1))/2
      
      
      #e_mat
      e_mat_taisho <-  datamake_emat(X_n_vec[[o]])
      e_mat_taisho <- as.matrix( e_mat_taisho)
      
      #V
      V_taisho <- e_mat_taisho%*%Y_taisho
      V_taisho <- as.matrix( V_taisho)
      
      #L
      Lambda_taisho <- matrix(0,row_Y_diff,d)
      
      #identity matrix
      I_n_taisho <- matrix(0,X_n_vec[[o]],X_n_vec[[o]])
      diag(I_n_taisho) <- 1
      
      
    
      res <- try({
        
        #calculate W from the objective function
        res_list <-imp_RSODC(
          Y_taisho,
          Z_taisho,
          H_taisho,
          X_taisho,
          W_taisho,
          V_taisho,
          Lambda_taisho,
          e_mat_taisho,
          I_n_taisho,
          eta, #tuning parameter for group lasso in SODC
          v, #threshold for update W
          rho, #parameter for augmented lagrandian (>0)
          gamma, #tuning parameter for update V
          w_vec, #weight vector in SODC
          thre, #threshold for update objective function
          kuri, #number of repetition for the function
          kuri_w, #number of repetition for update W
          kuri_y, #number of repetition for update Y
          kuri_admm, #number of repetition for update admm
          eps_primal, #threshold for primal convergence
          eps_dual   #threshold for secondary convergence
        )
        
      }, silent=TRUE)
      
      if(!inherits(res, "try-error")){ 
      
        W_list_res[[o]] <- res_list$W
        
        list_res[[o]] <- res_list$res_mat
        
        V_list_res[[o]] <- res_list$V
        
        TRUE_vec[o] <- TRUE
      }  
    }
    
    res_W_list[[kk_cv]] <-  W_list_res
    
    res_mat_list[[kk_cv]] <- list_res
    
    res_V_list[[kk_cv]] <- V_list_res
    
    
    if(sum(!is.na(TRUE_vec)) ==2){

      #3. calculate Kappa
      #generate clust1, clust2 from W
      clus1 <- which(W_list_res[[1]][,1] != 0)
      clus2 <- which(W_list_res[[2]][,1] != 0)
      
      res_kappa_vec[kk_cv] <- clust.kappa(clus1,clus2,p)
      
    }
    cat(paste0('calculate_kappa finish ', kk_cv , '\n'))
   
  }   
  
  res_kappa_vec2 <- res_kappa_vec[(!is.na(res_kappa_vec))]
 
  res <- sum(res_kappa_vec2)/length(res_kappa_vec2)
    
  res_list <- list("res"=res, "res_kappa_vec"=res_kappa_vec, "res_mat"=res_mat_list, "V"=res_V_list, "W"=res_W_list)
  
  return(res_list)
  
}
  