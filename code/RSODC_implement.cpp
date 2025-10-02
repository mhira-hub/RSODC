#include <RcppArmadillo.h>
#include <RSODC_function.cpp>
#include <RSODC_update_B.cpp>
#include <RSODC_updateY_admm.cpp>

// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]
List imp_RSODC(
    arma::mat Y, //scoring matrix Y
    arma::mat Z, //a block-diagonal matrix Z by repeating the matrix X along the diagonal d times
    arma::mat H, //centering matrix
    arma::mat X, //data matrix X
    arma::mat W, //dimension reduction matrix B 
    arma::mat V, //difference between two clusters
    arma::mat Lambda, //Lagrangian multiplier
    arma::mat e_mat, //a matrix of residual-like vectors representing all pairwise differences between subjects
    arma::mat I_n, //identity matrix with ones on the diagonal and zeros elsewhere
    double eta, //tuning parameter for group lasso in SODC
    double v, //threshold for update W
    double rho, //parameter for augmented Lagrangian (>0)
    double gamma, //tuning parameter for update V
    arma::vec w_vec , //weight vector in RSODC
    double thre, //threshold for update objective function
    double kuri, //number of repetition for the function
    double kuri_w,//number of repetition for update W
    double kuri_y, //number of repetition for update Y
    double kuri_admm, //number of repetition for update admm
    double eps_primal, //threshold for primal convergence
    double eps_dual  //threshold for secondary convergence
){
  
  arma::mat res_mat(kuri+1, 2, fill::zeros);
  arma::mat res_mat_lag(kuri+1, 2, fill::zeros);

  arma::mat res_y_upd_mat(kuri_admm+1, 3, fill::zeros);
  
  
  double r_norm=0;
  double dual_norm=0;
  
  res_mat(0,1) = rsodc_function_cpp(Y,H, X, W, V,  eta, gamma,  w_vec);
  res_mat_lag(0,1) = rsodc_lag_function_cpp(Y, X, W, H, V, Lambda, e_mat, eta, gamma, rho, w_vec);
  
  for(int ite= 0; ite < kuri ; ite++){

  
    //update W
    W = rsodc_update_B_cpp(Y,X, H, Z, W, V, eta, v, gamma,  w_vec , kuri_w);
    
    res_mat((ite+1),0) = rsodc_function_cpp(Y,H, X, W, V,  eta, gamma,  w_vec);
    res_mat_lag((ite+1),0) = rsodc_lag_function_cpp(Y, X, W, H, V, Lambda, e_mat, eta, gamma, rho, w_vec);
  
    //update Y by ADMM
    List res_admm = rsodc_update_Y_admm_cpp(Y, H, X, W, V, Lambda, e_mat, I_n, eta, v, rho, gamma, w_vec, 
                                   kuri_y, kuri_admm, eps_primal, eps_dual);
    
    Y = as<arma::mat>(res_admm["Y"]);
    V =  as<arma::mat>(res_admm["V"]);
    Lambda =  as<arma::mat>(res_admm["Lambda"]);
    res_y_upd_mat = as<arma::mat>(res_admm["res_lag_mat"]);
    r_norm = as<double>(res_admm["primal_resd"]);
    dual_norm = as<double>(res_admm["secondary_resd"]);
  
  
    res_mat((ite+1),1) = rsodc_function_cpp(Y,H, X, W, V,  eta, gamma,  w_vec);
    res_mat_lag((ite+1),1) = rsodc_lag_function_cpp(Y, X, W, H, V, Lambda, e_mat, eta, gamma, rho, w_vec);
    
    if(ite>10){
      if(res_mat((ite),1) - res_mat((ite+1),1) < thre)break;
    }
    
  }
                      
  
  //return res_mat;
  return List::create(Named("res_mat")=res_mat,
                      Named("res_mat_lag")=res_mat_lag,
                      Named("res_y_mat")=res_y_upd_mat,
                      Named("W")=W,
                      Named("Y")=Y,
                      Named("V")=V,
                      Named("Lambda")=Lambda,
                      Named("primal_resd")=r_norm,
                      Named("secondary_resd")=dual_norm);
  
}
