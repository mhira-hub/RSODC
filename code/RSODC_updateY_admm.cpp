
// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>

#include <RSODC_update_Y.cpp>
#include <RSODC_update_V.cpp>
#include <RSODC_update_L.cpp>



using namespace arma;
using namespace Rcpp;
using namespace std;


// [[Rcpp::export]]

List rsodc_update_Y_admm_cpp(
    arma::mat Y,
    arma::mat H,
    arma::mat X,
    arma::mat W,
    arma::mat V,
    arma::mat Lambda,
    arma::mat e_mat, //a matrix of residual-like vectors representing all pairwise differences between subjects
    arma::mat I_n,//identity matrix with ones on the diagonal and zeros elsewhere
    double eta, //tuning parameter for group lasso in SODC
    double v, //threshold for update W
    double rho, //parameter for augmented lagrandian (>0)
    double gamma, //tuning parameter for update V
    arma::vec w_vec , //weight vector in SODC
    double kuri_y, //number of repetition for update Y
    double kuri_admm, //number of repetition for update admm
    double eps_primal, //threshold for primal convergence
    double eps_dual  //threshold for secondary convergence
){

  arma::mat res_y_upd_mat(kuri_admm+1, 3, fill::zeros);
  arma::mat V_prev(V.n_rows, V.n_cols, fill::zeros);
  double r_norm = 0;
  double dual_norm =0;
  
  //update Y by ADMM
  for(int ite2 = 0; ite2 < kuri_admm ; ite2++){
    
    //update Y
    Y = rsodc_update_Y_cpp(Y, H, X, W, V, Lambda, e_mat, I_n,  w_vec,  eta, rho,  gamma,  kuri_y);
    
    res_y_upd_mat(ite2,0) = rsodc_lag_function_cpp(Y, X, W, H, V, Lambda, e_mat, eta, gamma, rho, w_vec);
    
    //update V
    
    arma::mat V_prev = V;
    
    arma::mat Y_diff_mat = e_mat*Y;
    
    arma::mat q_mat = Y_diff_mat - ((1.0/rho)*Lambda);
    
    V = rsodc_update_V_cpp(Y, X, W, H, V, Lambda, e_mat, q_mat, w_vec, eta, gamma, rho);
    
    res_y_upd_mat(ite2,1) = rsodc_lag_function_cpp(Y, X, W, H, V, Lambda, e_mat, eta, gamma, rho, w_vec);
    
    
    //upadae Lambda
    Lambda = update_L_admm_cpp(V, Y_diff_mat, Lambda,  rho );
    
    res_y_upd_mat(ite2,2) = rsodc_lag_function_cpp(Y, X, W, H, V, Lambda, e_mat, eta, gamma, rho, w_vec);
    
    
    
    //convergence condition
    int n = X.n_rows;
    int adiff = V.n_rows;
    double r_norm_sq = 0;
    double dual_norm_sq = 0;
    
    for (int d = 0; d < adiff; ++d) {
      // primary residual
      rowvec r_l = V.row(d) - Y_diff_mat.row(d);
      r_norm_sq += dot(r_l, r_l);
      
      // secondary residual
      rowvec dual_l = rho * (V.row(d) - V_prev.row(d));
      dual_norm_sq += dot(dual_l, dual_l);
    }
    
    r_norm = std::sqrt(r_norm_sq);
    
    r_norm = r_norm/n; 
    
    double dual_norm = std::sqrt(dual_norm_sq);
    
    dual_norm = dual_norm/n; 
    
    bool converged = (r_norm <= eps_primal) && (dual_norm <= eps_dual);
    
    
    
    //convergence
    if(converged){
      
      break;
      
    }
    
  }
  
  
  return List::create(Named("res_lag_mat")=res_y_upd_mat,
                      Named("Y")=Y,
                      Named("V")=V,
                      Named("V_prev")=V_prev,
                      Named("Lambda")=Lambda,
                      Named("primal_resd")=r_norm,
                      Named("secondary_resd")=dual_norm);
  
}  