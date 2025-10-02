#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;  // use the Armadillo library for matrix computations//
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]

arma::mat update_L_admm_cpp(arma::mat& V, arma::mat& A_diff_mat, arma::mat& Lambda, double rho ){
  
  int l = V.n_rows;
  
  for(int d = 0; d < l; d++){

    arma::rowvec V_vec = V.row(d);
    
    arma::rowvec A_diff_vec = A_diff_mat.row(d);
    
    arma::rowvec Lam_vec = Lambda.row(d);
    
    Lambda.row(d) = Lam_vec + (rho*(V_vec - A_diff_vec)) ;
  
  }
  
  return Lambda;
}