#include <RcppArmadillo.h>
#include <glasso_g1_function.cpp>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;  // use the Armadillo library for matrix computations//
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]

arma::vec update_beta_cpp(arma::vec y,arma::mat X,arma::vec  beta,double lambda,double v){
  
  //int p = X.n_cols;
  
  arma::vec res_mat(1000,fill::zeros);
  
  for(int bb = 0; bb < 1000 ; bb++){
    //int bb = 0;
    
    arma::vec gamma = beta + (v*trans(X)*(y- (X*beta)));
    
    if(sqrt(accu(square(gamma))) >= (v*lambda)){
      
      beta = gamma*(1 - ((v*lambda)/(sqrt(accu(square(gamma))))));
    
    }
    if(sqrt(accu(square(gamma))) < (v*lambda)){
      
      beta.zeros();
      
    }
  
    res_mat(bb) = glasso_g1_function_cpp(y, X, beta, lambda);
    
    if(bb>5){
      if(res_mat(bb-1) - res_mat(bb) < 0.001) break;
      
    }
  }
  
  return beta;
}