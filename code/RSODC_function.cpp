#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;  // use the Armadillo library for matrix computations//
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]
double rsodc_function_cpp(arma::mat Y,arma::mat H,arma::mat X, arma::mat W,arma::mat V, double eta, double gamma, arma::vec w_vec){
  
  int p = W.n_rows;

  int o = V.n_rows;

  
  //term1
  double ob1 =  0.5*(accu(square(Y - (H*X*W))));
  
  //term2
  arma::vec ob2(p, fill::zeros);
  
  for(int j = 0; j < p ; j++){
 
    ob2.row(j) = eta*sqrt(sum(square(W.row(j))));
  
  }
  
  //term3
  arma::vec v_vec(o, fill::zeros);
  
  for(int  s= 0; s < o ; s++){
    
    v_vec.row(s) = w_vec.row(s)*sqrt(sum(square(V.row(s))));
    
  }
  

  double ob3 = gamma*sum(v_vec);
  
  double res = ob1 + accu(ob2) + ob3;
  
  return res;
  
  
}

// [[Rcpp::export]]
double rsodc_lag_function_cpp(arma::mat Y,arma::mat X,arma::mat W,arma::mat H,arma::mat V,arma::mat Lambda,
                                 arma::mat e_mat,double eta,double gamma,double rho,arma::vec w_vec){
  
  int p = W.n_rows;
  
  int o = V.n_rows;
  
  arma::mat Y_diff_mat = e_mat*Y;
  
  //term1
  double ob1 = 0.5*(accu(square(Y - (H*X*W))));
  
  //term2
  arma::vec ob2(p, fill::zeros);
  
  for(int j = 0; j < p ; j++){
    
    ob2.row(j) = eta*sqrt(sum(square(W.row(j))));
  }
  
  //term3
  arma::vec v_vec(o, fill::zeros);
  
  for(int  s= 0; s < o ; s++){
    
    
    v_vec.row(s) = w_vec.row(s)*sqrt(sum(square(V.row(s))));
    
  }
  
  double ob3 = gamma*sum(v_vec);
  
  //term4
  double ob4;
  
  for(int s=0; s < o ; s++){
    
    ob4 = accu(trans(Lambda.row(s))*(V.row(s)-Y_diff_mat.row(s)));
    
  }
  
  //term5
  double tm5;
  for(int s=0; s < o ; s++){
    
    tm5 = accu( V.row(s)-Y_diff_mat.row(s));
    
  }
  
  double ob5 = rho*0.5*tm5;
  
  double res = ob1 + accu(ob2) + ob3 + ob4 + ob5;
  
  return res ;
  
}