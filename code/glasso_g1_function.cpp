#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;  // use the Armadillo library for matrix computations//
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]
double glasso_g1_function_cpp(arma::vec y,arma::mat X,arma::vec beta, double lambda){
  
  double ob1 = (accu(square(y - (X*beta))));
  ob1 = 0.5*ob1;
  
  double ob2 = lambda*sqrt(accu(square(beta)));
  
  
  double res = ob1 + ob2;
  
  return res;
  
}