// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
NumericVector weight_function_rcpp(const arma::mat& x, double mu, int dd) {
  int n = x.n_rows;
  arma::mat w(n, n, fill::zeros);
  arma::mat d(n, n, fill::zeros);
  
  // Simultaneous calculation of distance and weight matrices
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double dist_sq = arma::accu(arma::square(x.row(i) - x.row(j)));
      double weight = std::exp(-mu * dist_sq);
      d(i, j) = std::sqrt(dist_sq);  // keep sqrt distance
      d(j, i) = d(i, j);
      w(i, j) = weight;
      w(j, i) = weight;
    }
  }
  
  // Neighborhood restriction
  if (dd > 0) {
    for (int i = 0; i < n; ++i) {
      arma::rowvec di = d.row(i);
      arma::uvec sorted_idx = sort_index(di);
      arma::uvec keep = sorted_idx.head(dd + 1);
      
      for (int j = 0; j < n; ++j) {
        if (!any(keep == j)) {
          w(i, j) = 0;
          w(j, i) = 0;
        }
      }
    }
  }
  
  // Convert the upper triangular portion into a vector
  int len = n * (n - 1) / 2;
  NumericVector w_vec(len);
  int cnt = 0;
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      w_vec[cnt++] = w(i, j);
    }
  }
  
  return w_vec;
}
