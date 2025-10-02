// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>

using namespace arma;
using namespace Rcpp;
using namespace RcppParallel;


struct VUpdateWorkerNonZero : public Worker {
  RMatrix<double> V;
  const RMatrix<double> Q;
  const RVector<double> w_vec;
  const std::vector<int>& indices;
  double gamma, rho;
  
  VUpdateWorkerNonZero(NumericMatrix V, NumericMatrix Q, NumericVector w_vec,
                       const std::vector<int>& indices, double gamma, double rho)
    : V(V), Q(Q), w_vec(w_vec), indices(indices), gamma(gamma), rho(rho) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    int p = V.ncol();
    for (std::size_t i = begin; i < end; ++i) {
      int d = indices[i];
      double w = w_vec[d];
      double m = gamma * w / rho;
      
      std::vector<double> v_l(p), q_l(p);
      for (int j = 0; j < p; ++j) {
        v_l[j] = V(d, j);
        q_l[j] = Q(d, j);
      }
      
      arma::rowvec v = arma::rowvec(v_l);
      arma::rowvec q = arma::rowvec(q_l);
      arma::rowvec s = v - m * (v - q);
      arma::rowvec v_new = (norm(s, 2) > m) ? s * (1 - m / norm(s, 2)) : arma::rowvec(p, fill::zeros);
      
      for (int j = 0; j < p; ++j) {
        V(d, j) = v_new[j];
      }
    }
  }
};

// [[Rcpp::export]]
arma::mat rsodc_update_V_cpp(
    arma::mat Y,
    arma::mat X,
    arma::mat W,
    arma::mat H,
    arma::mat V,
    arma::mat Lambda,
    arma::mat e_mat,
    arma::mat q_mat,
    arma::vec w_vec,
    double eta,
    double gamma, double rho) {
  
  int adiff = V.n_rows;
  int p = V.n_cols;
  arma::mat res_v_vec(1000, 1, fill::zeros);

  std::vector<int> nonzero_indices;
  for (int i = 0; i < w_vec.n_elem; ++i) {
    if (w_vec[i] != 0.0) {
      nonzero_indices.push_back(i);
    }
  }
  
 
  Rcpp::NumericMatrix V_rcpp = Rcpp::wrap(V);
  Rcpp::NumericMatrix Q_rcpp = Rcpp::wrap(q_mat);
  Rcpp::NumericVector wvec_rcpp = Rcpp::wrap(w_vec);
  
  for (int ite_v = 0; ite_v < res_v_vec.n_rows; ++ite_v) {

    VUpdateWorkerNonZero worker(V_rcpp, Q_rcpp, wvec_rcpp, nonzero_indices, gamma, rho);
    parallelFor(0, nonzero_indices.size(), worker);
    

    arma::mat V_out(adiff, p);
    for (int i = 0; i < adiff; ++i) {
      for (int j = 0; j < p; ++j) {
        V_out(i, j) = V_rcpp(i, j);
      }
    }
    
    res_v_vec(ite_v, 0) = rsodc_lag_function_cpp(Y, X, W, H, V_out, Lambda, e_mat, eta, gamma, rho, w_vec);
    
    if (ite_v > 0 && std::abs(res_v_vec(ite_v, 0) - res_v_vec(ite_v - 1, 0)) < 0.01) break;
  }
 
  
  //
  for (int idx : nonzero_indices) {
    for (int j = 0; j < p; ++j) {
      V(idx, j) = V_rcpp(idx, j);
    }
  }
  return V;
 
}
