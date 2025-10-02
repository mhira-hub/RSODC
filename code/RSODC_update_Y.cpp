#include <RcppArmadillo.h>



// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]

arma::mat rsodc_update_Y_cpp(arma::mat Y,arma::mat H,arma::mat X,arma::mat W,arma::mat V,arma::mat Lambda,
                                      arma::mat e_mat,arma::mat I_n, arma::vec w_vec, double eta,double rho, double gamma, double kuri_y){ 

  int l = V.n_rows;
  
  int p = V.n_cols;
  
  int n = Y.n_rows;
  
  arma::mat ob2_y(n,p);
  
  arma::mat ob3_y(n,p);
  
  arma::mat C(n,n);
  
  arma::mat D;
  
  //term1
  arma::mat B = H*X*W;
  
  arma::mat res_y_mat(kuri_y, 1, fill::zeros);
  
  for(int ite_y = 0; ite_y < kuri_y; ite_y++){
  
    //term2
    for(int s = 0; s < l ; s++){
    
    
      ob2_y = ob2_y +  trans(e_mat.row(s))*Lambda.row(s) ;
    
    }
    
    //term3
    for(int s = 0; s < l ; s++){
      
      ob3_y = ob3_y +  (rho*trans(e_mat.row(s))*V.row(s));
    
    }
    
    //term4
    for(int ss=0; ss<l ; ss ++){
      
      C =  C + rho/2*(trans(e_mat.row(ss))*e_mat.row(ss));
      
    }
    
    arma::vec eigval;
    arma::mat eigvec;
    
    arma:: eig_sym(eigval,eigvec, C);
    double omega = max(eigval);
    
    arma::mat tm5 = ((2*omega*I_n)-(2*C));
    
    arma::mat ob4_y = (tm5*Y);
    
    D = (B + ob2_y + ob3_y +ob4_y);
    
    arma::mat U, R;
    arma::vec S;
    
    arma::svd_econ(U,S,R,D);
    
    Y = U*trans(R);
    
    res_y_mat(ite_y, 0) = rsodc_lag_function_cpp(Y, X, W, H, V, Lambda, e_mat, eta, gamma, rho, w_vec);
    
    if(ite_y > 11){
      if(( res_y_mat(ite_y, 0)  -  res_y_mat((ite_y-1), 0) ) < 0.01 ) break;
    }
   
  }
  
  return Y;

}