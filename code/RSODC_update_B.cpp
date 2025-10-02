
#include <RcppArmadillo.h>
#include <SODC_update_beta.cpp>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]


arma::mat rsodc_update_B_cpp(arma::mat Y,arma::mat X, arma::mat H, arma::mat Z,arma::mat W, arma::mat V ,double eta,double v, double gamma, arma::vec w_vec , int kuri_w){
  
  int p = W.n_rows;
  
  int d = Y.n_cols;
  
  int col_z = Z.n_cols;
  
  
  arma::mat Y_2 = sqrt(0.5)*Y;
  
  arma::mat Z_2 = sqrt(0.5)*Z;
  
  
  arma::mat res_mat_w((kuri_w +1),1, fill::zeros );
  
  arma::vec g_vec(d, fill::zeros);
  
  g_vec.row(0) = 0;
  
  //pick up the column number 
  for(int s = 1; s < d ; s++){
    
    g_vec.row(s) = ((p*s));
  
  }
  
  res_mat_w(0,0) = rsodc_function_cpp(Y, H, X,  W, V, eta, gamma, w_vec);
  
  for(int ite_w = 0; ite_w < kuri_w ; ite_w++){

  
  
    for(int int_j = 0; int_j < p; int_j++){

    
      arma::mat ZZ;
      arma::mat ZZ_exp;
      
      //extract the target column 
        arma::vec jj= g_vec + int_j;
      
      //make matrix ZZ
      for(int int_z =0;int_z < jj.n_rows ; int_z++){
        
        double n_j = 0;
        n_j = jj(int_z,0);
        
        ZZ = join_horiz(ZZ,Z_2.col(n_j));
        
      }
      
      //make matrix ZZ_exp
      arma::vec n_jexp;
      
      for(int int_zexp=0; int_zexp < col_z; int_zexp++ ){
        
        n_jexp = arma::zeros(jj.n_rows);
        
        for(int int_gvec=0; int_gvec< jj.n_rows;  int_gvec++){
          
          if(int_zexp != jj(int_gvec)){
            
            n_jexp(int_gvec) = 1;
            
          } 
        }
        if(accu(n_jexp)==jj.n_rows){
          
          ZZ_exp = join_horiz(ZZ_exp, Z_2.col(int_zexp));
          
        }
      }
      
      //make WW
      arma::mat WW = W.row(int_j);
      
      WW = vectorise(WW);
      
      
      //make WW_exp
      arma::mat WW_exp;
      
      for(int int_jvec=0; int_jvec< p;  int_jvec++){
        
        if(int_jvec != int_j){
          
          WW_exp = join_cols(WW_exp, W.row(int_jvec));
          
        }
      } 
      
      //vectorize Y
      arma::vec y_vec=vectorise(Y_2);
      
      //vectorize WW_exp
      arma::vec WW_exp_vec = vectorise(WW_exp);
      
      //calcuate r
      arma::vec r = y_vec - ZZ_exp*WW_exp_vec;
      
      //update W 
      W.row(int_j) = trans(update_beta_cpp(r, ZZ, WW, eta, v));
    
    }
    
    res_mat_w(((ite_w)+1),0) = rsodc_function_cpp(Y, H, X,  W, V, eta,  gamma,  w_vec);
    
    if(ite_w > 0){
      
      if(abs(res_mat_w((ite_w),0) - res_mat_w((ite_w)+1,0)) < 0.01) break;
    }
    
  
  }
  
  return  W;
  
}