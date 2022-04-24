#include "layers.h"
#include<cmath>
#include <iostream>

using namespace std;

void forward_fcc(vector<float> &x, vector<float> &w, vector<float> &y, vector<float> &b, int xdim, int ydim){

    float xbuf[xdim];
    float wbuf[ydim][xdim];
    float ybuf[ydim];
    float bbuf[ydim];

    for(int i=0;i<xdim;i++){
        xbuf[i] = x[i];
    }

    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            wbuf[i][j] = w[i*xdim+j];
        }
    }

    for(int i=0;i<ydim;i++){
        bbuf[i] = b[i];
    }

    for(int i=0;i<ydim;i++){
        ybuf[i] = bbuf[i];
        for(int j=0;j<xdim;j++){
            ybuf[i] += xbuf[j]*wbuf[i][j];
        }
    }



    for(int i=0;i<ydim;i++){
        y[i] = ybuf[i];
    }

}

void forward_conv(vector<float> &x, vector<float> &w, vector<float> &y, vector<float> &b, int F, int C, int H, int W, int FH, int FW){
    
    

    float xbuf[C][H][W];
    float wbuf[F][C][FH][FW];

    int outH=H-FH+1;
    int outW=W-FW+1;

    float ybuf[F][outH][outW];
    float bbuf[F];

    for(int i=0;i<C;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                xbuf[i][j][k] = x[i*H*W+j*W+k];
            }
        }
        
    }
    
//    for(int i=0;i<C;i++){
//        for(int j=0;j<H;j++){
//            for(int k=0;k<W;k++){
//                cout << xbuf[i][j][k] << ' ';
//            }
//        }
//
//    }
//    cout <<'\n';
//    for(int i=0;i<F;i++){
//        for(int j=0;j<C;j++){
//            for(int k=0;k<FH;k++){
//                for(int l=0;l<FW;l++){
//                    wbuf[i][j][k][l] = w[i*C*FH*FW+j*FH*FW+k*FW+l];
//                }
//            }
//        }
//    }



    for(int i=0;i<F;i++){
        for(int j=0;j<C;j++){
            for(int k=0;k<FH;k++){
                for(int l=0;l<FW;l++){
                    wbuf[i][j][k][l] = w[i*C*FH*FW+j*FH*FW+k*FW+l];
                }
            }
        }
    }

    for(int i=0;i<F;i++){
        bbuf[i] = b[i];
    }

    for(int f=0;f<F;f++){
        for(int c=0;c<C;c++){
            for(int h=0;h<outH;h++){
                for(int w=0;w<outW;w++){
                    ybuf[f][h][w]=bbuf[f];        
                    for(int fh=0;fh<FH;fh++){
                        for(int fw=0;fw<FW;fw++){
                            ybuf[f][h][w] += xbuf[c][h+fh][w+fw]*wbuf[f][c][fh][fw];
                        }
                    }
                }
            }
        }
    }
    
    
    for(int i=0;i<F;i++){
        for(int j=0;j<outH;j++){
            for(int k=0;k<outW;k++){
                y[i*outH*outW+j*outW+k]= ybuf[i][j][k];
                
            }
        }
    }

}

void backward_fcc(vector<float> &x, vector<float> &w, vector<float> &dx, vector<float> &dy,vector<float> &dw, vector<float> &db, int xdim,int ydim){
    //compute gradient of activations
    
    float xbuf[xdim];
    float wbuf[ydim][xdim];
    float dxbuf[xdim];
    float dwbuf[ydim][xdim];
    float dybuf[ydim];
    float dbbuf[ydim];

    for(int i=0;i<xdim;i++){
        xbuf[i] = x[i];
    }


    for(int i=0;i<ydim;i++){
        dbbuf[i] = db[i];
    }

    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            dwbuf[i][j] = dw[i*xdim+j];
        }
    }

    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            wbuf[i][j] = w[i*xdim+j];
        }
    }

    for(int i=0;i<ydim;i++){
        dybuf[i] = dy[i];
    }
    
    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            dxbuf[j] = dybuf[i] * wbuf[i][j];
            dwbuf[i][j] += dybuf[i]*xbuf[j];
        }
        
    }

    //compute gradient of biases
    for (int i=0;i<ydim;i++){
        dbbuf[i] += dybuf[i];
    }

    for(int i=0;i<xdim;i++){
         dx[i]=dxbuf[i];
    }

    for(int i=0;i<ydim;i++){
         db[i]=dbbuf[i];
    }

    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            dw[i*xdim+j]=dwbuf[i][j] ;
        }
    }

    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            w[i*xdim+j] =  wbuf[i][j] ;
        }
    }
    

}


void backward_conv(vector<float> &x, vector<float> &w, vector<float> &y, vector<float> &dx,vector<float> &dw,vector<float> &db, vector<float> &dy , int F, int C, int H, int W, int FH, int FW){

    //populate cache
    float xbuf[C][H][W];
    float wbuf[F][C][FH][FW];

    for(int i=0;i<C;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                xbuf[i][j][k] = x[i*H*W+j*W+k];
            }
        }
        
    }


    for(int i=0;i<F;i++){
        for(int j=0;j<C;j++){
            for(int k=0;k<FH;k++){
                for(int l=0;l<FW;l++){
                    wbuf[i][j][k][l] = w[i*C*FH*FW+j*FH*FW+k*FW+l];
                }
            }
        }
    }


    //dimensions of output
    int outH=H-FH+1;
    int outW=W-FW+1;

    //incoming gradient
//    float ybuf[F][outH][outW];
    float dybuf[F][outH][outW];

    for(int i=0;i<F;i++){
        for(int j=0;j<outH;j++){
            for(int k=0;k<outW;k++){
                dybuf[i][j][k] = dy[i*outH*outW+j*outW+k];
            }
        }
    }


    //gradients to be calculated

    float dxbuf[C][H][W];
    float dwbuf[F][C][FH][FW];
    float dbbuf[F];


    for(int i=0;i<F;i++){
        for(int j=0;j<C;j++){
            for(int k=0;k<FH;k++){
                for(int l=0;l<FW;l++){
                    dwbuf[i][j][k][l] = dw[i*C*FH*FW+j*FH*FW+k*FW+l];
                }
            }
        }
    }

    for(int i=0;i<C;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                dxbuf[i][j][k] = dx[i*H*W+j*W+k];
            }
        }
        
    }

    for(int i=0;i<F;i++){
        db[i] = dbbuf[i];
    }

    // compute gradients

    for(int f=0;f<F;f++){  
        for(int h=0;h<outH;h++){
            for(int w=0;w<outW;w++){      
                for(int c=0;c<C;c++){
                    for(int fh=0;fh<FH;fh++){
                        for(int fw=0;fw<FW;fw++){
                            dwbuf[f][c][fh][fw] += dybuf[f][h][w]*xbuf[c][h+fh][w+fw];
                            dxbuf[c][h+fh][w+fw] += dybuf[f][h][w]*wbuf[f][c][h+fh][w+fw];
                        }
                    }
                }
                dbbuf[f] += dybuf[f][h][w];    
            }
        }
    }

    //write back stage

    for(int i=0;i<F;i++){
        for(int j=0;j<C;j++){
            for(int k=0;k<FH;k++){
                for(int l=0;l<FW;l++){
                    dw[i*C*FH*FW+j*FH*FW+k*FW+l] = dwbuf[i][j][k][l];
                }
            }
        }
    }

    for(int i=0;i<C;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                dx[i*H*W+j*W+k] = dxbuf[i][j][k];
            }
        }
        
    }

    for(int i=0;i<F;i++){
        db[i] = dbbuf[i];
    }
    

}


void forward_relu(vector<float> &x, vector<float> &y, int dim){

    float xbuf[dim];
    float ybuf[dim];


    for(int i=0;i<dim;i++){
        xbuf[i] = x[i];
        if (xbuf[i] > 0){
            ybuf[i] = xbuf[i];
        }
        else{
            ybuf[i] = 0;
        }
    }

    for(int i=0;i<dim;i++){
        y[i] = ybuf[i];
    }
}

void backward_relu(vector<float> &x, vector<float> &dx, vector<float> &dy, int dim){

    float xbuf[dim];
    float dxbuf[dim];
    float dybuf[dim];

    for(int i=0;i<dim;i++){
        xbuf[i] = x[i];
        dybuf[i] = dy[i];

        if (xbuf[i] > 0){
            dxbuf[i] = dybuf[i];
        }
        else{
            dxbuf[i] = 0;
        }
    }

    for(int i=0;i<dim;i++){
        dx[i] = dxbuf[i];
    }

}

float cross_entropy_derivative(vector<float> x,vector<float> &dx, int y,long int N){
    
    float log_probs[x.size()];
    float probs[x.size()];

    float loss =0;


    float max = x[0];
    for(int i=1;i<x.size();i++){
        if(x[i] > max){
            max = x[i];
        }
    }

    for(int i=0;i<x.size();i++){
        log_probs[i] = x[i] - max;
    }

    float sum = 0;

    for(int i=0;i<x.size();i++){
        sum += exp(log_probs[i]);
    }

    for(int i=0;i<x.size();i++){
        probs[i] = exp(log_probs[i])/sum;
    }

    loss -= log(probs[y]);

    

    for(int i=0;i<x.size();i++){
        if(i == y){
            dx[i] = (probs[i] - 1)/N;
        }
        else{
            dx[i] = probs[i]/N;
        }
    }
    

    

    return loss;
}  
    




