//
//  main.cpp
//  FPGA_Training
//
//  Created by Anubhav  Agarwal on 20/10/21.
//

#include <iostream>
#include <vector>
#include <iostream>
#include "functions.h"
#include "math.h"
#include <algorithm>
#include <random>
//#include <opencv2/core.hpp>
using namespace std;

void sine_model(){
        
    float lr = 0.02;
    float** w1 =new float*[16];
    
    for(int i=0;i<16;i++){
        w1[i]= new float[1];
    }
    
    
    float** w2 = new float*[16];
    
    for(int i=0;i<16;i++){
        w2[i]= new float[16];
    }
    
    float** w3= new float*[1];
    
    for(int i=0;i<1;i++){
        w3[i]= new float[16];
    }
    
    float b1[16]={};
    float b2[16]={};
    float b3[1]={};
    

    for(int i=0;i<16;i++){
        w1[i][0]= 0.01;
        for(int j=0;j<16;j++){
            w2[i][j]=0.012;
        }
        
        w3[0][i]=0.01;
    }
    
    
    
    float h1[16]={};
    float dh1[16]={};
    
    float h2[16]={};
    float dh2[16]={};
    
    float h3[16]={};
    float dh3[16]={};
    
    float h4[16]={};
    float dh4[16]={};
    
    cout<<"Weights and biases initialized"<<'\n';
    
    float y[1]={0};
    float dy[1]={0};
    
    float xtrain[1000] = {};
    
    float x[1]={};
    float dx[1]={};
    
    float ytrue[1000] ={};
    float ypred[1000]={};
    
    for(int i=0;i<1000;i++){
        xtrain[i] = 0.001*i;
    }
    
    
    auto rng = std::default_random_engine{};
    
    shuffle(begin(x), end(x),rng);
    
    for(int i=0;i<1000;i++){
        ytrue[i] = sin(xtrain[i]);
//        cout<<ytrue[i]<<'\n';
    }
    
    int nbatch=400;
    
    
    
    cout<<"Dataset created"<<'\n';
    
    for(int i=0;i<300;i++){
        
        cout<<"Epoch "<<i<<" started"<<'\n';
        
        //calculating predicted values
        for(int j=0;j<nbatch;j++){
            x[0]=xtrain[j];
            forward_fcc(x, w1, h1, b1, 1, 16);
            forward_relu(h1,h2,16);
            forward_fcc(h2, w2, h3, b2, 16, 16);
            forward_relu(h3,h4,16);
            forward_fcc(h4, w3, y, b3, 16, 1);
            
            ypred[j]=y[0];
            
        }
        
        //calucalating mean squared error over the batch
        cout<<"mse loss= " << mse_loss(ypred, ytrue, nbatch)<<'\n';
        
        //iterating over batch
        for (int j=0;j<nbatch;j++) {
            
            //current sample
            x[0]=xtrain[j];
            
            //forward prop
            forward_fcc(x, w1, h1, b1, 1, 16);
            forward_relu(h1,h2,16);
            forward_fcc(h2, w2, h3, b2, 16, 16);
            forward_relu(h3,h4,16);
            forward_fcc(h4, w3, y, b3, 16, 1);
            
            //error calculation
            dy[0] = y[0]-ytrue[j];
            
            //bacpropagation
            backward_fcc(h4, w3 , b3, dh4, dy, 16, 1,lr);
            backward_relu(h3, dh3, dh4, 16);
            backward_fcc(h2, w2, b2, dh2, dh3, 16, 16, lr);
            backward_relu(h1, dh1, dh2, 16);
            backward_fcc(x, w1, b1, dx, dh1, 1, 16, lr);
            
            //updating weights and biases
            
            
        }
        
        
    }

}

int main(){
    
    sine_model();
    

    return 0;
}
