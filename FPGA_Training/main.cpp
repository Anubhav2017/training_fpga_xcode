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

using namespace std;

void sine_model(){
    
    int nepochs=5;
    
    float lr = 0.001;
    float w1[8]={};
    float w2[8]={};
    float dw1[8]={};
    float dw2[8]={};
    
    for(int i=0;i<8;i++){
        w1[i]= 0.01;
        w2[i]=0.012;
    }
    
    float b1[1]={0.01};
    float b2[1]={0.01};
    float db1[1]={};
    float db2[1]={};
    
    float h1[8]={};
    float dh1[8]={};
    
    cout<<"Weights and biases initialized"<<'\n';
    
    float y[1]={0};
    float dy[1]={0};
    
    float xtrain[1000] = {};
    
    float x[1]={};
    float dx[1]={};
    
    float ytrue[1000] ={};
    float ypred[1000]={};
    
    for(int i=0;i<1000;i++){
        xtrain[i] = 0.01*i;
    }
    
    
    auto rng = std::default_random_engine{};
    
    shuffle(begin(x), end(x),rng);
    
    for(int i=0;i<1000;i++){
        ytrue[i] = sin(x[i]);
    }
    
    cout<<"Dataset created"<<'\n';
    
    for(int i=0;i<10;i++){
        
        cout<<"Epoch "<<i<<" started"<<'\n';
        
        //calculating predicted values
        for(int j=0;j<1000;j++){
            x[0]=xtrain[j];
            forward_fcc(x, w1, h1, b1, 1, 8);
            forward_fcc(h1, w2, y, b2, 8, 1);
            ypred[j]=y[0];
            
        }
        
        //calucalating mean squared error over the batch
        cout<<mse_loss(ypred, ytrue, 1000)<<'\n';
        
        //iterating over batch
        for (int j=0;j<1000;j++) {
            
            //current sample
            x[0]=xtrain[j];
            
            //forward prop
            forward_fcc(x, w1, h1, b1, 1, 8);
            forward_fcc(h1, w2, y, b2, 8, 1);
            
            
            //error calculation
            dy[0] = y[0]-ytrue[j];
            
            //bacpropagation
            backward_fcc(h1, w2, y, b2, dh1, dy, db2, dw2, 8, 1);
            backward_fcc(x, w1, h1, b1, dx, dh1, db1, dw1, 1, 8);
            
            //updating weights and biases
            for(int k=0;k<8;k++){
                w1[k] -= lr*dw1[k];
                w2[k] -= lr*dw2[k];
                h1[k] -= lr*dh1[k];
            }
            b1[0] -=lr*db1[0];
            b2[0] -=lr*db2[0];
            
        }
        
        
    }

}

int main(){
    
    sine_model();
    

    return 0;
}
