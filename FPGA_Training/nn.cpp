#include "layers.h"
#include <vector>
#include <iostream>
#include <string>

using namespace std;

class Neural_Network {
    
public:

    vector<string> layers;
    int nlayers=0;

    vector<vector<float>> weights;
    vector<vector<float>> grads_weights;

    vector<vector<float>> biases;
    vector<vector<float>> grads_bias;

    vector<vector<float>> activations;
    vector<vector<float>> grads_activations;

    vector<vector<int>> shapes;
    

    void add_fcc(int xdim, int ydim) {

        layers.push_back("fcc");
        nlayers++;

        vector<float> weight_vec(xdim*ydim);
        fill(weight_vec.begin(),weight_vec.end(),0.01);
        
        weights.push_back(weight_vec);
        grads_weights.push_back(vector<float>(ydim * xdim));

        biases.push_back(vector<float>(ydim));
        grads_bias.push_back(vector<float>(ydim));

        activations.push_back(vector<float>(ydim));
        grads_activations.push_back(vector<float>(ydim));

        shapes.push_back(vector<int>(2));
        shapes[nlayers - 1][0] = xdim;
        shapes[nlayers - 1][1] = ydim;
    }

    void add_conv(int F, int C, int H, int W, int FH, int FW) {
        layers.push_back("conv");
        nlayers++;
        
        vector<float> weight_vec(F * C * FH * FW);
        fill(weight_vec.begin(),weight_vec.end(),0.01);
        
        weights.push_back(weight_vec);
        grads_weights.push_back(vector<float>(F * C * FH * FW));

        biases.push_back(vector<float>(F));
        grads_bias.push_back(vector<float>(F));

        activations.push_back(vector<float>(F * (F-FH+1) * (W-FW+1)));
        grads_activations.push_back(vector<float>(F * (F-FH+1) * (W-FW+1)));

        shapes.push_back(vector<int>(6));
        shapes[nlayers - 1][0] = F;
        shapes[nlayers - 1][1] = C;
        shapes[nlayers - 1][2] = H;
        shapes[nlayers - 1][3] = W;
        shapes[nlayers - 1][4] = FH;
        shapes[nlayers - 1][5] = FW;

    }

    void add_relu(int dim) {
        layers.push_back("relu");
        nlayers++;

        weights.push_back(vector<float>(0));
        grads_weights.push_back(vector<float>(0));

        biases.push_back(vector<float>(0));
        grads_bias.push_back(vector<float>(0));

        activations.push_back(vector<float>(dim));
        grads_activations.push_back(vector<float>(dim));

        shapes.push_back(vector<int>(1));
        shapes[nlayers - 1][0] = dim;
    }

    void fwprop(vector<float> &x){

        if (layers[0] == "fcc") {
            forward_fcc(x, weights[0], activations[0], biases[0], shapes[0][0], shapes[0][1]);
        }
        else if (layers[0] == "conv") {
            
            forward_conv(x, weights[0], activations[0], biases[0], shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3], shapes[0][4], shapes[0][5]);
        }
        
//        for(int i=0;i<weights[0].size();i++){
//            cout << weights[0][i]<<' ';
//        }
//        cout <<'\n';
      



        for(int i=1;i<nlayers;i++){
            if(layers[i]=="fcc"){
                forward_fcc(activations[i-1], weights[i], activations[i], biases[i], shapes[i][0], shapes[i][1]);
            }
            else if(layers[i]=="conv"){
                forward_conv(activations[i-1], weights[i], activations[i], biases[i], shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][4], shapes[i][5]);
            }
            else if(layers[i]=="relu"){
                forward_relu(activations[i-1], activations[i], shapes[i][0]);
            }
        }
    }

    void backprop(){
            
        for(int i=nlayers-1;i>0;i--){
            if(layers[i]=="fcc"){
                backward_fcc(activations[i-1], weights[i], grads_activations[i-1], grads_activations[i], grads_weights[i],grads_bias[i], shapes[i][0], shapes[i][1]);
            }
            else if(layers[i]=="conv"){
                backward_conv(activations[i-1], weights[i],activations[i], grads_activations[i-1], grads_weights[i], grads_bias[i],grads_activations[i],  shapes[i][0],shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][4], shapes[i][5]);
            }
            else if(layers[i]=="relu"){
                backward_relu(activations[i-1], grads_activations[i-1], grads_activations[i], shapes[i][0]);
            }
        }
    }
    
    void update_weights(float lr){
        for(int i=0;i< nlayers;i++){
            
            for(int j=0;j<weights[i].size();j++){
                
                weights[i][j] -= grads_weights[i][j]*lr;
            }
            
            for (int j=0;j<biases[i].size();j++){
                biases[i][j] -= grads_bias[i][j]*lr;
            }
            
        }
    }

    void train(vector<vector<float> > &x, vector<int> &y, float lr){
        

        long int N=x.size();
                
        for(int epoch=0;epoch < 100;epoch++){
            float loss=0;

            for (int i = 0; i < x.size(); i++) {
                fwprop(x[i]);
                loss+=cross_entropy_derivative(activations[nlayers-1],grads_activations[nlayers-1],y[i],N);
                ///----------------------------------------------------------------------------------------------------------------
//                if( i==10){
//                    for(int l=0;l<nlayers;l++){
//                        cout<< "layer "<<l<<" activations= ";
//                        for(int k=0;k<activations[l].size();k++){
//                            cout<< activations[l][k]<<' ';
//                        }
//                            cout<< '\n';
//
//                    }
//
//                    cout<< '\n';
//
//                }
//                if( i==10){
//                    for(int l=0;l<nlayers;l++){
//                        cout<< "layer "<<l<<" weights= ";
//                        for(int k=0;k<weights[l].size();k++){
//                            cout<< weights[l][k]<<' ';
//                        }
//                            cout<< '\n';
//
//                    }
//                for(int k=0;k<10;k++){
//                    cout<<activations[nlayers-2][i]<<' ';
//                }
//                    cout<< '\n';
//
//                }
                ////-------------------------------------------------------------------------------------------------------------
                backprop();
    //            if((i % 100) ==0){
    //                update_weights(lr);
    //                cout << "current loss =" << loss/i<<'\n';
    //            }
    //            cout << "current loss = "<<curr_loss<<'\n';
            }
            loss=loss/N;
            update_weights(0.1);

            cout<<"epoch:"<<epoch<<" , "<< "loss: "<<loss<<'\n';
        }

    }




};
