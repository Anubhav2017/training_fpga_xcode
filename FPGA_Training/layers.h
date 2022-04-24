//
//  layers.h
//  FPGA_Training
//
//  Created by Anubhav  Agarwal on 22/04/22.
//

#ifndef layers_h
#define layers_h
#include<vector>

using namespace std;
void forward_fcc(vector<float> &x, vector<float> &w, vector<float> &y, vector<float> &b, int xdim, int ydim);

void forward_conv(vector<float> &x, vector<float> &w, vector<float> &y, vector<float> &b, int F, int C, int H, int W, int FH, int FW);


void backward_fcc(vector<float> &x, vector<float> &w, vector<float> &dx, vector<float> &dy,vector<float> &dw, vector<float> &db, int xdim,int ydim);


void backward_conv(vector<float> &x, vector<float> &w, vector<float> &y, vector<float> &dx,vector<float> &dw,vector<float> &db, vector<float> &dy , int F, int C, int H, int W, int FH, int FW);

void forward_relu(vector<float> &x, vector<float> &y, int dim);

void backward_relu(vector<float> &x, vector<float> &dx, vector<float> &dy, int dim);

float cross_entropy_derivative(vector<float> x,vector<float> &dx, int y,long int N);
#endif /* layers_h */
