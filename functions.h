#include<iostream>
#include<math.h>

void forward_fcc(float* x, float* w, float* y, float* b, int xdim, int ydim);

void backward_fcc(float* x, float* w, float* y, float* b, float* dx, float* dy, float* db, float* dw, int xdim, int ydim);

void forward_softmax(float* z, float* a, int size_t);

void backward_softmax(float* dz, float* da, float* a,int size_t);

float mse_loss(float* pred, float* truth, int dim);

void mse_gradient(float* pred, float* truth, float* grad,int dim);

float cross_entropy_loss(float* truth, float* est, int dim);

void cross_entropy_derivative(float* q,int label,float* dz, int dim);