#include<iostream>
#include<math.h>
using namespace std;
void forward_fcc(float* x, float** w, float* y, float* b, int xdim, int ydim){

    for(int i=0; i< ydim;i++){
        y[i]= b[i];

        for (int j=0; j<xdim;j++){
            y[i]+= w[i][j]*x[j];
        }
    }

}

void backward_fcc(float* x, float** w, float* b, float* dx, float* dy, int xdim, int ydim,float lr){
    //compute gradient of activations
    
    float db[ydim];
    float dw[ydim][xdim];
    
    for(int i=0;i<xdim;i++){
        dx[i]=0;
    }
    
    for(int j=0;j<xdim;j++){
        for(int i=0;i<ydim;i++){
            dx[j] += dy[i] * w[i][j];
        }
        
    }
    //compute gradient of weights
    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            dw[i][j] = dy[i]*x[j];
        }
    }

    //compute gradient of biases
    for (int i=0;i<ydim;i++){
        db[i] = dy[i];
    }
    
    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            
            w[i][j]-=lr*dw[i][j];
            
        }
        
        b[i] -= lr*db[i];
    }
}

void forward_relu(float* x, float* y, int dim){
    for(int i=0;i<dim;i++){
        if(x[i]>0)
            y[i]=x[i];
        else{
            y[i]=0;
        }
//        y[i]=x[i];
    }
}

void backward_relu(float* x, float* dx, float* dy, int dim){
    for(int i=0;i<dim;i++){
        if (x[i] >= 0){
            dx[i] = dy[i];
        }
        else{
            dx[i]=0;
        }
//        dx[i]=dy[i];
    }
}


void forward_softmax(float* z, float* a, int size_t){

    float* expz = new float[size_t];

    for(int i=0; i< size_t;i++){
        expz[i] = exp(z[i]);
    }
    float expsum = 0;
    for (int i=0; i<size_t;i++){
        expsum += expz[i];
    }
    for(int i=0;i<size_t;i++){
        z[i]= expz[i]/expsum;
    }
    delete[] expz;

}

void backward_softmax(float* dz, float* da, float* a,int size_t){
    for (int i=0;i<size_t;i++){
        dz[i] = da[i]*a[i]*(1-a[i]);
    }
}

float mse_loss(float* pred, float* truth, int n){
    float loss=0;
    for(int i=0;i<n;i++){
        loss+= pow(pred[i]-truth[i],2);
    }
    loss = loss/n;
    return loss;
}

void mse_gradient(float* pred, float* truth, float* grad,int dim){

    for(int i=0;i<dim;i++){
        grad[i]=(pred[i]-truth[i])/dim;
    }
    
}

float cross_entropy_loss(float* truth, float* est, int dim){
    
    float loss=0.0;
    for(int i=0;i<dim;i++){
        loss-=truth[i]*log(est[i]);
    }

    return loss;
}

void cross_entropy_derivative(float* q,int label,float* dz, int dim){
    for(int i=0;i<dim;i++){
        dz[i] = q[i];
    }
    dz[label] -= 1; 
}

//int main(){
//    float x[2] = {1.0, 2.0};
//
//    float w[2] = {0.5, 0.1};
//    float b[1]= {0.01};
//    float z[1]={0};
//
//    float dx[2] = {1.0, 2.0};
//    float dw[2] = {0.5, 0.1};
//    float db[1] = {0.01};
//    float dz[1] = {0.2};
//
//    forward_fcc(x,w,z,b,2,1);
//    backward_fcc(x,w,z,b,dx,dz,db,dw,2,1);
//
//    cout << "dz=" << dz[0] <<"\n";
//    cout << "db=" << db[0] <<"\n";
//    cout << "dx = " << dx[0] <<" "<< dx[1] << "\n";
//    cout << "dw = " << dw[0] << " " <<dw[1] << "\n";
//
//    return 0;
//}
