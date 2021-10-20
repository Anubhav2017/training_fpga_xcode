//
//  main.cpp
//  FPGA_Training
//
//  Created by Anubhav  Agarwal on 20/10/21.
//

#include <iostream>
#include "functions.h"

using namespace std;
int main(){
    float x[2] = {1.0, 2.0};

    float w[2] = {0.5, 0.1};
    float b[1]= {0.01};
    float z[1]={0};

    float dx[2] = {1.0, 2.0};
    float dw[2] = {0.5, 0.1};
    float db[1] = {0.01};
    float dz[1] = {0.2};

    forward_fcc(x,w,z,b,2,1);
    backward_fcc(x,w,z,b,dx,dz,db,dw,2,1);

    cout << "dz=" << dz[0] <<"\n";
    cout << "db=" << db[0] <<"\n";
    cout << "dx = " << dx[0] <<" "<< dx[1] << "\n";
    cout << "dw = " << dw[0] << " " <<dw[1] << "\n";

    return 0;
}
