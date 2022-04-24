#include <vector>
#include "read.h"
#include "nn.cpp"

using namespace std;

int main(){

    vector<vector <float> > x_train;

    vector<int> y_train;

    read_input(x_train, y_train);
    
    cout<<x_train.size()<<'\n';
    
    Neural_Network nn;
    
    nn.add_conv(3,1,1,784,1,3);
    nn.add_relu(2346);
    nn.add_fcc(2346,10);
    
//    for (int i=0;i<784;i++){
//        cout<< x_train[1][i];
//    }
    cout<<'\n';
    
    nn.train(x_train,y_train,0.1);


}
