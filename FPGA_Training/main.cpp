#include <vector>
#include "read.h"
#include "nn.cpp"
#include "layers.h"
using namespace std;

//int main(){
//
//    vector<vector <float> > x_train;
//
//    vector<int> y_train;
//
//    read_input(x_train, y_train);
//
//    cout<<x_train.size()<<'\n';
//
//    Neural_Network nn;
//
//    nn.add_conv(3,1,1,784,1,3);
//    nn.add_relu(2346);
//    nn.add_fcc(2346,10);
//
////    for (int i=0;i<784;i++){
////        cout<< x_train[1][i];
////    }
//    cout<<'\n';
//
//    nn.train(x_train,y_train,0.1);
//
//
//}





int main(){
    const int F=1;
    const int C=1;
    const int H=3;
    const int W=3;
    const int FH=2;
    const int FW=2;

    const int outH=H-FH+1;
    const int outW=W-FW+1;

    vector<float> x={1.0,0.0,0.0,  1.0,2.0,1.0,  1.0,3.0,1.0};
    vector<float> dx(C*H*W);
    vector<float> w={1.0,1.0,1.0,1.0};
    vector<float> dw(F*C*FH*FW);
    vector<float> dy={0.1,0.2,0.3,0.4};
    vector<float> b={0.5};
    vector<float> db(F);

    for(int i=0;i<F;i++){
            for(int j=0;j<C;j++){
                for(int k=0;k<FH;k++){
                    for(int l=0;l<FW;l++){
                       dw[i*C*FH*FW+j*FH*FW+k*FW+l]=0.0;
                    }
                }
            }
        }

    for(int i=0;i<C;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                dx[i*H*W+j*W+k] = 0.0;
            }
         }
    }

    backward_conv(x,w, dx,dw,db, dy , F, C, H, W, FH,FW);

    for(int i=0;i<C;i++){
            for(int j=0;j<H;j++){
                for(int k=0;k<W;k++){
                    cout << dx[i*H*W+j*W+k]<< " ";
                }
                cout << '\n';
            }
            cout <<'\n';

        }

    for(int i=0;i<F;i++){
                for(int j=0;j<C;j++){
                    for(int k=0;k<FH;k++){
                        for(int l=0;l<FW;l++){
                           cout << dw[i*C*FH*FW+j*FH*FW+k*FW+l] << " ";
                        }
                        cout << '\n';
                    }
                    cout << '\n';
                }
                cout << '\n';
            }




}
