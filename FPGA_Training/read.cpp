#include "read.h"

using namespace std;

size_t split_float(const std::string &txt, std::vector<float> &strs, char ch)
{
    size_t pos = txt.find( ch );
    size_t initialPos = 0;
    

    // Decompose statement
    while( pos != std::string::npos ) {
        const char* s=txt.substr( initialPos, pos - initialPos ).c_str();
//        cout<<s<<'\n';
        float x;
        sscanf(s,"%f",&x); 
        strs.push_back(x);
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
//    const char* s=txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ).c_str();
//    float x;
//    sscanf(s,"%f",&x);
//    strs.push_back(x);

    return strs.size();
}

size_t split_int(const std::string &txt, std::vector<int> &strs, char ch)
{
    size_t pos = txt.find( ch );
    size_t initialPos = 0;

    // Decompose statement
    while( pos != std::string::npos ) {
        const char* s=txt.substr( initialPos, pos - initialPos ).c_str();
        int x;
        sscanf(s,"%d",&x); 
        strs.push_back(x);
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
//    const char* s=txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ).c_str();
//    int x;
//    sscanf(s,"%d",&x);
//    strs.push_back(x);
//
    return strs.size();
}

void read_input(vector< vector<float> > &x_train, vector<int> &y_train){
    fstream xfile;
    xfile.open("x_train.txt",ios::in); //open a file to perform read operation using file object
   if (xfile.is_open()){ //checking whether the file is open
//       cout << "file is opened"<<'\n';
      string tp;
      int iter=0;
      while(getline(xfile, tp)){ //read data from file object and put it into string.
//          cout<<tp<<'\n';
          x_train.push_back(vector<float>(0));
//          if (iter ==1)
//          cout<<tp<<'\n';
         split_float(tp, x_train[iter],' ');
//          if (iter == 1){
//              for(int k=0;k<x_train[iter].size();k++){
//                  cout << x_train[iter][k]<<' ';
//              }
//          }
//          cout << x_train[iter].size()<<'\n';
          iter+=1;
      }
       cout << x_train[0].size();
      cout<<'\n';
      xfile.close(); //close the file object.
   }
  


   fstream yfile;
    yfile.open("y_train.txt",ios::in); //open a file to perform read operation using file object
   if (yfile.is_open()){ //checking whether the file is open
//       cout << "inside file"<<'\n';
      string tp;
      while(getline(yfile, tp)){ //read data from file object and put it into string.
         const char* s = tp.c_str();
         int x;
        sscanf(s,"%d",&x); 
        y_train.push_back(x);
         
         }
 
      cout<<'\n';
      yfile.close(); //close the file object.
   }
}
