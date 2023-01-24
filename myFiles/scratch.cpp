#include "stdio.h"
#include "iostream"
using namespace std;
int main(){
    bool bln[200];
    fill_n(bln, 200, true);

    for (auto sum : bln){
        cout << sum << endl;
    }
}