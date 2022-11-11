/* Testing use of the standard library's threading functionality
-- NOTE:: more than likely will not use to implement the threaded version
          of extractWRFData.cpp */
#include <stdlib.h>
#include <thread>
#include <iostream>
using namespace std;

void accumulate(int &sum, int start, int end){
    sum = 0;
    for (int i=start; i<end; i++){
        sum+=i;
    }
}
int main(){
    const int numThreads = 2;
    const int numElements = 1000;
    const int step = 50;
    int sum1 = 0, sum2 = 0;

    thread t1(accumulate, ref(sum1), 0, step);
    thread t2(accumulate, ref(sum2), 10, step);

    t1.join();
    t2.join();

    cout << "sum1: " << sum1 << endl;
    cout << "sum2: " << sum2 << endl;

    return 0;
}