#include "stdio.h"
#include "iostream"
#include <filesystem>
// namespace fs = std::filesystem;
#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector> 
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <time.h>
#include <cassert>
#include <cmath>
#include "semaphore.h"
using namespace std;
bool dirExists(string filePath){
 	struct stat info;

	if(stat(filePath.c_str(), &info) != 0){
		printf("Error checking directory: %s", filePath.c_str());
        exit(0);
	}
	else if(info.st_mode & S_IFDIR){
		return true;
	}
	else return false;
}
int len(string str){
    int length = 0;
    for(int i=0; str[i]!= '\0';i++){
        length++;
    }
    return length;
}
vector<string> splitonDelim(string str, char sep){
    vector<string> returnedStr;
    int curridx = 0, i=0;
    int startidx=0, endidx = 0;
    while(i<=len(str)){
        if(str[i]==sep || i==len(str)){
            endidx=i;
            string subStr = "";
            subStr.append(str, startidx, endidx-startidx);
            returnedStr.push_back(subStr);
            curridx++;
            startidx = endidx+1;
        }
        i++;
    }
    return returnedStr;
}

int main(){
    //does not exist, need to create the path
    string writePath="/home/kaleb/Documents/test/";
    vector<string> splittedPath = splitonDelim(writePath, '/');
    for(int i=0;i<splittedPath.size();i++){
        printf("%s \n",splittedPath[i].c_str());
    }

    return 0;
}