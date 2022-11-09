#include <iostream>
#include <cstring>
#include <vector>
#include <ostream>
#include <sys/types.h>
#include <sys/stat.h>
using namespace std;


bool checkDateRange(vector<int> beginDate, vector<int> endDate);
vector<int> getNextDay(vector<int>);
vector<string> formatDay(vector<int>);
bool dirExists(string filePath);

int main(int argc, char**argv){
	/*
	cout << "Hello world" << endl;
	int j,k=0;
	for (int i=0; i<5; ++i){
		cout << i << " j:" << j++ << " ++k:" << ++k << endl;
	}

	// sci notation? 
	const double sci = 3.65432e4;
	cout << sci << endl;


	char * date = "20190101";
	//char  year[4];
	// for(int i =0; i<4; i++) year[i] = date[i];
	
	// cout << "YEARRRRR::: " << year << "DATE::" << date << endl;

	//cout << "\n\n\nARRRRRGS::::\nargc: " << argc << "argv: " << argv[1]<< endl;
	if(argv[1]!=NULL){
		cout << "you did it" << endl;
	}else{
		cout << "you didn't do it" << endl;
	}
	// strcat('i','a','b','c') // only takes 2 args
	int datesize=0;
	char date1 = date[0];
	while (date1 != '\0'){
		datesize++;
		date1 = date[datesize];
		cout << date1 << endl;
	}
	cout << datesize << endl;
	// vector<char> chararr = "what it do";


	cout << "----------------------------------\nTesting time.h\n----------------------------"<<endl;
	#include <time.h>
	// char beginDate[9] = "20190101";
	// char * endDate = "20190201";
	// // format %Y%m%d
	// char day[2] = {beginDate[4], beginDate[5]};

	// cout << day;
	tm * bd; 
	//bd->tm_mday = day;

	// Testing using an array of strings instead
	#include <string.h>	
	// FORMAT::			yyyy, mm, dd
	int begin_day[3] = {2019, 01, 02};
	int end_day[3] = {2019, 01, 03}; 

	// THIS IS NOT HOW TO USE
	// bd->tm_year = begin_day[0];
	// bd->tm_mon = begin_day[1] -1;
	// bd->tm_mday = begin_day[2];

	// cout << "year: " << bd->tm_year << endl;
	// cout << "month: " << bd->tm_mon << endl;
	// cout << "day: " << bd->tm_mday << endl;
	time_t rawtime;
	struct tm *timeinfo;
	int year = begin_day[0];
	int month = begin_day[1];
	int day = begin_day[2];
	char str[256];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	timeinfo->tm_year = (char)(year -1900);
	timeinfo->tm_mon = month -1;
	timeinfo->tm_mday = (char)day-1;
	mktime(timeinfo);

	strftime(str, sizeof(str), "%D", timeinfo);
	cout << str << endl;
	*/
	

	// FORMAT::			yyyy, mm, dd
	// int beginDay[3] = {2019, 01, 02};
	// int endDay[3] = {2019, 01, 03}; 
	// bool status = checkDateRange(beginDay, endDay);
	// cout << "Status: " << status << "\n\n" << endl;

	// cout << sizeof(beginDay) / sizeof(int) << endl;
	// cout << sizeof(endDay) / sizeof(int) << endl;

	// // trying again with array class
	#include <vector>
	vector<int> beginDate = {2019, 1, 2};
	vector<int> endDate = {2019, 01, 03};
	bool status = checkDateRange(beginDate, endDate);
	cout << status << endl;

	vector<int> nextDay = getNextDay(beginDate);
	vector<int> dayafterthat = getNextDay(nextDay);
	for(int i: nextDay){
		cout << i << " ";
	}
	cout << "\n";
	for(int i : dayafterthat){
		cout << i << " ";
	}
	
	// concatenate the file path
	string filePath = "../UtilityTools/extractTools/data/";  // path to "data" folder. File expects structure to be: 

	vector<string> formattedDate = formatDay(beginDate);

	cout << "\n\n";
	for (string i : formattedDate){
		cout << i << " ";
	}
	cout << "\n\n";

	string filePath1 = filePath + formattedDate.at(0) + "/" + formattedDate.at(3) + "/";
	cout << filePath1 << endl;
	// bool exists = filePath1.exists();
	bool exists = dirExists(filePath1);
	if (exists == true) cout << "the directory does exist" << endl;
	
	beginDate.clear();
	endDate.clear();
	nextDay.clear();
	dayafterthat.clear();

	
	/////////////////////////////////////////////////////////////
	// testing try catch blocks
	cout << "\n\n\n\n";
	string fullFilePath = "/home/kalebg/Desktop/School/Y4S1/REU/customExtraction/UtilityTools/extractTools/data/2019/20190101/hrrr.20190101.00.00.grib2";
	FILE* f;
	for(int i=0; i<3; i++){
		try{
		f = fopen(fullFilePath.c_str(), "rb");
		if(!f) throw(fullFilePath);
		}catch(string file){
			cout << "Exception thrown" << endl;
			continue;
		}
		cout<< "exception not thrown" << endl;
		fclose(f);

	}
	
	bool tru = true;
	cout << tru << endl;

	return 0;
}
// Getting a date range:: based off of : https://www.studymite.com/cpp/examples/program-to-print-the-next-days-date-month-year-cpp/
bool checkDateRange(vector<int> beginDate, vector<int> endDate){
	// check that the array is of the size expected // WORKS
	if((beginDate.size() != 3) || (endDate.size() != 3)) return false;

	// check that endDay is further than endDay
	if(beginDate.at(0)>endDate.at(0)) return false;
	else if(beginDate.at(0) == endDate.at(0)){
		if(beginDate.at(1) > endDate.at(1)) return false;
		else if(beginDate.at(1) == endDate.at(1)){
			if(beginDate.at(2) >= endDate.at(2)) return false;
		}
		
	}

	// check that they have actually passed a valid date

	return true;
}

vector<int> getNextDay(vector<int> beginDate){
	
	int day, month, year;
	day = beginDate.at(2);
	month = beginDate.at(1);
	year = beginDate.at(0);

	if (day > 0 && day < 28) day+=1; //checking for day from 0-27, can just increment
	else if(day == 28){
		if(month == 2){ // if it is february, need special care
			if ((year % 400 == 0) || (year % 100 != 0 || year % 4 == 0)){ // is a leap year
				day == 29;
			}
			else{
				day = 1;
				month = 3;
			}
		}
		else // its not feb
		{
			day += 1; 
		}
	}
	else if(day == 29) // last day check for feb on a leap year
	{
		if(month == 2){
			day = 1;
			month = 3;
		}
	}
	else if(day == 30) // last day check for april, june, sept, nov
	{
		if(month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12)
		{
			day += 1;
		}
		else
		{
			day = 1;
			month +=1;
		}
	}
	else if(day == 31) // last day of the month
	{
		day = 1;
		if(month == 12) // last day of the year
		{
			year += 1;
			month = 1;
		}
		else month+=1;
	}
	vector<int> nextDay = {year, month, day};
	return nextDay;
}

vector<string> formatDay(vector<int> date){
	string strYear, strMonth, strDay;
	int intYear = date.at(0);
	int intMonth = date.at(1);
	int intDay = date.at(2);

	if(intMonth < 10){
		strMonth = "0" + to_string(intMonth);
	}
	else strMonth = to_string(intMonth);
	if(intDay < 10) strDay = "0" + to_string(intDay);
	else strDay = to_string(intDay);
	strYear = to_string(intYear);

	vector<string> formattedDate = {strYear , strMonth, strDay , (strYear+strMonth+strDay)};
	return formattedDate;
}

bool dirExists(string filePath){
	struct stat info;

	if(stat(filePath.c_str(), &info) != 0){
		return false;
	}
	else if(info.st_mode & S_IFDIR){
		return true;
	}
	else return false;
}
