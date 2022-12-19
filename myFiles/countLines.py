# this file will count the number of lines in 
# the three separate implementations of extractWRFData

# python 
with open("/home/kalebg/Documents/GitHub/customExtraction/myFiles/extractWRFDataCopy.py", 'r') as fp:
    num_lines = sum(1 for line in fp if line.rstrip())
    print("Lines in Python File: ", num_lines)
    fp.close()

# Cpp
with open("/home/kalebg/Documents/GitHub/customExtraction/myFiles/extractWRFData3.cpp", 'r') as fp:
    num_lines_cpp = sum(1 for line in fp if line.rstrip())
    print("Lines in Cpp file: ", num_lines_cpp)
    fp.close()

# Threaded Cpp
with open("/home/kalebg/Documents/GitHub/customExtraction/myFiles/threadedExtractWRFData1.cpp") as fp:
    num_lines_threaded = sum(1 for line in fp if line.rstrip())
    print("Lines in threaded Cpp file: ", num_lines_threaded)
    fp.close()