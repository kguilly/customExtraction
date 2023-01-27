import csv
output = ""
with open("/home/kaleb/Documents/GitHub/customExtraction/myFiles/countyInfo/us-state-ansi-fips.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # output+=(cell.strip() for cell in row)
        # output+=("\n")
        for cell in row:
            output+=str(cell).strip()
            output+=","
        output+="\n"

with open("/home/kaleb/Documents/GitHub/customExtraction/myFiles/countyInfo/us-state-ansi-fips2.csv", 'w') as f:
    f.write(output)


