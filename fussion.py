import csv
import pandas as pd

f = open('output/submit.csv')

reader = csv.reader(f)
list1 = []
for row in reader:
    row = row[0]
    name, pro1, pro2= row.split(' ')
    set1 = {"img_name": name,  "prob": str((float(pro1)+float(pro2))/2)}
    list1.append(set1)
df = pd.DataFrame(list1)
df.to_csv("submission.csv", header=False, index=False, sep=' ')
