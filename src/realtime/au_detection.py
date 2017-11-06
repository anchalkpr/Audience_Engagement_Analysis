import csv
import os
import comman_utils as comman_utils
main_content = dict()
rowList = []
au_list = []
facs_calc = dict()
au_avg = []
path = comman_utils.PATH_ANALYSIS_DIR
for _, dirnames, filenames in os.walk(path):
    for files in filenames:
        f=open(os.path.join(path,files),'r')
        rows = csv.reader(f, delimiter=',', quotechar='|')
        rowList = list(rows)
        rowList.pop(0)
        for row in rowList:
            key = files[:-4]+'_'+row[0]
            au_list = []
            au_list = row[10:27]
            main_content[key] = au_list
list_length = len(main_content)
for key, value in main_content.items():
    for i in range(0, list_length):
        au_avg[i]+=value[i]
for i in range(0, list_length):
    au_avg[i] = au_avg / list_length
