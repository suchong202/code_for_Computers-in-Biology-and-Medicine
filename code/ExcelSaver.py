import xlwt
import numpy as np
import os
class ExcelSaver():
    def __init__(self,sheetname="sheet1"):
        self.sheetname = sheetname
        self.workbook = xlwt.Workbook(encoding='ascii')
        self.worksheet = self.workbook.add_sheet(sheetname,cell_overwrite_ok=True)
    def write_data(self,sheetvalues,start_line=1):
        for i,key in enumerate(sheetvalues.keys()):
            print("test1",i,key)
            self.worksheet.write(start_line,i+1,label=key)
            for j,val in enumerate(sheetvalues[key]):
                self.worksheet.write(start_line+1+j,i+1,label=val)
    def save(self,filename="temp"):
        self.workbook.save("excel/"+filename+".xls")
class ExcelSaver2():
    def __init__(self,sheetname="sheet1"):
        self.sheetname = sheetname
        self.workbook = xlwt.Workbook(encoding='ascii')
        self.worksheet = self.workbook.add_sheet(sheetname,cell_overwrite_ok=True)
    def write_data(self,info,start_line=1):
        for c in info:
            for info_part in c:
                gap = 0
                for ds in info_part:
                    for i,key in enumerate(ds.keys()):
                        if start_line == 1:
                            self.worksheet.write(0, i + 1 + gap, label=key)
                        self.worksheet.write(start_line, i + 1 + gap, label=ds[key])
                        print(key,ds[key])

                    gap += 6
                start_line += 1
        # for i,key in enumerate(sheetvalues.keys()):
        #     print("test1",i,key)
        #     self.worksheet.write(start_line,i+1,label=key)
        #     for j,val in enumerate(sheetvalues[key]):
        #         self.worksheet.write(start_line+1+j,i+1,label=val)
    def save(self,filename="temp"):
        self.workbook.save("excel/"+filename+".xls")
class ExcelSaver3():
    def __init__(self,sheetname="sheet1"):
        self.sheetname = sheetname
        self.workbook = xlwt.Workbook(encoding='ascii')
        self.worksheet = self.workbook.add_sheet(sheetname,cell_overwrite_ok=True)
    def write_data(self,info,start_line=1):
        for info_part in info:
            gap = 0
            for ds in info_part:
                for i,key in enumerate(ds.keys()):
                    if start_line == 1:
                        self.worksheet.write(0, i + 1 + gap, label=key)
                    self.worksheet.write(start_line, i + 1 + gap, label=str(ds[key]))
                    print(key,ds[key])
                gap += 6
            start_line += 1

    def save(self,filename="temp",dirname="temp"):
        path = "excel/"+dirname
        if not os.path.exists(path):
            os.makedirs(path)
        self.workbook.save(path+"/"+filename+".xls")
class PeakSaver():
    def __init__(self,sheetname="sheet1"):
        self.sheetname = sheetname
        self.workbook = xlwt.Workbook(encoding='ascii')
        self.worksheet = self.workbook.add_sheet(sheetname,cell_overwrite_ok=True)
    def write_data(self,info,start_line=1):
        for ds in info:
            for i,key in enumerate(ds.keys()):
                if start_line == 1:
                    self.worksheet.write(0, i , label=key)
                self.worksheet.write(start_line, i , label=ds[key])
            start_line += 1
    def save(self,filename="temp",dirname="temp"):
        self.workbook.save("excel/"+dirname+"/"+filename+".xls")
class ListSaver():
    def __init__(self,sheetname="sheet1"):
        self.sheetname = sheetname
        self.workbook = xlwt.Workbook(encoding='ascii')
        self.worksheet = self.workbook.add_sheet(sheetname,cell_overwrite_ok=True)
    def write_data(self,listData,start_line=1):
        self.worksheet.write(0, 0, label="cube")
        self.worksheet.write(0, 1, label="feature")
        for d in listData:
            self.worksheet.write(start_line, 0, label="cube"+str(start_line))
            self.worksheet.write(start_line, 1 , label=d)
            start_line += 1
    def save(self,filename="temp"):
        self.workbook.save("excel/"+filename+".xls")
# dict = {"name":[1,2,3,4,5],"age":[23,45,56,33,88]}
# excel = ExcelSaver()
# excel.write_data(dict)
# excel.save("test")

