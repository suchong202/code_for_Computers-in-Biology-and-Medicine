import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from ExcelSaver import *
def normal1(data):
    zero = 2.060437
    data_norm = list(map(lambda x:(x-zero)/(2.9-zero),data))
    return data_norm
def normal2(data):
    zero = 2.174906
    data_norm = list(map(lambda x:(x-zero)/(2.9-zero),data))
    return data_norm

class ReadFile():
    def __init__(self,path):
        self.path = path
    def reader(self):
        data_excel = pd.read_excel(self.path, usecols=[0, 1,2])
        data_excel = np.array(data_excel)
        time = data_excel[:, 0][50:]
        dat1 = data_excel[:, 1][50:]
        dat2 = data_excel[:, 2][50:]

        return time,normal1(dat1),normal2(dat2)
class ReadFile2():
    def __init__(self,path):
        self.path = path
    def reader(self):
        data_excel = pd.read_excel(self.path, usecols=[0, 1,2])
        data_excel = np.array(data_excel)

        dat1 = data_excel[:, 1]
        dat2 = data_excel[:, 2]
        time = np.array(range(1,len(dat1)+1))

        return time,normal1(dat1),normal2(dat2)


class FindPeak():
    def __init__(self,times,dats,threshold=0.2,min_width=10):
        self.threshold = threshold
        self.min_width = min_width
        self.times = times
        self.dats = dats
    def getPeak(self):
        is_peak = False
        peak_dat_part = []
        peak_time_part = []
        peak_time = []
        peak_dat = []

        for t,d in zip(self.times,self.dats):
            if d >= self.threshold:
                is_peak = True
                peak_dat_part.append(d)
                peak_time_part.append(t)
            else:
                if is_peak is True and len(peak_dat_part)>self.min_width:
                    peak_time.append(peak_time_part)
                    peak_dat.append(peak_dat_part)
                is_peak = False
                peak_dat_part = []
                peak_time_part = []

        return peak_time,peak_dat

class ComposeWindow():
    def __init__(self,time1,time2,dat1,dat2):
        self.time1 = time1
        self.time2 = time2
        self.dat1 = dat1
        self.dat2 = dat2
    def compose(self):
        windows = []
        sub_win = []
        i,j = 0,0
        while True:
            t1 = self.time1[i]
            t2 = self.time2[j]
            if t1[0] < t2[0]:
                if len(windows) == 0:
                    sub_win.append({"position":1,"dat":self.dat1[i],"time":self.time1[i]})
                else:
                    windows.append([windows[-1][-1],{"position":1,"dat":self.dat1[i],"time":self.time1[i]}])
                i += 1
            else:
                if len(windows) == 0:
                    sub_win.append({"position":2,"dat":self.dat2[j],"time":self.time2[j]})
                else:
                    windows.append([windows[-1][-1],{"position":2,"dat":self.dat2[j],"time":self.time2[j]}])
                j += 1
            if len(sub_win) == 2:
                windows.append(sub_win)
                sub_win = []

            if i == len(self.dat1) or j == len(self.dat2):
                break
        left_dat = []
        left_time = []
        position = 0
        if i==len(self.dat1):
            left_dat = self.dat2[j:]
            left_time = self.time2[j:]

            position = 2
        elif j == len(self.dat2):
            left_dat = self.dat1[i:]
            left_time = self.time1[i:]
            position = 1

        if len(left_dat) == 1:
            temp = windows[-1][1]
            windows.append([temp,{"position": position, "dat": left_dat[0],"time":left_time[0]}])
        elif len(left_dat) > 1:
            for i in range(len(left_dat)-1):
                windows.append([{"position":position,"dat":left_dat[i],"time":left_time[i]},
                                {"position":position,"dat":left_dat[i+1],"time":left_time[i]}])

        return windows
class PeakCalculation():
    def __init__(self,peak_time,peak_dat,position=1):
        self.peak_dat = peak_dat
        self.peak_time = peak_time
        self.position = position
    def calc(self):
        result = []
        for time_part,peak_part in zip(self.peak_time,self.peak_dat):

            position = None
            if self.position == 1:
                position = "01"
            elif self.position == 2:
                position = "10"


            timeLong = time_part[-1] - time_part[0] + 1
            integrate = sum(peak_part)
            max_peak = max(peak_part)
            average = integrate/timeLong
            d = {"pos":position,"max":max_peak,"integrate":integrate,"average":average,"timeLong":timeLong}
            result.append(d)
        return result
class ComposeCube():
    def __init__(self,windows):
        self.windows = windows
    def createCube(self):
        length = len(self.windows)
        cube = []
        for i in range(length-3):
            sub_cube = self.windows[i:i+4]
            cube.append(sub_cube)
        return cube

class Calculation():
    def __init__(self,cube):
        self.cube = cube
    def calc(self):
        result = []
        for c in self.cube:
            wave = c[0][0]
            if wave["position"] == 1:
                position = "01"
            elif wave["position"] == 2:
                position = "10"
            time = wave["time"]
            timeLong = time[-1]-time[0]+1
            integrate = sum(wave["dat"])
            max_peak = max(wave["dat"])
            average = integrate/timeLong
            d = {"pos":position,"max":max_peak,"integrate":integrate,"average":average,"timeLong":timeLong}
            result.append(d)
        return result


class Select():
    def __init__(self,waves,vectors):
        self.waves = waves
        self.vectors = vectors
    def binary2int(self,b):
        return int(b[0])*2 + int(b[1])

    def calc(self,v1,v2):
        s = 0
        for i in range(len(v1)):
            if i == 0:
                reduce = self.binary2int(v1[i])-self.binary2int(v2[i])
                temp = reduce * reduce
            else:
                temp = (v1[i]-v2[i])*(v1[i]-v2[i])
            s += temp
        return math.sqrt(s)
    def choice(self):
        result = []
        for w in self.waves:
            features = []
            for v in self.vectors:
                features.append(self.calc(w,v))
            value = min(features)
            index= features.index(value)
            if index in [1,4,6,7]:
                result.append(value)
        return result


class CubeFeature1():
    def __init__(self,cube,features):
        self.cube = cube
        self.featues = features
    def getMean(self,x1,x2):
        return (x1+x2)/2
    def getDx(self,x1,y1):
        u = self.getMean(x1,y1)
        return ((x1-u)**2 + (y1-u)**2)/2

    def getDistance(self,feature,dat):
        total = 0
        for x,y in zip(feature,dat):
            s = self.getDx(x,y)
            if s != 0:
                temp = (x-y)**2/(s**2)
                total += temp
        return math.sqrt(total)
    def getFeature(self,sub_win):
        position = None
        if sub_win["position"] == 1:
            position = 1
        elif sub_win["position"] == 2:
            position = 10

        timeLong = sub_win["time"][-1] - sub_win["time"][0] + 1

        integrate = sum(sub_win["dat"])
        max_peak = max(sub_win["dat"])
        average = integrate / timeLong
        d = [position, max_peak, integrate, average, timeLong]

        return d

    def findFeature(self):
        result = []
        reminder_cube = []
        for c in self.cube:
            window1 = c[0]
            peak1 = window1[0]
            peak_feature = self.getFeature(peak1)
            ds = []

            for feature in self.featues:
                if peak_feature[0] == 1 and feature[0] == Feature_flag:
                    d = self.getDistance(feature[1:],peak_feature[1:])

                    ds.append(d)
                else:
                    ds.append(-1)

            classification = ds.index(max(ds)) + 1

            if classification == Feature_num:
                reminder_cube.append(c)

            result.append(classification)
        return result,reminder_cube

class GetInfo():
    def __init__(self,times,dat1,dat2):
        self.times = list(times)
        self.dat1 = list(dat1)
        self.dat2 = list(dat2)
    def findMinPoint(self,dat,index_b,dire=1):
        i = index_b
        value = dat[i]
        index = i
        while True:
            temp = dat[i]
            if temp < 0 or temp > value:
                break
            else:
                value = dat[i]
                index = i
            if dire == 1:
                i += 1
            else:
                i -= 1
        return index,value
    def getDetailInfo(self,position,point1,point2):
        if position == 1:
            pos = "0001"
            dat_part = self.dat1[point1:point2]
            time_part = self.times[point1:point2]
            i = sum(dat_part)
            time = max(time_part) - min(time_part) + 1
            max_value = max(dat_part)
            ave = i / time
            info = {"pos": pos, "max": max_value, "integrate": i, "average": ave, "timeLong": time}
            return info,dat_part
        else:
            pos = "1000"
            dat_part = self.dat2[point1:point2]
            time_part = self.times[point1:point2]
            i = sum(dat_part)
            time = max(time_part) - min(time_part) + 1
            max_value = max(dat_part)
            ave = i / time
            info = {"pos":pos,"max":max_value,"integrate":i,"average":ave,"timeLong":time}
            return info,dat_part

    def getDetailInfo2(self, position1,position2, point1, point2):
        if position1 == 1:

            if position2 == 1:
                pos = "0001"
            else:
                pos = "0010"
            dat_part = self.dat1[point1:point2]
            time_part = self.times[point1:point2]
            i = sum(dat_part)
            time = max(time_part) - min(time_part) + 1
            max_value = max(dat_part)
            ave = i / time
            info = {"pos": pos, "max": max_value, "integrate": i, "average": ave, "timeLong": time}
            return info,dat_part
        else:
            pos = "1000"
            dat_part = self.dat2[point1:point2]
            time_part = self.times[point1:point2]
            i = sum(dat_part)
            time = max(time_part) - min(time_part) + 1
            max_value = max(dat_part)
            ave = i / time
            info = {"pos": pos, "max": max_value, "integrate": i, "average": ave, "timeLong": time}
            return info,dat_part

    def find_points(self,win1,win2):

        time_a = win1["time"][0]
        time_b = win1["time"][-1]
        time_d = win2["time"][0]
        time_e = win2["time"][-1]

        index_a = time_a - 1
        index_b = time_b - 1
        index_d = time_d - 1
        index_e = time_e - 1

        index_c = None
        #find the c point
        if win1["position"] == 1:
            index_c,value_c = self.findMinPoint(self.dat1,index_b)
            time_c = index_c + 1
        elif win1["position"] == 2:
            index_c,value_c = self.findMinPoint(self.dat2,index_b)
            time_c = index_c + 1
        return index_a,index_b,index_c,index_d,index_e

    def find_window_info(self,windows):

        win1,win2 = windows[0],windows[1]
        index_a,index_b,index_c,index_d,index_e = self.find_points(win1,win2)


        #get ab
        info_ab,dat_ab = self.getDetailInfo(win1["position"],index_a,index_b)
        #get bc
        info_bc,dat_bc = self.getDetailInfo2(win1["position"],win2["position"],index_b,index_c)
        #get de
        info_de,dat_de = self.getDetailInfo(win2["position"],index_d,index_e)
        #get cd
        info_cd = None
        dat_cd = None
        if win2["position"] == 1:
            index_zero,_ = self.findMinPoint(self.dat1,index_d,dire=2)
            # print("index_zero1:",index_zero,index_d)
            dat_part = self.dat1[index_zero:index_d]
            pos = "0001"
            i = sum(dat_part)
            time = index_d-index_c
            max_value = max(dat_part)
            ave = i / time
            info_cd = {"pos": pos, "max": max_value, "integrate": i, "average": ave, "timeLong": time}
            dat_cd = dat_part
        elif win2["position"] == 2:
            index_zero,_ = self.findMinPoint(self.dat2, index_d,dire=2)
            # print("index_zero2:",index_zero,index_d)
            dat_part = self.dat2[index_zero:index_d]
            pos = "1000"
            if win1["position"] == 1:
                pos = "0100"

            i = sum(dat_part)
            time = index_d - index_c
            max_value = max(dat_part)
            ave = i / time
            info_cd = {"pos": pos, "max": max_value, "integrate": i, "average": ave, "timeLong": time}
            dat_cd = dat_part
        index_info = [index_a, index_b, index_c, index_d, index_e,index_zero]
        return [info_ab,info_bc,info_cd,info_de],index_info,[dat_ab,dat_bc,dat_cd,dat_de]
    def find_cube_info(self,Cube):
        result = []
        for windows in Cube:
            win_info,index_info,dat_info = self.find_window_info(windows)
            result.append([win_info,index_info,dat_info])
        return result


threshold = 0.4
min_width = 10

vectors = [None for i in range(8)]
vectors[0] = [1,0.27,7.6,0.2,36]
vectors[1] = [10,0.31,10.5,0.25,42]
vectors[2] = [1, 1.2, 41.3, 1.0, 40]#
vectors[3] = [10,1.7,92,1.17,77]
vectors[4] = [1,0.35,37,0.29,118]
vectors[5] = [10,0.31,30,0.24,120]
vectors[6] = [1,1.2,120,0.82,133]
vectors[7] = [10,1.34,135,1.0,136]

filename = "name"
Feature_num = 8
Feature_flag = 10
reader = ReadFile2(path="data2/%s.xls"%(filename))
time,dat1,dat2 = reader.reader()


finder1 = FindPeak(time,dat1,threshold=threshold,min_width=min_width)
finder2 = FindPeak(time,dat2,threshold=threshold,min_width=min_width)

peak_time1,peak_dat1 = finder1.getPeak()
peak_time2,peak_dat2 = finder2.getPeak()
print("peaks length:",len(peak_dat1),len(peak_dat2))


"""
{"position":1,"dat":self.dat1[i],"time":left_time[i]}
"""
composer = ComposeWindow(peak_time1,peak_time2,peak_dat1,peak_dat2)
windows = composer.compose()

print("window number:",len(windows))


creater = ComposeCube(windows)
cube = creater.createCube()

print("cube length:",len(cube))

cubefeature1 = CubeFeature1(cube=cube,features=vectors)
classification,reminder_cube = cubefeature1.findFeature()
print(classification)

info = GetInfo(time,dat1,dat2)
cube_nums = 0
plt.figure()
for c in reminder_cube:
    cube_info = info.find_cube_info(c)
    print("cube:", cube_nums)
    win_nums = 0
    for win_info in cube_info:
        w_info, index_info, dat_info = win_info

        plt.plot(time[min(index_info):max(index_info)],dat1[min(index_info):max(index_info)],color="black",
                 label="channel1")
        plt.plot(time[min(index_info): max(index_info)], dat2[min(index_info):max(index_info)], color="gray",
                 label="channel2")
        plt.plot(time[min(index_info): max(index_info)],[threshold for i in range(max(index_info)-min(index_info))],color="red",label="threshold")
        plt.plot(time[min(index_info[0],index_info[1]):max(index_info[0],index_info[1])],dat_info[0],color="gold",label="ab")
        plt.plot(time[min(index_info[1],index_info[2]):max(index_info[1],index_info[2])], dat_info[1],color="skyblue", label="bc")
        plt.plot(time[min(index_info[5],index_info[3]):max(index_info[5],index_info[3])], dat_info[2],color="purple", label="cd")
        plt.plot(time[min(index_info[3],index_info[4]):max(index_info[3],index_info[4])], dat_info[3],color="green", label="de")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("U")
        plt.savefig('images/{}/cube{}window{}.png'.format(filename,cube_nums,win_nums))
        plt.clf()
        win_nums += 1

    cube_nums += 1



# plt.figure()
# plt.plot(time[:5000],dat1[:5000],color="red")
# plt.plot(time[:5000],[threshold for i in range(5000)],color="blue")
# plt.plot(w[0]["time"],w[0]["dat"],color="purple")
# plt.plot()
# plt.plot(time[:5000],dat2[:5000],color="black")
# plt.show()
