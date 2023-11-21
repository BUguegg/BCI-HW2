import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy
import math
from scipy.spatial.distance import cdist

class Neural_Motor_Map():
    def __init__(self):
        #load Data
        data_path = './data/neural_Data.mat'          #indy_20160407_02.mat
        neural_data = scipy.io.loadmat(data_path)

        self.chan_names = np.array(neural_data['chan_names'])
        self.chan = [name[0][0] for name in self.chan_names]
        self.cursor_pos = np.array(neural_data['cursor_pos'])
        self.finger_pos = np.array(neural_data['finger_pos'])
        self.spikes = np.array(neural_data['spikes'])
        self.t = np.array(neural_data['t'])
        self.target_pos = np.array(neural_data['target_pos'])
        self.wf = np.array(neural_data['wf'])

        # print(len(data_path))


        new_data_path = './data/indy_20170124.mat'        #indy_20170124_01.mat
        new_neural_data = scipy.io.loadmat(new_data_path)

        self.new_chan_names = np.array(new_neural_data['chan_names'])
        self.new_chan = [name[0][0] for name in self.new_chan_names]
        self.new_cursor_pos = np.array(new_neural_data['cursor_pos'])
        self.new_finger_pos = np.array(new_neural_data['finger_pos'])
        self.new_spikes = np.array(new_neural_data['spikes'])
        self.new_t = np.array(new_neural_data['t'])
        self.new_target_pos = np.array(new_neural_data['target_pos'])
        self.new_wf = np.array(new_neural_data['wf'])

        # print(len(new_data_path))

        # print("chan_names.shape:",self.chan_names.shape)
        # print("cursor_pos.shape:",self.cursor_pos.shape)
        # print("finger_pos.shape:",self.finger_pos.shape)
        # print("spikes.shape:",self.spikes.shape)
        # print("t.shape:",self.t.shape)
        # print("target_pos.shape:",self.target_pos.shape)
        # print("wf.shape:",self.wf.shape)

        self.resultPath = './result/Task2'
        pass
    
    #Slice Data according to self.target_pos
    def sliceMotor(self, target_pos, cursor_pos, t):
        moterStateSlice = []
        t_pos = [target_pos[0]]   #start 0
        tSlice = [t[0][0]]       #start 0
        # tSlice = []
        comx, comy = target_pos[0]
        sit = 0
        for idx, co in enumerate(target_pos):
            x = co[0]
            y = co[1]         
            if x == comx and y == comy :
                continue
            else:
                # print(idx)
                comx = x
                comy = y
                t_pos.append(co)
                moterStateSlice.append(cursor_pos[sit:idx]) 
                tSlice.append(t[idx][0])
                sit = idx
                # break
        # print(len(moterStateSlice))
        # print("tSlice:",len(tSlice))
        # print(len(t_pos))
        # print(t_pos[-1])
        return moterStateSlice, tSlice, t_pos

    #Slice Spikes according to self.t
    def sliceSpike(self, channel, u, tSlice, spikes, wf):
        SpikeSlice = []
        wfSlice = []
        start = 0
        t = 0
        for id, spike in enumerate(spikes[channel][u]):
            if spike >= tSlice[t]:
                SpikeSlice.append(spikes[channel][u][start:id])
                wfSlice.append(wf[channel][u][start:id])
                start = id
                if t == len(tSlice) - 1:
                    SpikeSlice.append(spikes[channel][u][id:-1])
                    wfSlice.append(wf[channel][u][start:id])
                    break
                else:
                    t += 1
            else:
                continue

        head = SpikeSlice.pop(0)
        tail = SpikeSlice.pop(-1)
        wfSlice.pop(0)
        wfSlice.pop(-1)
        # print("SpikeSlice:",len(SpikeSlice))
        # print("wfSlice:",len(wfSlice))
        # print(len(head))
        # print(len(tail))
        return  SpikeSlice, wfSlice

    def Classification(self, moterStateSlice, t_pos, SpikeSlice, wfSlice, tSlice):
        SpkiceClass = [[] for i in range(8)]
        MotorClass = [[] for i in range(8)]
        wfClass = [[] for i in range(8)]
        tClass = [[] for i in range(8)]
        # print(np.array(t_pos).shape)
        x = np.array(t_pos).T[0]
        y = np.array(t_pos).T[1]

        for i in range(1, len(x)):
            angle = math.atan2((y[i] - y[i-1]) , ( x[i] - x[i-1]))
            angle = 180 - int(angle * 180 / math.pi)
            if angle >= 0 and angle < 45:
                SpkiceClass[0].append(SpikeSlice[i-1])
                MotorClass[0].append(moterStateSlice[i-1])
                wfClass[0].append(wfSlice[i-1])
                tClass[0].append([tSlice[i-1],tSlice[i]])
            elif angle >= 45 and angle < 90:
                SpkiceClass[1].append(SpikeSlice[i-1])
                MotorClass[1].append(moterStateSlice[i-1])
                wfClass[1].append(wfSlice[i-1])
                tClass[1].append([tSlice[i-1],tSlice[i]])
            elif angle >= 90 and angle < 135:
                SpkiceClass[2].append(SpikeSlice[i-1])
                MotorClass[2].append(moterStateSlice[i-1])
                wfClass[2].append(wfSlice[i-1])
                tClass[2].append([tSlice[i-1],tSlice[i]])
            elif angle >= 135 and angle < 180:
                SpkiceClass[3].append(SpikeSlice[i-1])
                MotorClass[3].append(moterStateSlice[i-1])
                wfClass[3].append(wfSlice[i-1])
                tClass[3].append([tSlice[i-1],tSlice[i]])
            elif angle >= 180 and angle < 225:
                SpkiceClass[4].append(SpikeSlice[i-1])
                MotorClass[4].append(moterStateSlice[i-1])
                wfClass[4].append(wfSlice[i-1])
                tClass[4].append([tSlice[i-1],tSlice[i]])
            elif angle >= 225 and angle < 270:
                SpkiceClass[5].append(SpikeSlice[i-1])
                MotorClass[5].append(moterStateSlice[i-1])
                wfClass[5].append(wfSlice[i-1])
                tClass[5].append([tSlice[i-1],tSlice[i]])
            elif angle >= 270 and angle < 315:
                SpkiceClass[6].append(SpikeSlice[i-1])
                MotorClass[6].append(moterStateSlice[i-1])
                wfClass[6].append(wfSlice[i-1])
                tClass[6].append([tSlice[i-1],tSlice[i]])
            elif angle >= 315 and angle < 360:
                SpkiceClass[7].append(SpikeSlice[i-1])
                MotorClass[7].append(moterStateSlice[i-1])
                wfClass[7].append(wfSlice[i-1])
                tClass[7].append([tSlice[i-1],tSlice[i]])
        
        # for j in range(len(SpkiceClass)):
        #     print(len(SpkiceClass[j]))
        #     print(len(MotorClass[j]))
        #     print("###################")

        return SpkiceClass, MotorClass, wfClass, tClass
    
    def  SplitTinClass(self,tClass):

        tLast = [[] for i in range(8)]
        for idx,tC in enumerate(tClass):
            for last in tC:
                tLast[idx].append(last[1] - last[0])

        return tLast
    

    def SpeedCompare(self):

        #get 16 tClass
        moterStateSlice, tSlice, t_pos = self.sliceMotor(self.target_pos, self.cursor_pos, self.t)
        # print(tSlice)
        print(len())
        print(len())
        print(len())
        SpikeSlice, wfSlice = self.sliceSpike(2, 0, tSlice, self.spikes, self.wf)
        print(len())
        print(len())
        SpikeClass, MotorClass, wfClass, tClass =  self.Classification(moterStateSlice, t_pos, SpikeSlice, wfSlice, tSlice)
        print(len())
        print(len())
        print(len())
        
        t1 = self.SplitTinClass(tClass)

        #get 17 tClass
        moterStateSlice1, tSlice1, t_pos1 = self.sliceMotor(self.new_target_pos, self.new_cursor_pos, self.new_t)
        SpikeSlice1, wfSlice1 = self.sliceSpike(2, 0, tSlice, self.new_spikes, self.new_wf)
        SpikeClass1, MotorClass1, wfClass1, tClass1 =  self.Classification(moterStateSlice1, t_pos1, SpikeSlice1, wfSlice1, tSlice1)
 
        t2 = self.SplitTinClass(tClass1)

        #Compare len between t1 and t2
        direct = 0      #choose direct
        # print("before t1:",len(t1[direct]))
        # print("before t2:",len(t2[direct]))
        if len(t1[direct]) > len(t2[direct]):
            t1[direct] = t1[direct][:len(t2[direct])]
        else:
            t2[direct] = t2[direct][:len(t1[direct])]
        
        # print("after t1:",len(t1[direct]))
        # print("after t2:",len(t2[direct]))

        #calculate averge
        average1 = []
        average2 = []
        for leng in range(10, len(t1[direct]), 10):
            # print(leng)
            average1.append(np.mean(t1[direct][leng-10:leng]))
            average2.append(np.mean(t2[direct][leng-10:leng]))
        
        print(average1)
        print(average2)

        # x = [j for j in range(len(average1))]

        # fig = plt.figure()
        width = 0.35
        labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
        x = np.arange(len(labels)) 
        fig, ax = plt.subplots()
        
        rects1 = ax.bar(x - width/2, average1, width, label='indy_20160407_02')
        rects2 = ax.bar(x + width/2, average2, width, label='indy_20170124_01')
        ax.set_ylabel('Average Time')
        ax.set_xticks(x)
        ax.legend()
        # plt.bar(x - width/2,average1, width=0.3, color='r')
        # plt.bar(x + width/2,average2, width=0.3, color='b')
        fig.savefig("result/Task2/compare.png")
        
        
        
        
        pass
    
    def check(self):
        #get 16 tClass
        moterStateSlice, tSlice, t_pos = self.sliceMotor(self.target_pos, self.cursor_pos, self.t)
        # print(tSlice)
        # print(len(moterStateSlice))
        print(len(tSlice))
        print(tSlice)
        # print(len(t_pos))
        print(np.array(self.wf[30][0]).shape)
        SpikeSlice, wfSlice = self.sliceSpike(2, 0, tSlice, self.spikes, self.wf)
        # print(len(SpikeSlice))
        # print(len(wfSlice))
        # print(np.array(wfSlice).shape)
        SpikeClass, MotorClass, wfClass, tClass =  self.Classification(moterStateSlice, t_pos, SpikeSlice, wfSlice, tSlice)
        # print(len(SpikeClass))
        # print(len(MotorClass))
        # print(len(wfClass))
        # print(len(tClass))
        
    


if __name__ == "__main__":
    mapping = Neural_Motor_Map()
    # mapping.SpeedCompare()
    mapping.check()
