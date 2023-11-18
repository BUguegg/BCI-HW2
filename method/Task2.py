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


        new_data_path = './data/indy_2017.mat'        #indy_20170124_01.mat
        new_neural_data = scipy.io.loadmat(new_data_path)

        self.new_chan_names = np.array(new_neural_data['chan_names'])
        self.new_chan = [name[0][0] for name in self.new_chan_names]
        self.new_cursor_pos = np.array(new_neural_data['cursor_pos'])
        self.new_finger_pos = np.array(new_neural_data['finger_pos'])
        self.new_spikes = np.array(new_neural_data['spikes'])
        self.new_t = np.array(new_neural_data['t'])
        self.new_target_pos = np.array(new_neural_data['target_pos'])
        self.new_wf = np.array(new_neural_data['wf'])

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
    def sliceMotor(self):
        moterStateSlice = []
        t_pos = [self.target_pos[0]]   #start 0
        tSlice = [self.t[0][0]]       #start 0
        # tSlice = []
        comx, comy = self.target_pos[0]
        sit = 0
        for idx, co in enumerate(self.target_pos):
            x = co[0]
            y = co[1]         
            if x == comx and y == comy :
                continue
            else:
                # print(idx)
                comx = x
                comy = y
                t_pos.append(co)
                moterStateSlice.append(self.cursor_pos[sit:idx]) 
                tSlice.append(self.t[idx][0])
                sit = idx
                # break
        # print(len(moterStateSlice))
        # print(len(tSlice))
        # print(len(t_pos))
        # print(t_pos[-1])
        return moterStateSlice, tSlice, t_pos

    #Slice Spikes according to self.t
    def sliceSpike(self, channel, u, tSlice):
        SpikeSlice = []
        wfSlice = []
        start = 0
        t = 0
        for id, spike in enumerate(self.spikes[channel][u]):
            if spike >= tSlice[t]:
                SpikeSlice.append(self.spikes[channel][u][start:id])
                wfSlice.append(self.wf[channel][u][start:id])
                start = id
                if t == len(tSlice) - 1:
                    SpikeSlice.append(self.spikes[channel][u][id:-1])
                    wfSlice.append(self.wf[channel][u][start:id])
                    break
                else:
                    t += 1
            else:
                continue

        head = SpikeSlice.pop(0)
        tail = SpikeSlice.pop(-1)
        wfSlice.pop(0)
        wfSlice.pop(-1)
        # print(len(SpikeSlice))
        # print(len(head))
        # print(len(tail))
        return  SpikeSlice, wfSlice

    def Classification(self, moterStateSlice, t_pos, SpikeSlice, wfSlice):
        SpkiceClass = [[] for i in range(8)]
        MotorClass = [[] for i in range(8)]
        wfClass = [[] for i in range(8)]
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
            elif angle >= 45 and angle < 90:
                SpkiceClass[1].append(SpikeSlice[i-1])
                MotorClass[1].append(moterStateSlice[i-1])
                wfClass[1].append(wfSlice[i-1])
            elif angle >= 90 and angle < 135:
                SpkiceClass[2].append(SpikeSlice[i-1])
                MotorClass[2].append(moterStateSlice[i-1])
                wfClass[2].append(wfSlice[i-1])
            elif angle >= 135 and angle < 180:
                SpkiceClass[3].append(SpikeSlice[i-1])
                MotorClass[3].append(moterStateSlice[i-1])
                wfClass[3].append(wfSlice[i-1])
            elif angle >= 180 and angle < 225:
                SpkiceClass[4].append(SpikeSlice[i-1])
                MotorClass[4].append(moterStateSlice[i-1])
                wfClass[4].append(wfSlice[i-1])
            elif angle >= 225 and angle < 270:
                SpkiceClass[5].append(SpikeSlice[i-1])
                MotorClass[5].append(moterStateSlice[i-1])
                wfClass[5].append(wfSlice[i-1])
            elif angle >= 270 and angle < 315:
                SpkiceClass[6].append(SpikeSlice[i-1])
                MotorClass[6].append(moterStateSlice[i-1])
                wfClass[6].append(wfSlice[i-1])
            elif angle >= 315 and angle < 360:
                SpkiceClass[7].append(SpikeSlice[i-1])
                MotorClass[7].append(moterStateSlice[i-1])
                wfClass[7].append(wfSlice[i-1])
        
        # for j in range(len(SpkiceClass)):
        #     print(len(SpkiceClass[j]))
        #     print(len(MotorClass[j]))
        #     print("###################")

        return SpkiceClass, MotorClass, wfClass
    

    
    

    


if __name__ == "__main__":
    mapping = Neural_Motor_Map()
