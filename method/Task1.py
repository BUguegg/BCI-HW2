import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.optimize as optimize
import h5py
import torch
import math

class Neural_Spike_Visual():
    def __init__(self):
        #load Data
        data_path = './data/neural_Data.mat'
        neural_data = scipy.io.loadmat(data_path)

        self.chan_names = np.array(neural_data['chan_names'])
        self.chan = [name[0][0] for name in self.chan_names]
        self.cursor_pos = np.array(neural_data['cursor_pos'])
        self.finger_pos = np.array(neural_data['finger_pos'])
        self.spikes = np.array(neural_data['spikes'])
        self.t = np.array(neural_data['t'])
        self.target_pos = np.array(neural_data['target_pos'])
        self.wf = np.array(neural_data['wf'])

        # print("chan_names.shape:",self.chan_names.shape)
        # print("cursor_pos.shape:",self.cursor_pos.shape)
        # print("finger_pos.shape:",self.finger_pos.shape)
        # print("spikes.shape:",self.spikes.shape)
        # print("t.shape:",self.t.shape)
        # print("target_pos.shape:",self.target_pos.shape)
        # print("wf.shape:",self.wf.shape)

        self.resultPath = './result/Task1'
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
        start = 0
        t = 0
        for id, spike in enumerate(self.spikes[channel][u]):
            if spike >= tSlice[t]:
                SpikeSlice.append(self.spikes[channel][u][start:id])
                start = id
                if t == len(tSlice) - 1:
                    SpikeSlice.append(self.spikes[channel][u][id:-1])
                    break
                else:
                    t += 1
            else:
                continue

        head = SpikeSlice.pop(0)
        tail = SpikeSlice.pop(-1)
        # print(len(SpikeSlice))
        # print(len(head))
        # print(len(tail))
        return  SpikeSlice

    def Classification(self, moterStateSlice, t_pos, SpikeSlice):
        SpkiceClass = [[] for i in range(8)]
        MotorClass = [[] for i in range(8)]
        # print(np.array(t_pos).shape)
        x = np.array(t_pos).T[0]
        y = np.array(t_pos).T[1]

        for i in range(1, len(x)):
            angle = math.atan2((y[i] - y[i-1]) , ( x[i] - x[i-1]))
            angle = 180 - int(angle * 180 / math.pi)
            if angle >= 0 and angle < 45:
                SpkiceClass[0].append(SpikeSlice[i-1])
                MotorClass[0].append(moterStateSlice[i-1])
            elif angle >= 45 and angle < 90:
                SpkiceClass[1].append(SpikeSlice[i-1])
                MotorClass[1].append(moterStateSlice[i-1])
            elif angle >= 90 and angle < 135:
                SpkiceClass[2].append(SpikeSlice[i-1])
                MotorClass[2].append(moterStateSlice[i-1])
            elif angle >= 135 and angle < 180:
                SpkiceClass[3].append(SpikeSlice[i-1])
                MotorClass[3].append(moterStateSlice[i-1])
            elif angle >= 180 and angle < 225:
                SpkiceClass[4].append(SpikeSlice[i-1])
                MotorClass[4].append(moterStateSlice[i-1])
            elif angle >= 225 and angle < 270:
                SpkiceClass[5].append(SpikeSlice[i-1])
                MotorClass[5].append(moterStateSlice[i-1])
            elif angle >= 270 and angle < 315:
                SpkiceClass[6].append(SpikeSlice[i-1])
                MotorClass[6].append(moterStateSlice[i-1])
            elif angle >= 315 and angle < 360:
                SpkiceClass[7].append(SpikeSlice[i-1])
                MotorClass[7].append(moterStateSlice[i-1])
        
        # for j in range(len(SpkiceClass)):
        #     print(len(SpkiceClass[j]))
        #     print(len(MotorClass[j]))
        #     print("###################")

        return SpkiceClass, MotorClass


    #Raster plot
    def Raster(self):
        #Original Data [u1,u2,u3] Raster
        sort = [n for n in range(len(self.spikes.T))]
        spikeData = []
        for s in sort:
            for i in range(len(self.spikes.T[s])):
                sp = np.reshape(self.spikes.T[s][i],(len(self.spikes.T[s][i])))
                spikeData.append(sp)
            fig = plt.figure()
            plt.title("u"+str(s)+"Raster")
            plt.eventplot(spikeData)
            fig.savefig("result/Task1/u"+str(s)+"Raster.png")
            plt.close()

        #Raster according to motor state
        moterStateSlice, tSlice = self.sliceMotor()
        SpikeSlice = self.sliceSpike(channel=1, u=1, tSlice=tSlice)

    #PSTH plot
    def PSTH(self):
        moterStateSlice, tSlice, t_pos = self.sliceMotor()
        SpikeSlice= self.sliceSpike(channel=1, u=0, tSlice=tSlice)
        SpkiceClass, MotorClass =  self.Classification(moterStateSlice, t_pos, SpikeSlice)
        
        window = 0.1
        for idx,direct in enumerate(SpkiceClass):
            # print(len(direct))
            for id,spikeList in enumerate(direct):
                # print(spikeList)

                #count spike
                count = []
                timestamp = []
                
                lim = spikeList[0][0] + window
                # timestamp.append(lim)
                if lim < spikeList[-1][0]:
                    num = 0
                    for i, spike in enumerate(spikeList):
                        if  lim >= spike[0]:
                            num += 1
                        elif lim < spike[0] and lim + window < spikeList[-1][0]:
                            timestamp.append(lim)
                            count.append(num)
                            lim += window
                            num = 1
                        elif lim + window > spikeList[-1][0]:
                            count.append(len(spikeList[i:]))
                            timestamp.append(lim + window)

                fig = plt.figure()
                # plt.bar(timestamp, count, width=0.05, edgecolor='black')
                plt.hist(count,bins=30,edgecolor='black')
                # plt.hist(count,edgecolor='black')
                # plt.hist(np.reshape(np.array(spikeList),(len(spikeList))),bins=20)
                # plt.show()
                fig.savefig("F:\\Course\\BCI_HW2\\result\\Task1\\PSTH\\"+str(idx * 45)+"~"+str((idx+1)*45)+"\\"+str(id)+".png")
                plt.close()
            #     break
            # break

    #Tuning curve
    def Tuning_curve(self):
        moterStateSlice, tSlice, t_pos = SpikeImg.sliceMotor()
        SpikeSlice= SpikeImg.sliceSpike(channel=1, u=0, tSlice=tSlice)
        SpkiceClass, MotorClass =  self.Classification(moterStateSlice, t_pos, SpikeSlice)
        
        def target_func(x, a0, a1, a2, a3):
            return a0 * np.sin(a1 * x + a2) + a3

        window = 0.1
        for idx,direct in enumerate(SpkiceClass):
            # print(len(direct))
            for id,spikeList in enumerate(direct):
                # print(spikeList)

                #count spike
                count = []
                timestamp = []
                
                lim = spikeList[0][0] + window
                # timestamp.append(lim)
                if lim < spikeList[-1][0]:
                    num = 0
                    for i, spike in enumerate(spikeList):
                        if  lim >= spike[0]:
                            num += 1
                        elif lim < spike[0] and lim + window < spikeList[-1][0]:
                            timestamp.append(lim)
                            count.append(num)
                            lim += window
                            num = 1
                        elif lim + window > spikeList[-1][0]:
                            count.append(len(spikeList[i:]))
                            timestamp.append(lim + window)

                if len(timestamp) < 1:
                    continue
                else:
                    # fig, ax = plt.subplots()
                    fig = plt.figure()
                    # ax.scatter(timestamp,count,c='r')
                    plt.scatter(timestamp,count,c='r')
                    fs = np.fft.fftfreq(len(timestamp), timestamp[1] - timestamp[0])
                    Y = abs(np.fft.fft(count))
                    freq = abs(fs[np.argmax(Y[1:]) + 1])
                    a0 = max(count) - min(count)
                    a1 = 2 * np.pi * freq
                    a2 = 0
                    a3 = np.mean(count)
                    p0 = [a0, a1, a2, a3]
                    para, _ = optimize.curve_fit(target_func, timestamp, count, p0=p0, maxfev=800000)
                    print(para)
                    y_fit = [target_func(a, *para) for a in timestamp]
                    
                    # ax.plot(timestamp, y_fit, 'g')
                    plt.plot(timestamp, y_fit, 'g')

                    # print(count)
                    # print(timestamp)
                    # plt.plot(timestamp,count)
                    # plt.scatter(timestamp,count)

                    # plt.show()
                    fig.savefig("F:\\Course\\BCI_HW2\\result\\Task1\Tuning_Curve\\"+str(idx * 45)+"~"+str((idx+1)*45)+"\\"+str(id)+".png")
                    plt.close()
            #     break
            # break
        


    #Center_out
    def Center_out(self):
        #2D
        # cursor = self.cursor_pos[:100000]
        cursor = self.cursor_pos
        fig = plt.figure()
        # plt.plot(self.cursor_pos.T[0],self.cursor_pos.T[1], "b")
        # plt.plot(cursor.T[0], cursor.T[1], "b")
        plt.scatter(self.target_pos.T[0], self.target_pos.T[1], c="r")
        # fig.savefig("result/Task1/MotorTrace.png")
        fig.savefig("result/Task1/MotorSite.png")

        #Animation
        # strid = 1000
        # for x in range(0,len(self.target_pos),strid):
            # print(x)
            # plt.scatter(self.target_pos[x:x+strid].T[0], self.target_pos[x:x+strid].T[1], c="r")
            # fig.savefig("result/Task1/MotorS.png")
        
        #3D
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = self.finger_pos.T[0]
        y = self.finger_pos.T[1]
        z = self.finger_pos.T[2]
        ax.plot(x, y, z, label='Finger Moter Trace', c="b")
        # site = [0 for i in range(len(self.target_pos.T))]
        ax.scatter(self.target_pos.T[0],self.target_pos.T[0], c='r')
        fig.savefig("result/Task1/MotorSite3D.png")
        self.finger_pos.T[1]
 

if __name__=="__main__":
    SpikeImg = Neural_Spike_Visual()
    # moterStateSlice, tSlice, t_pos = SpikeImg.sliceMotor()
    # SpikeSlice = SpikeImg.sliceSpike(channel=1, u=1, tSlice=tSlice)
    # SpikeImg.Classification()
    # SpikeImg.Center_out()
    # SpikeImg.Raster()
    # SpikeImg.PSTH()
    SpikeImg.Tuning_curve()


