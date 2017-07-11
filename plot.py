#!/usr/bin/python
#coding: utf-8

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

#colorblind friendly color
import seaborn as sns
sns.set_palette('colorblind')
current_palette = sns.color_palette()

import const
import subprocess
from os.path import exists

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline


def saveTempLog(log):
    mode = const.MODEL
    numMeasure = const.NUM_EP
    print "Saving model before crash ..." 
    np.save(open("{}logTemp{}{}.npy".format(const.LOG_RL,mode,const.NUM_EXPE),'w'),log)

def loadTempLog():
    mode = const.MODEL
    numMeasure = const.NUM_EP

    tempFileName = "{}logTemp{}{}.npy".format(const.LOG_RL,mode,const.NUM_EXPE)
    print "tempFileName",tempFileName
    if exists(tempFileName):
        print "Experiments logging exists, loading it ..." 
        log = np.load(tempFileName)
        for i in range(log.shape[0]):
            if (log[i,:]==0).all():
                return log, i
        raise ValueError("Log should contain at least one line with only zeros")
    else:
        print "Experiments logging doesn't exist, doing all experiments from scratch" 
        return np.zeros((const.NUM_EXPE,numMeasure)), 0

def plotMeanScores(logAll):

    mode = const.MODEL
    np.save(open("{}logAllScores{}.npy".format(const.LOG_RL,mode),'w'),logAll)
    logScoresMean = logAll.mean(axis=0)
    np.save(open("{}logScoresMean{}.npy".format(const.LOG_RL,mode),'w'),logScoresMean)

    plt.figure()
    plt.plot(logScoresMean)
    #plt.show()
    plt.savefig("{}mean{}.png".format(const.LOG_RL,mode))

    subprocess.call(["rm","{}logTemp{}{}.npy".format(const.LOG_RL,mode,const.NUM_EXPE)])

def plotOneExp(meanScores):
    mode = const.MODEL
    plt.plot(meanScores)
    plt.title("Evolution of rl scores".format(const.PRINT_INFO))
    plt.savefig("{}OneExperiment{}.png".format(const.LOG_RL,mode))
    plt.show()
    
def plot(scoreFileStr):

    width = 2.5
    osefLog = np.load(const.LOG_RL+scoreFileStr)

    plt.plot(osefLog,label='Rl', linewidth=width)
    plt.xlabel('Number of epochs')
    plt.ylabel('Reward')
    plt.title("Evolution of scores on looking task")
    plt.legend(loc=4)
    plt.show()

def plotDiffTrainTest(directory, plot_std=False):

    plot_std = False
    width = 2.5
    name = 'comparison'

    t = np.arange(0,75)

    trueLog = np.load(const.LOG_RL+directory+"logScoresMeantrue.npy.save")
    reprLog = np.load(const.LOG_RL+directory+"logScoresMeanrepr.npy.save")
    endLog = np.load(const.LOG_RL+directory+"logScoresMeanend.npy.save")
    autoLog = np.load(const.LOG_RL+directory+"logScoresMeanauto.npy.save")

    trueLogSmoothed = UnivariateSpline(t,trueLog,k=2,s=2000)
    reprLogSmoothed = UnivariateSpline(t,reprLog,k=2,s=500)
    endLogSmoothed = UnivariateSpline(t,endLog,k=2,s=1000)
    autoLogSmoothed = UnivariateSpline(t,autoLog,k=3,s=1000)

    plt.plot(trueLogSmoothed(t),label='Rl using true state', linewidth=width)
    plt.plot(reprLogSmoothed(t),label='Rl using representation learned', linewidth=width)
    plt.plot(endLogSmoothed(t),label='Rl end-to-end', linewidth=width)
    plt.plot(autoLogSmoothed(t),label='Rl auto-encoders', linewidth=width)
    plt.plot((0,75), (54, 54), 'k-' , label='Maximum Reward on this task',linewidth=width)


    if plot_std:
        name = 'comparisonWithStd'

        trueStd = np.load(const.LOG_RL+directory+"logAllScorestrue.npy.save").std(axis=0)
        trueLowerBound = trueLog - trueStd
        trueUpperBound = trueLog + trueStd

        reprStd = np.load(const.LOG_RL+directory+"logAllScoresrepr.npy.save").std(axis=0)
        reprLowerBound = reprLog - reprStd
        reprUpperBound = reprLog + reprStd

        endStd = np.load(const.LOG_RL+directory+"logAllScoresend.npy.save").std(axis=0)
        endLowerBound = endLog - endStd
        endUpperBound = endLog + endStd

        autoStd = np.load(const.LOG_RL+directory+"logAllScoresauto.npy.save").std(axis=0)
        autoLowerBound = autoLog - autoStd
        autoUpperBound = autoLog + autoStd

        plt.fill_between(t, trueLowerBound, trueUpperBound, alpha=0.5, facecolor=current_palette[0])
        plt.fill_between(t, reprLowerBound, reprUpperBound, alpha=0.5, facecolor=current_palette[1])
        plt.fill_between(t, endLowerBound, endUpperBound, alpha=0.5, facecolor=current_palette[2])
        plt.fill_between(t, autoLowerBound, autoUpperBound, alpha=0.5, facecolor=current_palette[3])


    plt.xlabel('Number of epochs')
    plt.ylabel('Reward')
    #plt.title("Evolution of scores on looking task")
    plt.legend(loc=4)
    plt.savefig(const.LOG_RL+name)
    plt.show()

    
def main():

    #plot("3Dsave/logScoresMeantrue.npy.save")
    directory = 'save/'
    plotDiffTrainTest(directory)

if __name__ == '__main__':
    main()


