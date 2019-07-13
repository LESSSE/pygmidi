##LESSSE
##10 November 2018
##gmidi
##____________
##Tools to evaluate similarity between variable sized sequences of random variables
##____________

import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_ind, normaltest
from pandas.plotting import table

def gcd(a,b):
    """Compute the greatest common divisor of a and b"""
    while b > 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    return a * b / gcd(a, b)

## Eval Tools
#### Three methods to study variable sized sequences of random variables 
#1) Characterizors: Defining new random variables that characterize important aspects of what we want to study and use the usual estimators to study them. The only hipothesis we use is to supose these random variables have normal distributions 

#2) Regression: Using one alignment criteria we may get a set of points we are able to approximate using regression techniques 

#3) Uniformization: Using some techniques based on some hipothesis we may characterize our sequence using all same sized sequences, these may be studied as simple random vectors

class evaltool:
    def __init__(self,fun,name="metric"):
         self.fun = fun
         self.v = []
         self.name = name
    
    def describe(self):
        return self.table().describe()
    
    def table(self):
        col = self.col
        return pd.DataFrame(self.v,columns=col)
    
    def all_nan(x):
        return np.isnan(x).all()
    
    def add(self,pianoroll,table=False):
        s = len(self.v)
        if not isinstance(pianoroll,list):
            pianoroll = list(pianoroll)
        for p in pianoroll:
            r = self.eval(p)
            self.v += r
        if table:
            col = self.col
            return pd.DataFrame(self.v[s:],columns=col)
        return self.v[s:]
    
    def to_csv(self,path=None):
        self.table().to_csv(path, sep=';',index=False)
        
    def from_csv(self,path=None):
        df = pd.read_csv(path,sep=';')
        self.v = df.values
    
    def correlation(self,path=None):
        if path is None:
            path = self.name.lower().replace(" ","_")+"corr.jpg"
        x_train = self.table()
        names = x_train.columns
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        im = ax.matshow(self.table().corr())
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        fig.colorbar(im,cax=cax, ticks=[-1,-.75,-.5,-.25,0,.25,.5,.75,1])
        fig.savefig(path,dpi=400,bbox_inches='tight')


class characterizors(evaltool):
    
    def __init__(self,fun,name="characterizors"):
        self.col = ["Global", "Seq Size", "Center", "Standard Amplitude", "Min", "25%", "50%", "75%", "Max", "Minimizer", "Maximizer","Var Center", "Var Standard Amplitude", "Var Min", "Var 25%", "Var 50%", "Var 75%", "Var Max", "Var Minimizer", "Var Maximizer"]
        super().__init__(fun,name)
 
    def eval(self,pianoroll,table=False):
        m = self.fun(pianoroll)
        d = m[1:]-m[:-1]
        n = m.shape[0]

        glob = [self.fun(pianoroll.reshape(1,-1,pianoroll.shape[-2],pianoroll.shape[-1]))]
        desc = np.array(pd.DataFrame(m).describe())
        if evaltool.all_nan(m):
            prop = [[np.nan],[np.nan]]
        else:
            prop = [[np.nanargmin(m)/n],[np.nanargmax(m)/n]]
        vardesc = np.array(pd.DataFrame(d).describe())[1:]
        if evaltool.all_nan(d):
            varprop = [[np.nan],[np.nan]]
        else:
            varprop = [[np.nanargmin(d)/n],[np.nanargmax(d)/n]]
        r = [np.concatenate([glob,desc,prop,vardesc,varprop]).reshape(20)]
        if table:
            col = self.col
            return pd.DataFrame(r,columns=col)
        return r
    
    def normality(self,table=False,sort=False):
        """Gives the max confidence on which we can say the distribution is gaussean
        It returns the p-value of a normality test"""
        pop = self.table()
        col = self.col
        pval = []
        for i in pop:
            pval += [normaltest(pop[i],nan_policy='omit')[1]]
        df = pd.DataFrame([pval],columns=col,index=["Normality"]).transpose()
        if sort:
            df = df.sort_values("Normality",ascending=False)
        df = df.transpose()
        if table:
            return df
        return df.values
    
    def similarity(self,pianoroll,table=False,sort=False):
        """Gives the max confidence on which we can say the sample is from the same distirbution as self.v
        It returns the p-value of a t test for mean values"""
        if not isinstance(pianoroll,list):
            pianoroll = [pianoroll,pianoroll]
        elif len(pianoroll)==1:
            pianoroll = [pianoroll[0],pianoroll[0]]
        sample = pianoroll
        pop = self.table()
        for i in range(len(sample)):
            sample = sample[:i]+self.eval(sample[i])+sample[i+1:]
        col = self.col
        sample = pd.DataFrame(sample,columns=col)
        pval = []
        for i in pop:
            pval += [ttest_ind(sample[i],pop[i],nan_policy='omit')[1]]
            
        df = pd.DataFrame([pval],columns=col,index=["Similarity"]).transpose()
        if sort:
            df = df.sort_values("Normality",ascending=False)
        df = df.transpose()
        if table:
            return df
        return df.values
    
    def relevance(self,table=False,sort=False):
        """Gives a measure of relevance of each one of the characterizors based on relative value of std"""
        t = self.table().describe().transpose()
        d = (t["std"]/abs(t["mean"]))
        d = (1/d)
        df = pd.concat([pd.DataFrame(d,columns=["Relevance"]),
                       #self.table().transpose()
                      ],1)
        if sort:
            df = df.sort_values("Relevance",ascending=False)
        df = df.transpose()
        
        if table:
            return df
        return df.values
        
    def correlation(self,path=None):
        if path is None:
            path = self.name.lower().replace(" ","_")+"_corr.jpg"
        x_train = self.table()
        names_rows = x_train.columns
        names_columns = ["G","Size","C","SA","m","25","50","75","M","mr","Mr","VC","VSA","Vm","V25","V50","V75","VM","Vmr","VMr"]
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        im = ax.matshow(x_train.corr())
        ax.set_xticks(np.arange(len(names_columns)))
        ax.set_yticks(np.arange(len(names_rows)))
        ax.set_xticklabels(names_columns)
        ax.set_yticklabels(names_rows)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        fig.colorbar(im,cax=cax, ticks=[-1,-.75,-.5,-.25,0,.25,.5,.75,1])
        fig.savefig(path,dpi=400,bbox_inches='tight')
    
    def present(self,path=None):
        if path is None:
            path = self.name.lower().replace(" ","_")+".jpg"
        fig,ax = plt.subplots()
        plt.title(self.name)
        grid = plt.GridSpec(4, 10, wspace=0.7, hspace=0.4)
        
        df = self.table()
        for i,x in enumerate(df):
            ax = plt.subplot(grid[(i)//10,(i)%10])
            v = df[x].values
            v = np.array(list(filter(lambda x: np.logical_not(np.sum(np.isnan(x),0)),v)))
            ax.hist(v)
            ax.set_title(x)
        
        ax = plt.subplot(grid[2:,0:])
        ax.axis('off')
        #ax.axis('tight')
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        tb = pd.concat([self.describe(),self.normality(True),self.relevance(True)],0)

        t = ax.table(cellText=self.table().describe().values,
                    rowLabels=self.table().describe().index,
                    colLabels=self.table().describe().columns,
                    loc='center')
        t.scale(1.5,1)
        fig.set_figheight(8)
        fig.set_figwidth(40)
        fig.suptitle(self.name, fontsize=16)
        fig.savefig(path,dpi=400,bbox_inches='tight')


class regression(evaltool):
    def __init__(self,fun,name="regression",degree=4):
        super().__init__(fun,name)
        self.col=["X","Y"]
        self.m = 0
        self.degree = degree
    
    def eval(self,pianoroll,table=False):
        v=[]
        r=self.fun(pianoroll)
        for x,y in enumerate(r):
            v += [np.array([x/(len(r)-1),y])]
        v = list(filter(lambda x: np.logical_not(np.sum(np.isnan(x),0)),v))
        if table:
            col = self.col
            return pd.DataFrame(v,columns=col)
        return v
        
    def normality(self,table=False):
        pop = self.table()
        f = self.regression()
        pval = []
        pop["Error"] = pd.Series(None, index=pop.index)
        for index, row in pop.iterrows():
            row["Error"] = row["Y"] - f(row["X"])
        _,p = normaltest(pop["Error"],nan_policy='omit')
        if table:
            return pd.DataFrame([p],columns=["Error"],index=["Normality"])
        return np.array(pval)
    
    def similarity(self,pianoroll,table=False):
        if not isinstance(pianoroll,list):
            pianoroll = [pianoroll]
        sample = pianoroll
        pop = self.table()
        f = self.regression()
        for i in range(len(sample)):
            sample = sample[:i]+self.eval(sample[i])+sample[i+1:]
        sample = pd.DataFrame(sample,columns=self.col)
        sample["Error"] = pd.Series(None, index=sample.index)
        pop["Error"] = pd.Series(None, index=pop.index)
        for index, row in sample.iterrows():
            row["Error"] = row["Y"] - f(row["X"])
        for index, row in pop.iterrows():
            row["Error"] = row["Y"] - f(row["X"])
        _,p = ttest_ind(sample["Error"],pop["Error"],nan_policy='omit')
        if table:
            return pd.DataFrame([p],columns=["Error"],index=["Similarity"])
        return p
      
    def regression(self):
        degree = self.degree
        df=self.table()
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                 ('linear', LinearRegression(fit_intercept=False))])
        model = model.fit(df[self.col[0]][:, np.newaxis], df[self.col[1]])
        coef = model.named_steps['linear'].coef_
        poly = np.poly1d(coef[::-1])
        return poly
    
    def present(self,path=None):
        degree = self.degree
        steps = 2000
        def step(x):
            return (self.table()[self.col[0]].max()-self.table()[self.col[0]].min()) * x/steps + self.table()[self.col[0]].min()
        if path is None:
            path = self.name.lower().replace(" ","_")+".jpg"
        fig,ax = plt.subplots( nrows=1, ncols=1 )
        self.table().plot.scatter(self.col[0],self.col[1],ax=ax)
        f = self.regression()
        for i in range(steps):
            ax.plot([step(i),step(i+1)],[f(step(i)),f(step(i+1))],color="b",linestyle="-")
        plt.title(self.name)
        fig.set_figwidth(15)
        fig.set_figheight(10)
        fig.savefig(path,dpi=400,bbox_inches='tight')

class uniformization(evaltool):
    def __init__(self,fun,name="uniformization"):
         super().__init__(fun,name)
         self.m = 0

    def interpol_signal(s1,l):
         if evaltool.all_nan(s1):
                return np.array(l*[np.nan])
         if len(s1) == l:
             return s1
         m = int(lcm(len(s1)-1,l-1))
         x1 = np.array(range(0,m+1,m//(len(s1)-1)))
         mask1=~np.isnan(s1)
         a1 = np.interp(list(range(m+1)),x1[mask1],s1[mask1])
         return a1[::min(m//(len(s1)-1),m//(l-1))]
    
    def calc_col(self,m):
        return np.around(np.array(list(range(0,m)))/(m-1),3)
        
    def update_m(self,m):
        self.m=m
        self.col = self.calc_col(self.m)
    
    def eval(self,pianoroll,table=False):
        m = max(self.m,pianoroll.shape[0])
        col = self.calc_col(m)
        r = self.fun(pianoroll)
        r = [uniformization.interpol_signal(r,m)]
        if table:
            return pd.DataFrame(r,columns=col)
        return r
    
    def add(self,pianoroll,table=False):
        
        s = len(self.v)
        if not isinstance(pianoroll,list):
            pianoroll = [pianoroll]
        for p in pianoroll:
            if p.shape[0] > self.m:
                self.update_m(p.shape[0])
            r = self.eval(p)
            self.v += r
        for l in range(len(self.v)):
            self.v[l] = uniformization.interpol_signal(self.v[l],self.m)
        if table:
            col = self.col
            return pd.DataFrame(self.v[s:],columns=col)
        return self.v[s:]
    
    def normality(self,table=False):
        pop = self.table()
        col = self.col
        pval = []
        for i in pop:
            pval += [normaltest(pop[i],nan_policy='omit')[1]]
        if table:
            return pd.DataFrame([pval],columns=col,index=["Normality"])
        return np.array(pval)
    
    def similarity(self,pianoroll,table=False):
        if not isinstance(pianoroll,list):
            pianoroll = [pianoroll,pianoroll]
        elif len(pianoroll)==1:
            pianoroll = [pianoroll[0],pianoroll[0]]
        sample = pianoroll
        for i in range(len(sample)):
            sample = sample[:i]+self.eval(sample[i])+sample[i+1:]
        m = max(list(map(len,sample))+[self.m])
        col = self.calc_col(m)
        new_sample = pd.DataFrame(columns=col)
        for row in sample:
            new_sample = new_sample.append(pd.DataFrame([uniformization.interpol_signal(row,m)],columns=col),ignore_index=True)
        
        pop = self.table()
        new_pop = pd.DataFrame(columns=col)
        for index, row in pop.iterrows():
            new_pop = new_pop.append(pd.DataFrame([uniformization.interpol_signal(row.values,m)],columns=col),ignore_index=True)
        
        pop = new_pop
        sample = new_sample
        pval = []
        for i in pop:
            pval += [ttest_ind(sample[i],pop[i],nan_policy='omit')[1]]
        if table:
            return pd.DataFrame([pval],columns=col,index=["Similarity"])
        return np.array(pval)
    
    def present(self,path=None):
        if path is None:
           path = self.name.lower().replace(" ","_")+".jpg"
        fig,ax = plt.subplots( nrows=1, ncols=1 )
        b = np.array(self.table().std())
        y = np.array(self.table().mean())
        t = 1-(b-np.min(b))/np.where(np.max(b)!=0,np.max(b),1)
        b = uniformization.interpol_signal(b,2000)
        t = uniformization.interpol_signal(t,2000)
        y = uniformization.interpol_signal(y,2000)
        x = range(len(t))
        for i in range(len(t)-1):
                ax.plot(x[i:i+2],y[i:i+2],alpha=t[i]**3,color="k",linestyle="-")
        for i in range(len(b)-1):
                ax.plot(x[i:i+2],y[i:i+2]+b[i],alpha=0.1,color="r",linestyle="-")
        for i in range(len(b)-1):
                ax.plot(x[i:i+2],y[i:i+2]-b[i],alpha=0.1,color="r",linestyle="-")
        ax.xaxis.set_visible(False)
        plt.title(self.name)
        fig.set_figwidth(15)
        fig.set_figheight(10)
        fig.savefig(path,dpi=400,bbox_inches='tight')

