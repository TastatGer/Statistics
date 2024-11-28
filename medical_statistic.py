# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:40:39 2023

Aufgabe 1 
aus den Daten den Kaplan-Meier-Schätzer mit Konfidenzintervall in einem
Diagramm darstellen (OS und LRC) und für LRC die Option der inverse
kumulativen Inzidenz mit Kofidenzbändern

Aufgabe 2
den Kofaktor auswählen und den Kaplan-Meier-Schätzer für beide Gruppen
berechnen und in einem Diagramm darstellen. Für LRC die Option der inversen
kumulativen Inziden ermöglichen.
Beim Kaplan-Meier-Schätzer soll ebenfalls der Log-Rang-Test durchgeführt werden


@author: Noah
"""

import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import warnings
import sys

# Suppress SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.core.common.
                      SettingWithCopyWarning)


class KaplanMeier:
    def __init__(self,df,id_col,time,event):
        self.df = df
        self.id_col = id_col
        self.time = time
        self.event = event
        
    """
    the following method calculates the kaplan-meier-estimator. 
    """

###############################################################################
###### Exercise (I) Estimation of a survival curve ############################
###############################################################################


    def calculate_kaplan_meier(self, alpha):
        """
        

        Parameters
        ----------
        alpha : float
            desrcribes the significance intervall.

        Returns
        -------
        km : dataframe
            dataframe with the calculated Kaplan-Meier-Estimator.

        """
        km = self.df[[self.id_col, self.time, self.event]]
        
        #extra line necessary to start the plot at S(t) = 1
        km.loc[-1] = np.zeros(len(km.columns))
        km = km.sort_values(by=[self.time,self.event],
        ascending=[True, False]) 
        km['n'] = len(km)
        
        #loop to gradually reduce n, for every event (0 & 1) in the dataframe
        for i in range(1,len(self.df)+1):
            km.iloc[i,3] = km.iloc[i-1,3] - 1
            
        km['dn'] = 1 - km[self.event] / km['n']
        #Kaplan-Meier-Estimator S(t) is the product of 'dn' for time t0 to t
        km['St'] = np.cumprod(km['dn'])
        #Greenwoodfactor
        km['GF'] = km[self.event] / (km['n'] * (km['n'] - km[self.event]))

        #calculates the sums of the Greenwoodfactor for every line
        for i in range(1, len(self.df)):
            km.iloc[i, 6] = km.iloc[i - 1, 6] + km.iloc[i, 2] /               \
                (km.iloc[i, 3] * (km.iloc[i, 3] - km.iloc[i, 2]))

        #calculates standarddeviation of S(t) with the Greenwoodfactor
        km['StdSt'] = km['St'] * np.sqrt(km['GF'])

        z = sp.norm.ppf(1 - alpha / 2)
                
        km['St+StdSt'] = km['St'] + z * km['StdSt']
        km['St-StdSt'] = km['St'] - z * km['StdSt']
        
        #removes StdSt >1 and <0 as they are illogical
        km['St+StdSt'] = km['St+StdSt'].apply(lambda x: 1 if x > 1 else x)
        km['St-StdSt'] = km['St-StdSt'].apply(lambda x: 0 if x < 0 else x)

        return km
    

    def calculate_cumulative_incidence(self,alpha,time2,event2):
        """
        calculates inverse cummulative incidence for competing risk
        
        This methode calculates for the event of a local recurrence the inverse
        cumulative incidence. The endpoint should not be death for this method.
        Both event-types and times are combined in a new dataframe. First the 
        Kaplan-Meier is calculated for the local recurrence only. 
        Next the cumulative incidence is calculated and the inverse cumulative
        incidence is 1 - cumulative incidence
         

        Parameters
        ----------
        alpha : float
            desrcribes the significance intervall.
        time2 : string
            retrieves the time column for the competing risk.
        event2 : string
            retrieves the event columns for the competing risk.

        Returns
        -------
        Ci : df
            dataframe with calculated inverse cumulative incidence with lower +
            upper limit.

        """
                
        #imports necessary data from the KaplanMeier dataframe
        Ci = self.df[['ID',self.time,self.event,time2,event2]]
        Ci['TCR'] = Ci[[self.time, time2]].min(axis=1)
        
        Ci['delta'] = 0
        Ci.loc[(Ci['LRC'] == 1), 'delta'] = 1
        Ci.loc[(Ci['LRC'] == 0) & (Ci['OS'] == 1), 'delta'] = 2
        
        Ci = Ci.sort_values(by=['TCR','delta'],ascending=[True,False])
        
        Ci['di'] = Ci['delta'].apply(lambda x: 1 if x > 0 else 0)
        
        #number of patients at the beginning of a time intervall
        Ci['n'] = np.linspace(len(Ci),1,len(Ci))
        
        #calculates Kaplan-Meier-Estimator for self.event (LRC)
        Ci['dn'] = 1 - Ci['di'] / Ci['n']
        Ci['SRD'] = np.cumprod(Ci['dn'])
        
        #cumulative incidenc for the probability of a local recurrence
        Ci['CIR'] = 0.0 + 1.0 * Ci['LRC'] / Ci['n']
        for i in range(1,len(Ci)):
        #indices:[i,11] = 'CIR', [i,10] = 'SRD', [i,8] = 'n', [i,2] =self.event
            Ci.iloc[i,11] = Ci.iloc[i-1,11] + Ci.iloc[i-1,10] *               \
            Ci.iloc[i,2] / Ci.iloc[i,8]
            
        #inverse cumulative incidence function with competing risk
        Ci['IC'] = 1 - Ci['CIR']
                
        #pointwise variance via the delta method
        Ci['Var'] = 0
        for i in range(1,len(Ci)):
        #indices: 13 = Var, 11 = CIR, 10 = SRD, 8 = n, 7 = di, 2 = self.event
            Ci.iloc[i,13] = Ci.iloc[i-1,13]+((Ci.iloc[i,11]-
            Ci.iloc[i-1,11])**2*Ci.iloc[i,7]/(Ci.iloc[i,8]*(Ci.iloc[i,8]-
            Ci.iloc[i,7]))+Ci.iloc[i-2,10]**2*(Ci.iloc[i,8]-Ci.iloc[i,2])*
            Ci.iloc[i,2]/Ci.iloc[i,8]**3-2*(Ci.iloc[i,11]-Ci.iloc[i-1,11])*
            Ci.iloc[i-2,10]*Ci.iloc[i,2]/Ci.iloc[i,8]**2)
                    
        
        z = sp.norm.ppf(1 - alpha / 2)
        
        #calculates the confidence intervalls
        Ci['lower'] = 1-(Ci['CIR']+z*np.sqrt(Ci['Var']))
        Ci['upper'] = 1-(Ci['CIR']-z*np.sqrt(Ci['Var']))
        
        #if the last paient has an event it usually results in a division by 0 
        #which is calculateda as inf which leads to an upper/lower confidence
        #intervall of 1 and 0 which is an ugly line from top to bottom right in
        #the middle of the plot. 
        Ci['upper'] = Ci['upper'].apply(lambda x: 0 if x == abs(np.inf) else x)
        Ci['lower'] = Ci['lower'].apply(lambda x: 0 if x == abs(np.inf) else x)
                
        #filters confidence intervalls for values >1 and <0
        Ci['upper'] = Ci['upper'].apply(lambda x: 1 if x > 1 else x)
        Ci['lower'] = Ci['lower'].apply(lambda x: 0 if x < 0 else x)
        
        return Ci
                    
        
###############################################################################
###### Exercise (II) comparison of 2 curves ###################################
###############################################################################

    def compare_two_curves(self,cof,limit,alpha):
        """
        splits the imported data in 2 groups and calculates log-rank-test
        
        This methode is for comparing the subsets of Data that are separated by
        a cofactor. The imported data has to be transfered as one Dataframe to
        the class Kaplan-Meier, with column names that define the observed 
        events and time.
        
        First the methode separates the original Dataframe in 2 different
        Dataframes (x,y) and filters the events based on the defined 
        cofacotr with the lambda function to NaN.
        On unique times, events and patients are appended to an array and
        the p-value of the log-rank-test is calculated
        
        Parameters
        ----------
        cof : sring
            retrieves the desired cofactor from the imported Dataframe.
        limit : float
            cut off to split the patients into 2 cohorts.
        alpha : float
            desrcribes the significance intervall for the test.

        Returns
        -------
        z : float
            test variable for the log-rank-test.
        p : float
            probability that the null hyptothesis is true.
        h0 : string
            describes wether the null hypothesis is most likely true or not.
        x : df
            dataframe with the patientdata above the limit of the cofactor.
        y : df
            dataframe with the patientdata below the limit of the cofactor.

        """
        x = self.df[[self.id_col,self.time,self.event,cof]]
        #removes events from patients that are above the cofactor-limit
        x[cof] = x[cof].apply(lambda x: 1 if x > limit else 0)
        x.loc[x[cof] == 0, self.event] = np.nan
        
        x['n1'] = len(x) - pd.isna(x[self.event]).sum()
        x = x.sort_values(by=[self.time,self.event],ascending=[True, False]) 
       
        #loop to calculate patiants alive at beginning of a time interval
        for i in range(1,len(x)):
            x.iloc[i,4] = x.iloc[i,4]-(len(x[:i])-pd.isna(x.iloc[:i,2]).sum())


        y = self.df[[self.id_col,self.time,self.event,cof]]
        
        y[cof] = y[cof].apply(lambda x: 1 if x <= limit else 0)
        y.loc[y[cof] == 0, self.event] = np.nan
        
        y['n2'] = len(y) - pd.isna(y[self.event]).sum()
        y = y.sort_values(by=[self.time,self.event], ascending=[True, False])

        for i in range(1,len(y)):
            y.iloc[i,4]= y.iloc[i,4] - len(y[:i]) + pd.isna(y.iloc[:i,2]).sum()
            
                
        time = np.array(x[self.time])
        #creates an array with time, events and patients in that order for 
        #Group1 but with non unique times
        d0 = np.reshape(np.array(x[self.event]),(len(x),1))
        d0 = np.hstack((np.reshape(time,(len(time),1)),d0)) 
        #d0[:,0] = time all     d0[:,1] = OS1
        
        d0 = np.hstack((d0,np.reshape(np.array(x['n1']),(len(x),1))))
        #d0[:,2] = n1
        
        append = np.zeros((1,3))
        d1 = np.empty((0,3))           
        #adds every value row that has an unique time value to array d1
        for i in range(0,len(d0)):
            if d0[i,0] != d0[i-1,0]:
                append[0,0] = d0[i,0]  #fügt korrekte Zeit an
                 
                z0 = 0   #laufvariable
                while d0[i,0] == d0[i+z0,0]: #Events (nur "1") für t von d0
                    z0=z0+1
                    append[0,1] = np.sum(d0[i:i+z0,1]>0)
                    if i+z0 > len(d0)-1:
                        break
                append[0,2] = d0[i,2]  #fügt n1 am Anfang von t ein
                d1 = np.vstack((d1,append))    #Werte für t,d,n anhängen
                
        d0[:,1] = y[self.event]
        d0[:,2] = y['n2']
        
        d2 = np.empty((0,3))
        for i in range(0,len(d0)):
            if d0[i,0] != d0[i-1,0]:
                append[0,0] = d0[i,0]  #fügt korrekte Zeit an
                 
                z0 = 0   #laufvariable
                while d0[i,0] == d0[i+z0,0]: #Events (nur "1") für t von d0
                    z0=z0+1
                    append[0,1] = np.sum(d0[i:i+z0,1]>0)
                    if i+z0 > len(d0)-1:
                        break
                append[0,2] = d0[i,2]  #fügt n1 am Anfang von t ein
                d2 = np.vstack((d2,append))    #Werte für t,d,n anhängen
        
        log_rank = pd.DataFrame(d1,columns=["Time","d1","n1"])
        log_rank['d2'] = d2[:,1]
        log_rank['n2'] = d2[:,2]
        log_rank['dt'] = log_rank['d1'] + log_rank['d2']
        log_rank['nt'] = log_rank['n1'] + log_rank['n2']
        
        log_rank['f'] = log_rank['dt'] / log_rank['nt']
        log_rank['e1'] = log_rank['n1'] * log_rank['f']
        log_rank['UL'] = log_rank['d1'] - log_rank['e1']
        
        log_rank['s'] = log_rank['n1'] * log_rank['n2'] * log_rank['dt'] *    \
          (log_rank['nt'] - log_rank['dt']) /                                 \
          ((log_rank['nt']**2)*(log_rank['nt']-1))
        
        ul = np.sum(log_rank['UL'])
        s = np.sum(log_rank['s'])   
        
        z = (abs(ul - 0.5)) / np.sqrt(s)    #Test variable (Norm-distributed)
        
        if z < 0:
            p = 2*sp.norm.cdf(z)
        else:
            p = (1-sp.norm.cdf(z))*2        #p-Value
        
        if p <= alpha:                  #accept/reject Nullhypothesis
            h0 = ("die Nullhypothese, dass beide Kurven gleich sind, " 
                  "wird abgelehnt")
        else:
            h0 = ("die Nullhypothese, dass beide Kurven gleich sind, "
                 "wird angenommen")
        
        #prepares x and y for later use to plot 
        x = x.dropna(subset=[self.event])
        del x[cof]
        del x['n1']
        
        y = y.dropna(subset=[self.event])
        del y[cof]
        del y['n2']


        return z,p,h0,x,y
    
###############################################################################
##### plotting for exercise (I) and (II) ######################################
###############################################################################
    
    
    def plot_kaplan_meier(self, km, f, f1, f2, label, color):
        """
        plots the step-functions for 1 given dataset

        Parameters
        ----------
        km : TYPE
            DESCRIPTION.
        f : strin
            x-value for the plot, can be estimator for KM or IC.
        f1 : string
            lower confidence intervall for the x-value.
        f2 : string
            upper condifence intervall for the x-value.
        label : string
            string to add title .
        color : string
            plots the data in a given colour, has to be from plot colourtable.

        Returns
        -------
        None.

        """
        
        #kaplan-meier estimator
        plt.step(km[self.time], km[f], where='post', label=label) 
        
        #fills confidence interval between upper and lower limit
        plt.fill_between(km[self.time], km[f2], km[f1],
        color=color, alpha=0.3,step='post')
        
        #Scatterplot for scensored events
        censors = km[km[self.event] == 0].index  
        plt.scatter(km.loc[censors, self.time], km.loc[censors, f],
        marker = '+', alpha=0.8)
        
        plt.xlabel('Time in months')
        plt.xlim((0,80))
        plt.ylim((0,1.05))
        plt.ylabel('Survival Probability')
                
###############################################################################        
###############################################################################
###############################################################################
   
def main():
        
    df = pd.read_excel('Beleg_Biostatistik_2023_Daten.xlsx')
    
    print("Programm zur Berechnung von Überlebenszeiten")
    print("Wählen Sie eine der beiden Optionen:")
    print("1. Schätzen einer Überlebenskurve")
    print("2. Vergleichen von zwei Überlebenskurven")
    
    option = input("Geben Sie ihre Wahl ein (1/2): ")
    
    event = input("Geben Sie die Eventspalte an (OS/LRC): ")
    time = input("Geben Sie die Zeitspalte an (OStime/LRCtime): ")
    alpha = float(input("Geben Sie das gewünschte Signifikanzniveau ein "
                        "(Bp. 0.05): "))

    # alpha = 0.05
    # time = 'LRCtime'
    # event = 'LRC'    
    id_col = 'ID'
    km = KaplanMeier(df, id_col, time, event)
    
######### universally used varibales to plot ##################################    

#these are variables universally used for plotting kaplan-meier plots, they are
#given to the plotting method. In case the cumulative incidence is plotted they
#are overwritten in the according cases (eg. y=IC, y1=upper,y2=lower,...)
    color1 = 'blue'
    f = 'St'
    f1 = 'St-StdSt'
    f2 = 'St+StdSt'
    label = 'Kaplan-Meier-Schätzer'
    
    
#### if loops for estimating 1 survival curve #################################    
        
    if option == "1":
        print("""Wähle eine Rechenmethode:
              1. Kaplan-Meier-Schätzer
              2. kumulative Inzidenz""")
        method = input("Geben Sie ihre Wahl ein (1/2): ")
        
        
        if method == "1":   #calls method to calculate kaplan-meier estimator
            km_results = km.calculate_kaplan_meier(alpha)
            km.plot_kaplan_meier(km_results,f,f1,f2,label,color1)
            plt.title("Kaplan-Meier-Schätzer")
            return km_results
            
        if method == "2":   #calls method to calculate cumulative incidence
            if event == 'OS':
                print("kumulative Inzidenz kann nicht für",'\n',
                      "Gesamtüberleben berechnet werden")
            else:
                time2 = 'OStime'
                event2 = 'OS'
                f = 'IC'
                f1 = 'lower'
                f2 = 'upper'
                label = 'kumulative Inzidenz'
                
                ki = KaplanMeier(df, id_col, time, event)
                ki_results = ki.calculate_cumulative_incidence(alpha,time2,event2)
                ki.plot_kaplan_meier(ki_results,f,f1,f2,label,color1)
                plt.title("kumulative Inzidenzfunktion")
                return ki_results
                
#### if loops for comparing 2 curves ##########################################     
                       
    elif option == "2":
        
        color2 = 'red'
        # cof = input("Select observed Cofactor (Age/GTV/TBR/HV): ")
        # limit = float(input("Select classification for cofactor without " '\n'
        #           "units (eg. for Age: 60): "))

        cof = input("Geben Sie den betrachteten Kofaktor ein"
                    "(Age/GTV/TBR/HV): ")
        limit = float(input("Geben Sie den Wert für die Gruppeneinteilung"'\n'
                            "ohne Einheit ein (Bp Alter: 60): "))
        print("Wählen Sie eine Methode: " '\n'
              "1.Kaplan-Meier-Schätzer" '\n'
              "2.kumulative Inzidenz")
        method = input("Geben Sie ihre Wahl ein (1/2): ")
        
        
        if method == "1":
            
            z,p,ho,x,y = km.compare_two_curves(cof, limit, alpha)
            
            label = cof + " > " + str(limit)
            x = KaplanMeier(x,id_col,time,event)
            x_results = x.calculate_kaplan_meier(alpha)
            x.plot_kaplan_meier(x_results,f,f1,f2,label,color1)
            
            label = cof + " <= " + str(limit)
            y = KaplanMeier(y,id_col,time,event)
            y_results = y.calculate_kaplan_meier(alpha)
            y.plot_kaplan_meier(y_results,f,f1,f2,label,color2)
            plt.title("Kaplan-Meier-Schätzer")
            plt.legend()
            print("Testvariable: ",z,'\n',"p-Wert: ",p,'\n',ho)
            return x_results, y_results
            
        if method == "2":
            if time == 'OS':
                print("""kumulative Inzidenz kann nicht für
                      "Gesamtüberleben berechnet werden""")
            else:
                time2 = 'OStime'
                event2 = 'OS'
                f = 'IC'
                f1 = 'lower'
                f2 = 'upper'
                
                label = cof + " <= " + str(limit)
                x = df.loc[df[cof] <= limit]
                d1 = KaplanMeier(x,id_col,time,event)
                d1_results = d1.calculate_cumulative_incidence(alpha, time2, event2)
                d1.plot_kaplan_meier(d1_results, f, f1, f2, label, color1)
                
                label = cof + " > " + str(limit)
                y = df.loc[df[cof] >limit]
                d2 = KaplanMeier(y, id_col, time, event)
                d2_results = d2.calculate_cumulative_incidence(alpha, time2, event2)
                d2.plot_kaplan_meier(d2_results, f, f1, f2, label, color2)
                plt.title("kumulative Inzidenz")
                plt.legend()
                return d1_results, d2_results
                
    else:   #Error Massage if entered wrong option
        print("Ungültige Eingabe. Beende Programm")
        sys.exit()

if __name__ == "__main__":
#in the case of comparing 2 curves the first dataframe in the tuple returned 
#is cof > limit
    results = main()