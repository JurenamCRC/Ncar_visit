import netCDF4 as nc
import numpy as np
import time as tm
from datetime import datetime

#startTime1 = tm.time()
taux_mean = np.array([])
tauy_mean = np.array([])
tau_mean = np.array([])
sal_mean = np.array([])
for a in range(1993,2019):
    anio = str(a)
    print("year",a)
    for d in range(1,366):
        if d<10:
            day='00'+str(d)    
        elif d <100:
             day='0'+str(d)
        else:
            day = str(d)
        startTime2 = tm.time()
        data=nc.Dataset("/glade/campaign/cgd/oce/people/bachman/ETP_1_20_tides/HOURLY/ocean_hourly__"+anio+"_"+day+".nc")
        print("Processing file /glade/campaign/cgd/oce/people/bachman/ETP_1_20_tides/HOURLY/ocean_hourly__"+anio+"_"+day+".nc")
        
        lon = np.array(data.variables["xh"][:],dtype=np.float64)
        lat = np.array(data.variables["yh"][:],dtype=np.float64)
        time= np.array(data.variables["time"][:],dtype=np.float64)
        taux= np.array(data.variables["taux"][:,:,:],dtype=np.float64)
        tauy= np.array(data.variables["tauy"][:,:,:],dtype=np.float64)
        tau = np.array(data.variables["wind_stress_curl"][:,:,:],dtype=np.float64)
        sal = np.array(data.variables["sss"][:,:,:],dtype=np.float64)
        data.close()
        

        
#        lon = np.array(data.variables["xh"][521:820],dtype=np.float64)
#        lat = np.array(data.variables["yh"][450:550],dtype=np.float64)
#        time= np.array(data.variables["time"][:],dtype=np.float64)
#        taux= np.array(data.variables["taux"][:,450:550,521:821],dtype=np.float64)
#        taux= np.mean(taux, axis=0)
#        tauy= np.array(data.variables["tauy"][:,450:551,521:820],dtype=np.float64)
#        tauy= np.mean(tauy, axis=0)
        
        
        taux= np.mean(taux, axis=0)
        tauy= np.mean(tauy, axis=0)
        tau= np.mean(tau, axis=0)
        sal= np.mean(sal, axis=0)
                
        if ((a ==1993) and (d == 1)):
#            tauy = np.sqrt(np.power(0.5*(taux[:,0:-1]+taux[:,1:]),2) , np.power(0.5*(tauy[0:-1,:]+tauy[1:,:]),2))
            tauxf = taux
            tauyf = tauy
            tauf = tau
            salf = sal
        else:
            tauxf = np.dstack((tauxf,taux))
            tauyf = np.dstack((tauyf,tauy))
            tauf = np.dstack((tauf,tau))
            salf = np.dstack((salf,sal))
            
#            tauf = np.append(tauf,tau)
#            tau1=np.sqrt(np.power(0.5*(taux[:,0:-1]+taux[:,1:]),2) , np.power(0.5*(tauy[0:-1,:]+tauy[1:,:]),2))
            
            
            taux_mean=np.mean(tauxf, axis=2)
            tauy_mean=np.mean(tauyf, axis=2)
            tau_mean=np.mean(tauf, axis=2)
            sal_mean=np.mean(salf, axis=2)
        
        
        
        np.save("/glade/scratch/pmora/tau_mean.npy", [lon,lat,taux_mean,tauy_mean,tau_mean,sal_mean]) 
 #       executionTime = (tm.time() - startTime2)
#        print('Execution time in seconds: ' + str(executionTime))
        with open('/glade/scratch/pmora/log.txt', 'w') as outfile:
            outfile.write('Data collected on:' +str(datetime.now())+"\n")  