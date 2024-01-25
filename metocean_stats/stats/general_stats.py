import numpy as np
import calendar 
import math
import shutil
import os 
import windrose
import dataframe_image as dfi 
from matplotlib import pyplot as plt
import pandas as pd 
import scipy.ndimage as nd
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import pickle 
import scipy.io as io
import scipy.stats as stats
from pyextremes import get_extremes
from scipy.stats import expon
from scipy.io import savemat
from datetime import datetime, timedelta
from matplotlib.font_manager import FontProperties

## get index of location, and find the nearest
def getIndexes(infile,lon0,lat0): 
    nc = Dataset(infile)
    lon = nc.variables['lon_rho'][:]
    lat = nc.variables['lat_rho'][:]
    y0, x0 = find_position(lon0, lat0, lon, lat) # indexes 
    #print (y0,x0,lon[y0,x0],lat[y0,x0]) # 197 388 3.847956715019585 61.538507642909515
    print ('lon,lat from model (...): ',round(lon[y0,x0],2),round(lat[y0,x0],2)) # 197 388 3.847956715019585 61.538507642909515
    return y0, x0 
    
def getIndexes_newNK800(infile,lon0,lat0): 
    nc = Dataset(infile)
    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]
    y0, x0 = find_position(lon0, lat0, lon, lat) # indexes 
    #print (y0,x0,lon[y0,x0],lat[y0,x0]) # 197 388 3.847956715019585 61.538507642909515
    print ('lon,lat from model (...): ',round(lon[y0,x0],2),round(lat[y0,x0],2)) # 197 388 3.847956715019585 61.538507642909515
    return y0, x0 
    
### FIGURE 4     
#########################################################################################
#########################################################################################
#########################################################################################

def color_max_white(val, max_val):
    color = 'white' if val == max_val else 'black'
    return 'color: %s' % color

def highlight_max(data, color='white'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)
   
def heatmap(var1, step_var1, var2, step_var2, output_file):     
   
    # this function make joint distribution table of two variable with known bins/steps in percentage
    # it also calculate sum in coresponding columns/rows 
    # it then save the table to a file 
    # it call two other functions color_max_white and highlight_max, so be careful when copying 
    # var1: variable 1
    # var2: variable 2
    # step_var1: interval of var1 
    # step_var2: interval of var2
    # output_file: save to a file. *.png works

    # create bins for each variables 
    bins_var1 = np.arange(math.floor(np.min(var1)),math.ceil(np.max(var1))+step_var1,step_var1) # one cell more 
    bins_var2 = np.arange(math.floor(np.min(var2)),math.ceil(np.max(var2))+step_var2,step_var2) # one cell at beginning and ending 
    tbl = np.zeros([len(bins_var1)-1,len(bins_var2)-1])
    row_var1, col_var2 = tbl.shape
    
    # calculate values for the table
    N = len(var1)
    for i in range(N):
        for row in range(row_var1) : 
            for col in range(col_var2) : 
                if (bins_var1[row] <= var1[i] < bins_var1[row+1]) and (bins_var2[col] <= var2[i] < bins_var2[col+1]):
                    tbl[row,col] += 1 # cout the number of occurence in each cell of the table 
    
    tbl = tbl/N*100 # convert occurence to percentage        
                    
    sumcols = np.sum(tbl, axis=0)
    sumrows = np.sum(tbl, axis=1)
    
    sumrows = np.around(sumrows, decimals=1) # calculate the sume in each row
    sumcols = np.around(sumcols, decimals=1) # calculate the sume in each column
    
    
    ## make heading for rows and columns for the table 
    rows = []
    for i in range(len(bins_var1)-1):
        
        # take one more space for above 10m 
        if bins_var1[-1] >= 10 : 
            rows.append('%04.1f' % bins_var1[i]+'-'+'%04.1f' % bins_var1[i+1]+' | '+'%04.1f' % sumrows[i]+'%')
        else :
            rows.append('%03.1f' % bins_var1[i]+'-'+'%03.1f' % bins_var1[i+1]+' | '+'%04.1f' % sumrows[i]+'%')
    
    rows = rows[::-1] # resever, make them upward 
    tbl = tbl[::-1,:] # resever, make them upward 
        
    cols = []
    for i in range(len(bins_var2)-1):
        cols.append(str(int(bins_var2[i]))+'-'+str(int(bins_var2[i+1]))+'<br>-------<br>'+str(sumcols[i])+'%')
        
    
    # Erase 0 in the table, and highlight the table 
    # Replace 0 by nan 
    for i in range(tbl.shape[0]): 
        for j in range(tbl.shape[1]):
            if tbl[i,j]==0 : 
                tbl[i,j]=np.nan
    
    # Assign to dataframe 
    dfout = pd.DataFrame()
    dfout.index = rows 
    for i in range(len(cols)) : 
        dfout[cols[i]] = tbl[:,i] 
    
    # highlight the table 
    dfout.fillna(dfout.max().max()+1, inplace=True)
    max_val = dfout.max().max()
    hi = dfout.style.format("{:.1f}").background_gradient(cmap='Blues', axis=None).applymap(lambda x: color_max_white(x, max_val)).apply(highlight_max, axis=None)
    
    # save to file 
    dfi.export(hi,output_file)
    
    return hi 





def directional_heatmap(intensity, step_intensity, direction, step_direction, output_file) : 

    # this function make joint distribution table of a variable with intensity and direction with known interval in percentage
    # it also calculate sum in coresponding columns/rows 
    # it then save the table to a file 
    # it call two other functions color_max_white and highlight_max, so be careful when copying 
    # intensity 
    # direction 
    # step_intensity: interval of intensity 
    # step_direction: interval of direction
    # output_file: save to a file. *.png works

    # # create bins for each variables 
    half_step_direction = step_direction/2 
    bins_intensity = np.arange(math.floor(np.min(intensity)),math.ceil(np.max(intensity))+step_intensity,step_intensity) # one cell more 
    bins_direction = np.arange(0, 360 + step_direction, step_direction) # 0,30,...,300,330,360 
    tbl = np.zeros([len(bins_intensity)-1,len(bins_direction)-1])
    row_intensity, col_direction = tbl.shape
    
    # calculate values for the table
    N = len(intensity)
    for i in range(N):
        for row in range(row_intensity) : 
            if bins_intensity[row] <= intensity[i] < bins_intensity[row+1] : 
                if 360-half_step_direction <= direction[i] :
                    tbl[row,0] += 1 
                else : 
                    for col in range(col_direction) : 
                        if bins_direction[col] - half_step_direction <= direction[i] < bins_direction[col] + half_step_direction :
                            tbl[row,col] += 1 # cout the number of occurence in each cell of the table 
                    
    tbl = tbl/N*100 # convert occurence to percentage  
                    
    sumcols = np.sum(tbl, axis=0) # calculate the sume in each column
    sumrows = np.sum(tbl, axis=1) # calculate the sume in each row
    sumrows = np.around(sumrows, decimals=1)
    sumcols = np.around(sumcols, decimals=1)
    
    ## ## make heading for rows and columns for the table 
    rows = []
    for i in range(row_intensity):
        if bins_intensity[-1] >= 10 : 
            rows.append('%04.1f' % bins_intensity[i]+'-'+'%04.1f' % bins_intensity[i+1]+' | '+'%04.1f' % sumrows[i]+'%')
        else : 
            rows.append('%03.1f' % bins_intensity[i]+'-'+'%03.1f' % bins_intensity[i+1]+' | '+'%04.1f' % sumrows[i]+'%')
    
    rows = rows[::-1] # resever, make them upward 
    tbl = tbl[::-1,:] # resever, make them upward 
        
    cols = []
    for i in range(col_direction):
        if i == 0 : 
            cols.append(str(int(360-half_step_direction))+'-'+str(int(half_step_direction))+'<br>-------<br>'+str(sumcols[i])+'%')
        else : 
            cols.append(str(int(bins_direction[i]-half_step_direction))+'-'+str(int(bins_direction[i]+half_step_direction))+'<br>-------<br>'+str(sumcols[i])+'%')
        
        
    # Replace 0 by nan 
    for i in range(tbl.shape[0]): 
        for j in range(tbl.shape[1]):
            if tbl[i,j]==0 : 
                tbl[i,j]=np.nan
    
    # Assign to dataframe 
    dfout = pd.DataFrame()
    dfout.index = rows 
    for i in range(len(cols)) : 
        dfout[cols[i]] = tbl[:,i] 
    
    # highlight the table 
    dfout.fillna(dfout.max().max()+1, inplace=True)
    max_val = dfout.max().max()
    hi = dfout.style.format("{:.1f}").background_gradient(cmap='Blues', axis=None).applymap(lambda x: color_max_white(x, max_val)).apply(highlight_max, axis=None)
    
    # save to file 
    dfi.export(hi,output_file)
    
    return hi
    
    
    
    
    
def directional_annual_MAX_statistics(direction, intensity, output_file) : 
    
    # this function calculate min, mean, max in each direction/sector based on annual max
    # it also calculate omni values 
    # output_file: save to a file. *.tex or *.png works
    # by default directional bin is 30 degree 

    time = intensity.index  
    
    # sorted by sectors/directions, keep time for the next part 
    bins_dir = np.arange(0,360,30) # 0,30,...,300,330
    dic_Hs = {}
    dic_time = {}
    for i in range(len(bins_dir)) : 
        dic_Hs[str(int(bins_dir[i]))] = [] 
        dic_time[str(int(bins_dir[i]))] = [] 
    
    for i in range(len(intensity)): 
        if 345 <= direction[i] :
            dic_time[str(int(bins_dir[0]))].append(time[i])
            dic_Hs[str(int(bins_dir[0]))].append(intensity[i]) 
        else: 
            for j in range(len(bins_dir)): 
                if bins_dir[j]-15 <= direction[i] < bins_dir[j] + 15 : # -15 --> +345 
                    dic_time[str(int(bins_dir[j]))].append(time[i])
                    dic_Hs[str(int(bins_dir[j]))].append(intensity[i]) 
    
    # make statistical table 
    tbl = np.zeros([len(bins_dir)+1,3]) # 12 directions + omni, 3: min, mean, max 
    rows = []
    # monthly 
    for j in range(len(bins_dir)):
        df_dir = pd.DataFrame()
        df_dir.index = dic_time[str(int(bins_dir[j]))]
        df_dir['Hs'] = dic_Hs[str(int(bins_dir[j]))]
        annual_max_dir = df_dir.resample('Y').max() # get annual values 
        mind = round(annual_max_dir.min()['Hs'],1)  # get min of annual values 
        meand = round(annual_max_dir.mean()['Hs'],1) # get mean of annual values 
        maxd = round(annual_max_dir.max()['Hs'],1) # get max of annual values 
        tbl[j,0] = mind
        tbl[j,1] = meand
        tbl[j,2] = maxd
        
        # create rows for data frame 
        start = bins_dir[j] - 15
        if start < 0 : 
            start = 345 
        rows.append(str(start) + '-' + str(bins_dir[j]+15))
        
    # omni
    annual_max = intensity.resample('Y').max()
    mind = round(annual_max.min(),1)
    meand = round(annual_max.mean(),1)
    maxd = round(annual_max.max(),1)
    tbl[-1,0] = mind
    tbl[-1,1] = meand
    tbl[-1,2] = maxd
    rows.append('Omni / annual')

    # write to file 
    if output_file[-4:] == '.png' : 
        
        dfout = pd.DataFrame()
        #dfout['Direction'] = rows
        dfout.index = rows
        dfout['Minimum'] = tbl[:,0]
        dfout['Mean'] = tbl[:,1]
        dfout['Maximum'] = tbl[:,2]

        dfout['Minimum'] = dfout['Minimum'].map('{:,.1f}'.format)
        dfout['Mean'] = dfout['Mean'].map('{:,.1f}'.format)
        dfout['Maximum'] = dfout['Maximum'].map('{:,.1f}'.format)
        dfi.export(dfout.style.hide(axis='index'),output_file)
        return dfout
        
    elif output_file[-4:] == '.tex' :
        
        with open(output_file, 'w') as f :
            f.write('\\begin{tabular}{l | c c c }' + '\n')
            f.write('Direction & Minimum & Mean & Maximum \\\\' + '\n')
            f.write('\hline' + '\n')
            
            # sorted by years, get max in each year, and statistical values 
            for j in range(len(bins_dir)):   
                f.write(rows[j] + ' & ' + str(tbl[j,0]) + ' & ' + str(tbl[j,1]) + ' & ' + str(tbl[j,2]) + ' \\\\' + '\n')
                
            ## annual row 
            f.write(rows[-1] + ' & ' + str(tbl[-1,0]) + ' & ' + str(tbl[-1,1]) + ' & ' + str(tbl[-1,2]) + ' \\\\' + '\n')
            f.write('\hline' + '\n')
            f.write('\end{tabular}' + '\n')
        return
        
    else :
        
        print ('please check output file *.tex or *.png')
        return





def monthly_MAX_statistics(var,output_file) : # of max 

    # this function calculate min, mean, max in each month 
    # it also calculate omni values 
    # output_file: save to a file. *.tex or *.png works
    
    # get month names 
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    # calculate monthly values 
    monthly_max = var.resample('M').max() # max in every month, all months 
    minimum = monthly_max.groupby(monthly_max.index.month).min() # min, sort by month
    mean = monthly_max.groupby(monthly_max.index.month).mean() # mean, sort by month
    maximum = monthly_max.groupby(monthly_max.index.month).max() # max, sort by month
    
    ## annual values 
    annual_max = var.resample('Y').max()
    min_year = annual_max.min()
    mean_year = annual_max.mean()
    max_year = annual_max.max()
    
    # write to file 
    if output_file[-4:] == '.png' : 
        
        # add monthly
        dfout = pd.DataFrame()
        dfout.index = months
        dfout['Minimum'] = np.round(minimum.values,1)
        dfout['Mean'] = np.round(mean.values,1)
        dfout['Maximum'] = np.round(maximum.values,1)
        
        # add annual
        dfout.loc['Omni / annual'] = [np.round(min_year,1), np.round(mean_year,1), np.round(max_year,1)] 
        
        # save to file 
        dfi.export(dfout,output_file)
        
        return dfout
        
    elif output_file[-4:] == '.tex' :
        
        # write to file 
        with open(output_file, 'w') as f :
            f.write('\\begin{tabular}{l | c c c }' + '\n')
            f.write('Month & Minimum & Mean & Maximum \\\\' + '\n')
            f.write('\hline' + '\n')
            
            # write monthly values to the table
            for i in range(len(months)):
                f.write(months[i] + ' & ' + str(round(minimum.values[i],1)) + ' & ' + str(round(mean.values[i],1)) + ' & ' + str(round(maximum.values[i],1)) + ' \\\\' + '\n')
            f.write('Annual & ' + str(round(min_year,1)) + ' & ' + str(round(mean_year,1)) + ' & ' + str(round(max_year,1)) + ' \\\\' + '\n')
            
            f.write('\hline' + '\n')
            f.write('\end{tabular}' + '\n')
        return

    else :
        
        print ('please check output file *.tex or *.png')
        return
    
    
    
    
    
def monthly_percentile(Var,varName,output_file):  

    # this function will sort a variable (Var) by month and calculate percentiles 5, 50, 95, 99 also the mean
    # it then writes to Latex file 
    # Var : panda series 
    # varName : a text for heading of the table. Needed for *.tex file but not *.png file 
    # outputfile : a latex file 
    # Note that this is made for the metocean report report, table 3 and table 9 
    
    # sort data by month 
    # geth month index 
    Var_month = Var.index.month
    M = Var_month.values
    
    # get monthly names 
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    monthlyVar = {}
    for i in range(len(months)) : 
        monthlyVar[months[i]] = [] # create empty dictionaries to store data 
    
    for i in range(len(Var)) : 
        m_idx = int(M[i]-1) 
        monthlyVar[months[m_idx]].append(Var[i])  # fill data 
        
    # # make statistical table 
    tbl = np.zeros([len(months)+1,5]) # 12 directions + omni, 3: min, mean, max 
    rows = []
    # # monthly 
    for j in range(len(months)):
        Var_P5 = round(np.percentile(monthlyVar[months[j]],5),1)
        Var_P50 = round(np.percentile(monthlyVar[months[j]],50),1)
        Var_mean = round(np.mean(monthlyVar[months[j]]),1)
        Var_P95 = round(np.percentile(monthlyVar[months[j]],95),1)
        Var_P99 = round(np.percentile(monthlyVar[months[j]],99),1)
        tbl[j,0] = Var_P5
        tbl[j,1] = Var_P50
        tbl[j,2] = Var_mean
        tbl[j,3] = Var_P95
        tbl[j,4] = Var_P99
        rows.append(months[j])
               
    # # omni
    Var_P5 = round(np.percentile(Var,5),1)
    Var_P50 = round(np.percentile(Var,50),1)
    Var_mean = round(np.mean(Var),1)
    Var_P95 = round(np.percentile(Var,95),1)
    Var_P99 = round(np.percentile(Var,99),1)
    tbl[-1,0] = Var_P5
    tbl[-1,1] = Var_P50
    tbl[-1,2] = Var_mean
    tbl[-1,3] = Var_P95
    tbl[-1,4] = Var_P99
    rows.append('Omni / annual')

    # write to file 
    if output_file[-4:] == '.png' : 
        
        dfout = pd.DataFrame()
        dfout.index = rows
        dfout['5%'] = tbl[:,0]
        dfout['50%'] = tbl[:,1]
        dfout['Mean'] = tbl[:,2]
        dfout['95%'] = tbl[:,3]
        dfout['99%'] = tbl[:,4]

        dfi.export(dfout,output_file)
        return dfout 

    elif output_file[-4:] == '.tex' :
        # Write to table/ file 
        with open(output_file, 'w') as f:
            f.write('\\begin{tabular}{l | p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
            f.write(' & \multicolumn{5}{c}{' + varName + '} \\\\' + '\n')
            f.write('Month & 5\% & 50\% & Mean & 95\% & 99\% \\\\' + '\n')
            f.write('\hline' + '\n')
            
            # monthly row 
            for j in range(len(months)) : 
                f.write(rows[j] + ' & '+str(round(tbl[j,0],1))+' & '+str(round(tbl[j,1],1))+' & '+str(round(tbl[j,2],1))+' & '+str(round(tbl[j,3],1))+' & '+str(round(tbl[j,4],1))+' \\\\' + '\n')
            
            ## annual row 
            j = -1
            f.write(rows[j] + ' & '+str(round(tbl[j,0],1))+' & '+str(round(tbl[j,1],1))+' & '+str(round(tbl[j,2],1))+' & '+str(round(tbl[j,3],1))+' & '+str(round(tbl[j,4],1))+' \\\\' + '\n')
        
            f.write('\hline' + '\n')
            f.write('\end{tabular}' + '\n')
        return
        
    else :
        
        print ('please check output file *.tex or *.png')
        
        return  




  
def var_sorted_by_Hs(Hs,Var,varName,output_file): ### TABLE 2 and 10 

    # this will sort variable Var by 1m wave interval
    # then calculate min, percentile 5, mean, percentile 95 and max of variable Var in each bin 
    # write out a table in latex format (table 2 and 10 in metocean reports)
    
    # Hs bins 
    binsHs = np.arange(0.,math.ceil(np.max(Hs))+0.1) # +0.1 to get the last one   
    Var_binsHs = {}
    for j in range(len(binsHs)-1) : 
        Var_binsHs[str(int(binsHs[j]))+' - '+str(int(binsHs[j+1]))] = [] 
    
    # gather values in the same Hs bins
    N = len(Hs)
    for i in range(N):
        for j in range(len(binsHs)-1) : 
            if binsHs[j] <= Hs[i] < binsHs[j+1] : 
                Var_binsHs[str(int(binsHs[j]))+' - '+str(int(binsHs[j+1]))].append(Var[i])

    # # make statistical table 
    tbl = np.zeros([len(binsHs),6]) # len(binsHs)-1+2, exclude Hs-bins  
    rows = []

    for j in range(len(binsHs)-1) : 
        key_bin_hs = str(int(binsHs[j]))+' - '+str(int(binsHs[j+1]))
        rows.append(key_bin_hs)
        Var_binsHs_temp = Var_binsHs[key_bin_hs]
        
        entries = len(Var_binsHs_temp)
        Var_min = round(np.min(Var_binsHs_temp), 1)
        Var_P5 = round(np.percentile(Var_binsHs_temp , 5), 1)
        Var_mean = round(np.mean(Var_binsHs_temp), 1)
        Var_P95 = round(np.percentile(Var_binsHs_temp, 95), 1)
        Var_max = round(np.max(Var_binsHs_temp), 1)
        tbl[j,0] = entries
        tbl[j,1] = Var_min
        tbl[j,2] = Var_P5
        tbl[j,3] = Var_mean
        tbl[j,4] = Var_P95
        tbl[j,5] = Var_max

    # final row/ all Hs  
    rows.append('0 - '+str(int(binsHs[-1])))
    entries = len(Var)
    Var_min = round(np.min(Var), 1)
    Var_P5 = round(np.percentile(Var , 5), 1)
    Var_mean = round(np.mean(Var), 1)
    Var_P95 = round(np.percentile(Var, 95), 1)
    Var_max = round(np.max(Var), 1)
    j = -1 
    tbl[j,0] = entries
    tbl[j,1] = Var_min
    tbl[j,2] = Var_P5
    tbl[j,3] = Var_mean
    tbl[j,4] = Var_P95
    tbl[j,5] = Var_max
    
    if output_file[-4:] == '.png' : 
        
        dfout = pd.DataFrame()
        dfout['Hs'] = rows
        dfout['Entries'] = tbl[:,0]
        dfout['Min'] = tbl[:,1]
        dfout['5%'] = tbl[:,2]
        dfout['Mean'] = tbl[:,3]
        dfout['95%'] = tbl[:,4]
        dfout['Max'] = tbl[:,5]
        
        dfout['Entries'] = dfout['Entries'].map('{:,.0f}'.format)
        dfout['Min'] = dfout['Min'].map('{:,.1f}'.format)
        dfout['5%'] = dfout['5%'].map('{:,.1f}'.format)
        dfout['Mean'] = dfout['Mean'].map('{:,.1f}'.format)
        dfout['95%'] = dfout['95%'].map('{:,.1f}'.format)
        dfout['Max'] = dfout['Max'].map('{:,.1f}'.format)
        dfi.export(dfout.style.hide(axis='index'), output_file)
        return dfout
        
    elif output_file[-4:] == '.tex' :
        with open(output_file, 'w') as f:
            f.write('\\begin{tabular}{l p{1.5cm}|p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
            f.write('& & \multicolumn{5}{c}{'+varName+'} \\\\' + '\n')
            f.write('Hs & Entries & Min & 5\% & Mean & 95\% & Max \\\\' + '\n')
            f.write('\hline' + '\n')

            ## calucalte statistical values in each Hs bin    
            for j in range(len(binsHs)-1) : 
                f.write(rows[j] + ' & '+str(int(tbl[j,0]))+' & '+str(round(tbl[j,1],1))+' & '+str(round(tbl[j,2],1))+' & '+str(round(tbl[j,3],1))+' & '+str(round(tbl[j,4],1))+' & '+str(round(tbl[j,5],1))+' \\\\' + '\n')
            hs_bin_temp = str(int(binsHs[-1]))+' - '+str(int(binsHs[-1]+1)) # +1 for one empty row 
            f.write(hs_bin_temp + ' & 0 & - & - & - & - & - \\\\' + '\n')
            
            # annual row 
            hs_bin_temp = str(int(binsHs[0]))+'-'+str(int(binsHs[-1]+1)) # +1 for one empty row 
            j = -1
            f.write('0 - '+str(int(binsHs[-1]+1)) + ' & '+str(int(tbl[j,0]))+' & '+str(round(tbl[j,1],1))+' & '+str(round(tbl[j,2],1))+' & '+str(round(tbl[j,3],1))+' & '+str(round(tbl[j,4],1))+' & '+str(round(tbl[j,5],1))+' \\\\' + '\n')
            f.write('\hline' + '\n')
            f.write('\end{tabular}' + '\n')        
        return 
        
    else :
        
        print ('please check output file *.tex or *.png')
        return 
        
        


### FIGURE 5   
#########################################################################################
#########################################################################################
#########################################################################################

def var_rose(direction,intensity,title,output_file):

    # this plot (wind, waves, ocean currents) rose and save to a file
    # Figure 5 in metocean report 
    # Rembember the direction must be in standard format 
    # format type is array, numpy array works, pandas also works 
    max_ = max(intensity)
    bins_range = np.arange(0,max_*1.05,round(max_*0.25,1)) # this sets the legend scale
    
    fig = plt.figure(figsize = (8,8))
    
    ax = fig.add_subplot(111, projection="windrose")
    ax.bar(direction, intensity, normed=True, bins=bins_range, opening=0.8, nsector=12)
    ax.set_legend()
    size = 5
    ax.set_title(title)
    ax.figure.set_size_inches(size, size)
    
    plt.savefig(output_file,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 





def monthly_var_rose(direction,intensity,output_file1,output_file2) : 

    # This will sort data by month.
    # It then plot (wind, waves, ocean currents) rose and save to two files, following metocean report
    # Rembember the direction must be in standard format 
    # directional and intensity is pandas frames with time index  
       
    # get month from panda series 
    M = intensity.index.month.values
    
    # get month names 
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    # sort them outh by months 
    dic_intensity = {} # dic_intensity
    dic_direction = {} # dic_direction
    for i in range(len(months)) : 
        dic_intensity[months[i]] = [] 
        dic_direction[months[i]] = [] 
        
    for i in range(len(intensity)) : 
        m_idx = int(M[i]-1)
        dic_intensity[months[m_idx]].append(intensity[i])
        dic_direction[months[m_idx]].append(direction[i])
        
        
    ### Figure 6 
    fig = plt.figure(figsize = (8,15))
    #size = 8
    
    ax = fig.add_subplot(321, projection="windrose")
    j = 0
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(322, projection="windrose")
    j = 1
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(323, projection="windrose")
    j = 2
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(324, projection="windrose")
    j = 3
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(325, projection="windrose")
    j = 4
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(326, projection="windrose")
    j = 5
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    plt.savefig(output_file1,dpi=100,facecolor='white',bbox_inches='tight')
    
    
    
    ## figure 7 
    fig = plt.figure(figsize = (8,15))
    #size = 8
    
    ax = fig.add_subplot(321, projection="windrose")
    j = 6
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(322, projection="windrose")
    j = 7
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(323, projection="windrose")
    j = 8
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(324, projection="windrose")
    j = 9
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(325, projection="windrose")
    j = 10
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(326, projection="windrose")
    j = 11
    ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    plt.savefig(output_file2,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 





def Tp_correction(Tp):  ### Tp_correction

    # This function will correct the Tp from ocean model which are vertical straight lines in Hs-Tp distribution 
    # Example of how to use 
    # 
    # df = pd.read_csv('NORA3_wind_wave_lon3.21_lat56.55_19930101_20021231.csv',comment='#',index_col=0, parse_dates=True)
    # df['tp_corr'] = df.tp.values # new Tp = old Tp
    # Tp_correction(df.tp_corr.values) # make change to the new Tp
    #

    new_Tp=1+np.log(Tp/3.244)/0.09525
    index = np.where(Tp>=3.2) # indexes of Tp
    r = np.random.uniform(low=-0.5, high=0.5, size=len(Tp[index])) 
    Tp[index]=np.round(3.244*np.exp(0.09525*(new_Tp[index]-1-r)),1)
    
    return Tp 





def RVE_AM(dataframe,periods,method): ### annual GEV and GUM
    
    # This calulate return value estimates for periods using GEV or GUM
    # First fit data to get parameters 
    # Then calulate return values from the parameters 
    #
    #
    # script to test 
    # df = pd.read_csv('NORA3_wind_wave_lon3.21_lat56.55_19930101_20021231.csv',comment='#',index_col=0, parse_dates=True)
    # periods = np.array([1,10,100,10000],dtype=float)
    # return_values = RVE_AM(df.hs,periods,'GUM')
    # print (return_values)
    # out: [ 9.17030668 12.84136765 16.7078635  24.29371291]

    
    data = dataframe.resample('Y').max() # get annual maximum 
    
    for i in range(len(periods)) :
        if periods[i] == 1 : 
            periods[i] = 1.6
                
    if method == 'GEV' : 
        
        from scipy.stats import genextreme
        shape, loc, scale = genextreme.fit(data) # fit data
        
        # Compute the return levels for several return periods       
        return_levels = genextreme.isf(1/periods, shape, loc, scale)
    
    elif method == 'GUM' : 
        from scipy.stats import gumbel_r 
        loc, scale = gumbel_r.fit(data) # fit data
        
        # Compute the return levels for several return periods.
        return_levels = gumbel_r.isf(1/periods, loc, scale)
        
    else : 
        
        print ('please check method/distribution, must be either GEV or GUM')
    
    return return_levels




###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

def RVE_Weibull_2P(df,return_period,threshold):

    # how to use, to test this 
    #return_period = np.array([1,10,100,10000],dtype=float)
    #return_values = RVE_Weibull_2P(df.hs,return_period,threshold=6.2)
    #print (return_values)

    extremes = get_extremes(df, method="POT", threshold=threshold, r="48H")
    
    # Fit a 2-parameter Weibull distribution to the data
    shape, loc, scale = stats.weibull_min.fit(extremes, floc=0)
       
    duration = (df.index[-1]-df.index[0]).days + 1 
    
    length_data = extremes.shape[0]
    time_step = duration*24/length_data # in hours 
    return_period = return_period*24*365.2422/time_step # years is converted to K-th
    
    # Calculate the 2-parameter Weibull return value
    return_value = stats.weibull_min.isf(1/return_period, shape, loc, scale)
    
    for i in range(len(return_value)) : 
        return_value[i] = round(return_value[i],1)
    
    return return_value




def RVE_EXP(df,return_periods,threshold):
    
    # how to use this function 
    #return_periods = np.array([5, 10, 100, 10000]) # in years
    #return_values = RVE_EXP(df.hs,return_periods,4)
    #print (return_values)

    
    
    extremes = get_extremes(df, method="POT", threshold=threshold, r="48H")
    loc, scale = expon.fit(extremes)
    #print (loc,scale)
    
    duration = (df.index[-1]-df.index[0]).days + 1 
    
    length_data = extremes.shape[0]
    interval = duration*24/length_data # in hours 
    
    return_periods = return_periods*24*365.2422/interval # years is converted to K-th
        
    return_levels = expon.isf(1/return_periods, loc, scale)
    
    return return_levels


def Weibull_RVE(df,method,periods):

    import scipy.stats as stats
    interval = 3 # nora10
    
    # Fit a Weibull distribution to the data
    if method == '2P' : 
        shape, loc, scale = stats.weibull_min.fit(df.values, floc=0) # (ML)
    elif method == '3P' :
        shape, loc, scale = stats.weibull_min.fit(df.values) # (ML)
    else : 
        print ('please check method 2P or 3P')
        
    return_period = np.array([1,10,100,10000],dtype=float)*24*365.2422/interval  
    
    return_value = stats.weibull_min.ppf(1 - 1 / return_period, shape, loc, scale)

    for i in range(len(return_value)) : 
        return_value[i] = round(return_value[i],1)
        
    return return_value













































        
        
### TABLE 3 and 9 
#########################################################################################
#########################################################################################
#########################################################################################

def monthly_percentileNORA10(M,intensity,varName,outputfile): # months and intensity 
    # month from 1 to 12 
    # this function will sort by month and calculate percentiles 
    # it then writes to Latex file 
    
    Hs = intensity
    
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    monthlyHs = {}
    for i in range(len(months)) : 
        monthlyHs[months[i]] = [] # create empty dictionaries to store data 
    
    
    for i in range(len(Hs)) : 
        m_idx = int(M[i]-1) # Not need, get from dataframe index 
        monthlyHs[months[m_idx]].append(Hs[i])  
        
        
    with open(outputfile, 'w') as f:
        f.write('\\begin{tabular}{l | p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
        f.write(' & \multicolumn{5}{c}{' + varName + '} \\\\' + '\n')
        f.write('Month & 5\% & 50\% & Mean & 95\% & 99\% \\\\' + '\n')
        f.write('\hline' + '\n')
    
        for j in range(len(months)) : 
            Hs_P5 = round(np.percentile(monthlyHs[months[j]],5),1)
            Hs_P50 = round(np.percentile(monthlyHs[months[j]],50),1)
            Hs_mean = round(np.mean(monthlyHs[months[j]]),1)
            Hs_P95 = round(np.percentile(monthlyHs[months[j]],95),1)
            Hs_P99 = round(np.percentile(monthlyHs[months[j]],99),1)
            f.write(months[j] + ' & '+str(Hs_P5)+' & '+str(Hs_P50)+' & '+str(Hs_mean)+' & '+str(Hs_P95)+' & '+str(Hs_P99)+' \\\\' + '\n')
        
        ## annual row 
        Hs_P5 = round(np.percentile(Hs,5),1)
        Hs_P50 = round(np.percentile(Hs,50),1)
        Hs_mean = round(np.mean(Hs),1)
        Hs_P95 = round(np.percentile(Hs,95),1)
        Hs_P99 = round(np.percentile(Hs,99),1)
        f.write('Annual & '+str(Hs_P5)+' & '+str(Hs_P50)+' & '+str(Hs_mean)+' & '+str(Hs_P95)+' & '+str(Hs_P99)+' \\\\' + '\n')
    
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')
    
    return 
    



### FIGUREs 6, 7 & 18, 19  
#########################################################################################
#########################################################################################
#########################################################################################

def monthly_var_roseNORA10(month,direction,intensity,output1,output2) : 

    M = month
    DirM = direction
    Hs = intensity
    
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    dic_Hs = {}
    dic_DirM = {}
    for i in range(len(months)) : 
        dic_Hs[months[i]] = [] 
        dic_DirM[months[i]] = [] 
        
    for i in range(len(Hs)) : 
        m_idx = int(M[i]-1)
        dic_Hs[months[m_idx]].append(Hs[i])
        dic_DirM[months[m_idx]].append(DirM[i])
        
        
    ### Figure 6 
    fig = plt.figure(figsize = (8,15))
    #size = 8
    
    ax = fig.add_subplot(321, projection="windrose")
    j = 0
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(322, projection="windrose")
    j = 1
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(323, projection="windrose")
    j = 2
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(324, projection="windrose")
    j = 3
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(325, projection="windrose")
    j = 4
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(326, projection="windrose")
    j = 5
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    plt.savefig(output1,dpi=100,facecolor='white',bbox_inches='tight')
    
    
    
    ## figure 7 
    fig = plt.figure(figsize = (8,15))
    #size = 8
    
    ax = fig.add_subplot(321, projection="windrose")
    j = 6
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(322, projection="windrose")
    j = 7
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(323, projection="windrose")
    j = 8
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(324, projection="windrose")
    j = 9
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(325, projection="windrose")
    j = 10
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    ax = fig.add_subplot(326, projection="windrose")
    j = 11
    ax.bar(dic_DirM[months[j]], dic_Hs[months[j]], normed=True, opening=0.8, nsector=12)
    ax.set_legend()
    ax.set_title(months[j])
    #ax.figure.set_size_inches(size, size)
    
    plt.savefig(output2,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 






    
    
    
    
    
### TABLE 4
#########################################################################################
#########################################################################################
#########################################################################################

def monthly_MIN_MEAN_MAX_NORA10(Hs,output_file) : 
    
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    time = pd.date_range(start='1958-01-01 00:00:00', end='2022-12-31 21:00:00', freq='3H')
    df = pd.DataFrame()
    df.index = time
    df['Hs'] = Hs
    
    monthly_max = df.resample('M').max() # max in every month, all months 
    minimum = monthly_max.groupby(monthly_max.index.month)['Hs'].min() # min, sort by month
    mean = monthly_max.groupby(monthly_max.index.month)['Hs'].mean() # mean, sort by month
    maximum = monthly_max.groupby(monthly_max.index.month)['Hs'].max() # max, sort by month
    
    with open(output_file, 'w') as f :
        f.write('\\begin{tabular}{l | c c c }' + '\n')
        f.write('Month & Minimum & Mean & Maximum \\\\' + '\n')
        f.write('\hline' + '\n')
        for i in range(len(months)):
            f.write(months[i] + ' & ' + str(minimum.values[i]) + ' & ' + str(round(mean.values[i],1)) + ' & ' + str(maximum.values[i]) + ' \\\\' + '\n')
        
        ## annual row 
        annual_max = df.resample('Y').max()['Hs']
        min_year = annual_max.min()
        mean_year = annual_max.mean()
        max_year = annual_max.max()
        f.write('Annual & ' + str(min_year) + ' & ' + str(round(mean_year,1)) + ' & ' + str(max_year) + ' \\\\' + '\n')
            
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')

    return





### TABLE 5 & 12 
#########################################################################################
#########################################################################################
#########################################################################################

def directional_MIN_MEAN_MAX_NORA10(direction, intensity, output_file) : 
    
    DirP = direction 
    Hs = intensity
    
    df = pd.DataFrame()
    df.index = pd.date_range(start='1958-01-01 00:00:00', end='2022-12-31 21:00:00', freq='3H')
    df['Hs'] = Hs
    df['DirP'] = DirP
    
    # sorted by sectors/directions, keep time for the next part
    time = df.index
    bins_dir = np.arange(0,360,30) # 0,30,...,300,330
    dic_Hs = {}
    dic_time = {}
    for i in range(len(bins_dir)) : 
        dic_Hs[str(int(bins_dir[i]))] = [] 
        dic_time[str(int(bins_dir[i]))] = [] 
    
    for i in range(len(Hs)): 
        if 345 <= DirP[i] :
            dic_time[str(int(bins_dir[0]))].append(time[i])
            dic_Hs[str(int(bins_dir[0]))].append(Hs[i]) 
        else: 
            for j in range(len(bins_dir)): 
                if bins_dir[j]-15 <= DirP[i] < bins_dir[j] + 15 : # -15 --> +345 
                    dic_time[str(int(bins_dir[j]))].append(time[i])
                    dic_Hs[str(int(bins_dir[j]))].append(Hs[i]) 
                    
    
    with open(output_file, 'w') as f :
        f.write('\\begin{tabular}{l | c c c }' + '\n')
        f.write('Direction & Minimum & Mean & Maximum \\\\' + '\n')
        f.write('\hline' + '\n')
        
        # sorted by years, get max in each year, and statistical values 
        for j in range(len(bins_dir)):
            df_dir = pd.DataFrame()
            df_dir.index = dic_time[str(int(bins_dir[j]))]
            df_dir['Hs'] = dic_Hs[str(int(bins_dir[j]))]
            annual_max_dir = df_dir.resample('Y').max()
            mind = annual_max_dir.min()['Hs']
            meand = annual_max_dir.mean()['Hs']
            maxd = annual_max_dir.max()['Hs']
            start = bins_dir[j] - 15
            if start < 0 : 
                start = 345 
            f.write(str(start) + '-' + str(bins_dir[j]+15) + ' & ' + str(mind) + ' & ' + str(round(meand,1)) + ' & ' + str(maxd) + ' \\\\' + '\n')
            
        
        ## annual row 
        df = pd.DataFrame()
        df.index = time
        df['Hs'] = Hs
        
        annual_max = df.resample('Y').max()
        mind = annual_max.min()['Hs']
        meand = annual_max.mean()['Hs']
        maxd = annual_max.max()['Hs']
        f.write('Annual & ' + str(mind) + ' & ' + str(round(meand,1)) + ' & ' + str(maxd) + ' \\\\' + '\n')
            
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')

    return





### AM_GEV
#########################################################################################
#########################################################################################
#########################################################################################
def RVE_AM_GEV(dataframe,periods):
    
    ## example 
    #data = datafame['AnnualMax']
    #periods = np.array([1, 10, 100, 1000, 10000],dtype=float)
    #returns = RVE_AM_GEV(data,periods)


    from scipy.stats import genextreme as fuction
    # Fit the generalized extreme value distribution to the data.
    shape, loc, scale = fuction.fit(data)
    print("Fit parameters:")
    print(f"  shape: {shape:.4f}")
    print(f"  loc:   {loc:.4f}")
    print(f"  scale: {scale:.4f}")
    print()
    
    # Compute the return levels for several return periods.
    #return_periods = np.array([1.6, 10, 100, 10000])
    for i in range(len(periods)) :
        if periods[i] == 1 :
            periods[i] = 1.6
            
    #print (periods)
    return_levels = fuction.isf(1/periods, shape, loc, scale)
    
    ###plt.plot(return_periods,return_levels,'+')
    ###plt.xscale('log')
    ###plt.xlabel('Average Return Interval (Year)')
    ###print("Period    Level")
    ###print("(years)   (wave height)")
    
    #for period, level in zip(return_periods, return_levels):
    #    print(f'{period:6.0f}  {level:9.1f}')
    
    return return_levels
    


### AM_GUM
#########################################################################################
#########################################################################################
#########################################################################################
def RVE_AM_GUM(dataframe,periods):

    #####data = dfam['AnnualMax']
    #####periods = np.array([1, 10, 100, 1000, 10000],dtype=float)
    #####returns = RVE_AM_GUM(data,periods)
    #####print (returns)


    from scipy.stats import gumbel_r as fuction
    
    # Fit the generalized extreme value distribution to the data.
    loc, scale = fuction.fit(data)
    print("Fit parameters:")
    print(f"  loc:   {loc:.4f}")
    print(f"  scale: {scale:.4f}")
    print()
    
    for i in range(len(periods)) :
        if periods[i] == 1 :
            periods[i] = 1.6
            
    # Compute the return levels for several return periods.
    return_levels = fuction.isf(1/periods, loc, scale)
    
    #plt.plot(return_periods,return_levels,'ro')
    #plt.xscale('log')
    #plt.xlabel('Average Return Interval (Year)')
    #print("Period    Level")
    #print("(years)   (wave height)")
    
    #for period, level in zip(return_periods, return_levels):
    #    print(f'{period:6.0f}  {level:9.1f}')
    return return_levels



def uv_to_spd_dir_current1(u_ms,v_ms): # individual values 
    
    speed = np.sqrt(u_ms**2 + v_ms**2)
    uv_to_dir = np.arctan2(u_ms/speed, v_ms/speed) # uv to direction
    direction = uv_to_dir * 180/np.pi ## direction to degree
    
    if u_ms < 0 :
        direction = 360 + direction
        
    return speed, direction
    
    
def uv_to_spd_dir_current_roms(u_ms,v_ms,angle): # individual values 
    
    u_EW = u_ms*np.cos(angle) - v_ms*np.sin(angle)
    v_NS = v_ms*np.cos(angle) + u_ms*np.sin(angle)
    speed,direction = uv_to_spd_dir_current1(u_EW,v_NS)
        
    return speed, direction

def uv_to_spd_dir_current(u_ms,v_ms): # data frame 
    # work for wind rose, where the current drift to 
      
    speed = np.sqrt(u_ms**2 + v_ms**2)
    uv_to_dir = np.arctan2(u_ms/speed, v_ms/speed) # uv to direction
    direction = uv_to_dir * 180/np.pi ## direction to degree
    direction[direction < 0] = direction[direction < 0] + 360 
    
    return speed, direction

def uv_to_spd_dir_wind(u_ms,v_ms): # data frame 
    # work for wind rose, where the wind come from  
       
    speed = np.sqrt(u_ms**2 + v_ms**2)
    uv_to_dir = np.arctan2(u_ms/speed, v_ms/speed) # uv to direction
    direction = uv_to_dir * 180/np.pi ## direction to degree
    direction[direction < 0] = direction[direction < 0] + 360 
    
    direction = direction + 180 
    direction[direction > 360] = direction[direction > 360] - 360 
    
    return speed, direction



def make_map(lon1,lon2,lat1,lat2): # make_map(lon1,lon2,lat1,lat2,outfile)

    plt.figure(figsize=(11,7))
    m = Basemap(llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2,resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries() 
    m.fillcontinents()
    parallels = np.arange(-90,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0,360,2)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    
    #m.plot(lon0, lat0, 'ro', markersize=5)
    
    #from matplotlib.font_manager import FontProperties
    #font0 = FontProperties()
    #font = font0.copy()
    #font.set_weight('bold')
    
    #m.plot((lon1+lon2)/2, (lat1+lat2)/2, 'ro', markersize=5)
    #plt.text((lon1+lon2)/2+0.2,(lat1+lat2)/2+0.2,'Test', fontproperties=font) 
    
    
    
    #plt.savefig(outfile,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 
    
    
def make_map_wind_satistics(lon1,lon2,lat1,lat2,lon0,lat0): # make_map(lon1,lon2,lat1,lat2,outfile)

    plt.figure(figsize=(11,7))
    m = Basemap(llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2,resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries() 
    m.fillcontinents()
    parallels = np.arange(-90,90,0.5)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0,360,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    
    m.plot(lon0, lat0, 'ro', markersize=5)
    
    #from matplotlib.font_manager import FontProperties
    #font0 = FontProperties()
    #font = font0.copy()
    #font.set_weight('bold')
    
    #m.plot((lon1+lon2)/2, (lat1+lat2)/2, 'ro', markersize=5)
    #plt.text((lon1+lon2)/2+0.2,(lat1+lat2)/2+0.2,'Test', fontproperties=font) 
    
    
    
    #plt.savefig(outfile,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 
    
    
    
def make_VN_map(lon0,lat0,outfile): # make_map(lon1,lon2,lat1,lat2,outfile)

    lon1 = 98
    lon2 = 122
    lat1 = 0
    lat2 = 24 
    
    plt.figure(figsize=(11,7))
    m = Basemap(llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2,resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries() 
    m.fillcontinents()
    parallels = np.arange(-90,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0,360,2)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    
    
    #from matplotlib.font_manager import FontProperties
    #font0 = FontProperties()
    #font = font0.copy()
    #font.set_weight('bold')
    
    m.plot(lon0, lat0, 'ro', markersize=5)
    #plt.text((lon1+lon2)/2+0.2,(lat1+lat2)/2+0.2,'Test', fontproperties=font) 
    
    
    
    plt.savefig(outfile,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 
    
    
def make_Mosambik_map(lon1,lon2,lat1,lat2,outfile,lonS,latS): # make_map(lon1,lon2,lat1,lat2,outfile)

    plt.figure(figsize=(11,7))
    m = Basemap(llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2,resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries() 
    m.fillcontinents()
    parallels = np.arange(-90,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0,360,2)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    m.plot(lonS, latS, 'ro', markersize=5)    
        
    from matplotlib.font_manager import FontProperties
    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    font.set_size('xx-small') #sizes = ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
    
    
    lon0 = 32.6
    lat0 = -25.8
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-1,lat0+0.25,'Maputo', fontproperties=font) 
    
    lon0 = 35.4
    lat0 = -23.9
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-2,lat0+0.25,'Inhambane', fontproperties=font) 
    
    lon0 = 35.3
    lat0 = -22
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-2,lat0+0.25,'Vilankulo', fontproperties=font)     
    
    lon0 = 34.9
    lat0 = -19.8
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-1,lat0+0.25,'Beira', fontproperties=font)  
    
    lon0 = 36.9
    lat0 = -17.8
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-1,lat0+0.25,'Quelimane', fontproperties=font)      
        
    lon0 = 40.0
    lat0 = -16.2
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-3.5,lat0+0.25,'Cidade de Angoche', fontproperties=font) 
    
    
    lon0 = 40.7
    lat0 = -14.5
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-1,lat0+0.25,'Nacala', fontproperties=font)      
        
    lon0 = 40.5
    lat0 = -13.0
    m.plot(lon0, lat0, 'go', markersize=5)
    plt.text(lon0-1,lat0+0.25,'Pemba', fontproperties=font)     
       
    plt.savefig(outfile,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 




def make_Mosambik_map_current(infile,t): 

    nc = Dataset(infile)
    time = convert_ncdate(nc)
    lon=nc.variables['longitude'][:]
    lat=nc.variables['latitude'][:]
    u10 = nc.variables['uo'][:]
    v10 = nc.variables['vo'][:]
    
    
    
    fig = plt.figure(figsize=(10,7))
    m = Basemap(llcrnrlat=lat[0],urcrnrlat=lat[-1],llcrnrlon=lon[0],urcrnrlon=lon[-1], resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    #m.etopo()
    m.fillcontinents(color='0.6')
    parallels = np.arange(-60.,60,1.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0.,360.,2.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    x2, y2 = m(lon,lat)
    x2, y2 = np.meshgrid(x2,y2) 
    
    
    
    n=5 # arrow 
    #t=1555 # time 
    lev = 0 # surface 
    w10 = np.sqrt(u10[t,lev,:]**2 + v10[t,lev,:]**2)
    cf = m.contourf(x2,y2,w10,levels = np.arange(0,2,0.3)) #, extend='both'
    q = m.quiver(x2[0::n,0::n],y2[0::n,0::n],u10[t,lev,0::n,0::n]/w10[0::n,0::n],v10[t,lev,0::n,0::n]/w10[0::n,0::n],headlength=3, scale=9,units='inches')
    cbar = m.colorbar(cf,location='right',pad="10%")
    cbar.ax.set_title('[m/s]')
    
    plt.title('Ocean currents at ' + time[t].strftime("%H:%M %d/%m/%Y") + ' (UTC)')
    plt.savefig('test.png',dpi=100,facecolor='white',bbox_inches='tight')
    
    
    
def make_map_wind(infile,t): # make_map(lon1,lon2,lat1,lat2,outfile)
    nc = Dataset(infile)
    time = convert_ncdate(nc)
    lon=nc.variables['longitude'][:]
    lat=nc.variables['latitude'][:]
    u10 = nc.variables['u10'][:]
    v10 = nc.variables['v10'][:]
    
    speed_, direction_ = uv_to_spd_dir_wind(u10,v10) 
    
    t = -121 # select time 
    speed = speed_[t,:] 
    direction = direction_[t,:]
    
    fig = plt.figure(figsize=(11,7))
    m = Basemap(llcrnrlat=lat[-1],urcrnrlat=lat[0],llcrnrlon=lon[0],urcrnrlon=lon[-1], resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.fillcontinents(color='0.7')
    parallels = np.arange(-90,90,1.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0.,360.,2.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    x2, y2 = m(lon,lat)
    x2, y2 = np.meshgrid(x2,y2) 
    # for max waves 
    #cf = m.contourf(x2,y2,wave,levels = [0,2,4,6,8,10], colors=('white','lavender','lightsteelblue','steelblue','mediumseagreen','lime','yellow'),extend='both')
    #cs = plt.contour(x2,y2,wave,levels = [1,3,5,7,9], colors='g')
    
    # for a snapshot 
    n = 3 # distance between arrows 
    cf = m.contourf(x2,y2,speed,levels = [0,4,8,12,16,20,24], colors=('lavender','lightsteelblue','steelblue','mediumseagreen','lime','yellow'))
    #cs = plt.contour(x2,y2,speed,levels = [2,6,10,14,18,22], colors='g')
    q = m.quiver(x2[0::n,0::n],y2[0::n,0::n],np.sin(-np.pi*direction[0::n,0::n]/180),-np.cos(np.pi*direction[0::n,0::n]/180),headlength=5, scale=9,units='inches')
    
    #plt.clabel(cs,inline=1,fmt='%3.1f',fontsize=10,colors='k')
    cbar = m.colorbar(cf,location='right',pad="10%")
    cbar.ax.set_title('Speed (m/s)')
    
    plt.title('Wind at ' + time[t].strftime("%H:%M %d/%m/%Y") + ' (UTC)')
    plt.savefig('wind.png',dpi=100,facecolor='white',bbox_inches='tight')
    
    
    
    
def make_Mosambik_map_waves(infile,t):
    nc = Dataset(infile)
    lon=nc.variables['longitude'][:]
    lat=nc.variables['latitude'][:]
    time = convert_ncdate(nc)
    wave_height = nc.variables['swh'][:]
    
    # for max waves
    #wave = wave_height.max(0) 
    
    # for a snapshot t 
    #t = -121
    wave = wave_height[t,:,:]
    wave_direction = nc.variables['mwd'][:]
    direction = wave_direction[t,:,:]
    
    
    
    fig = plt.figure(figsize=(11,7))
    m = Basemap(llcrnrlat=lat[-1],urcrnrlat=lat[0],llcrnrlon=lon[0],urcrnrlon=lon[-1], resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.fillcontinents(color='0.7')
    parallels = np.arange(-90,90,1.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0.,360.,2.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    x2, y2 = m(lon,lat)
    x2, y2 = np.meshgrid(x2,y2) 
    # for max waves 
    #cf = m.contourf(x2,y2,wave,levels = [0,2,4,6,8,10], colors=('white','lavender','lightsteelblue','steelblue','mediumseagreen','lime','yellow'),extend='both')
    #cs = plt.contour(x2,y2,wave,levels = [1,3,5,7,9], colors='g')
    
    # for a snapshot 
    n = 2 # distance between arrows 
    cf = m.contourf(x2,y2,wave,levels = [0,1,2,3,4,5,6], colors=('lavender','lightsteelblue','steelblue','mediumseagreen','lime','yellow'))
    cs = plt.contour(x2,y2,wave,levels = [0.5,1.5,2.5,3.5,4.5,5.5], colors='g')
    q = m.quiver(x2[0::n,0::n],y2[0::n,0::n],np.sin(-np.pi*direction[0::n,0::n]/180),-np.cos(np.pi*direction[0::n,0::n]/180),headlength=5, scale=6,units='inches')
    
    plt.clabel(cs,inline=1,fmt='%3.1f',fontsize=10,colors='k')
    cbar = m.colorbar(cf,location='right',pad="10%")
    cbar.ax.set_title('Wave (m)')
    
    plt.title('Waves at ' + time[t].strftime("%H:%M %d/%m/%Y"))
    plt.savefig('Waves.png',dpi=100,facecolor='white',bbox_inches='tight')
    
    
    
    
def make_VN_map_waves(infile,t):
    nc = Dataset(infile)
    lon=nc.variables['longitude'][:]
    lat=nc.variables['latitude'][:]
    time = convert_ncdate(nc)
    wave_height = nc.variables['swh'][:]
    
    # for max waves
    #wave = wave_height.max(0) 
    
    # for a snapshot t 
    #t = -121
    wave = wave_height[t,:,:]
    wave_direction = nc.variables['mwd'][:]
    direction = wave_direction[t,:,:]
    
    
    
    fig = plt.figure(figsize=(11,7))
    m = Basemap(llcrnrlat=lat[-1],urcrnrlat=lat[0],llcrnrlon=lon[0],urcrnrlon=lon[-1], resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.fillcontinents(color='0.7')
    parallels = np.arange(-90,90,1.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0.,360.,2.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    x2, y2 = m(lon,lat)
    x2, y2 = np.meshgrid(x2,y2) 
    # for max waves 
    #cf = m.contourf(x2,y2,wave,levels = [0,2,4,6,8,10], colors=('white','lavender','lightsteelblue','steelblue','mediumseagreen','lime','yellow'),extend='both')
    #cs = plt.contour(x2,y2,wave,levels = [1,3,5,7,9], colors='g')
    
    # for a snapshot 
    n = 2 # distance between arrows 
    cf = m.contourf(x2,y2,wave,levels = [0,0.5,1,1.5,2,3,4,5], colors=('white','lavender','lightsteelblue','steelblue','mediumseagreen','lime','yellow'))
    cs = plt.contour(x2,y2,wave,levels = [6,8,10,12], colors='g')
    q = m.quiver(x2[0::n,0::n],y2[0::n,0::n],np.sin(-np.pi*direction[0::n,0::n]/180),-np.cos(np.pi*direction[0::n,0::n]/180),headlength=5, scale=6,units='inches')
    
    plt.clabel(cs,inline=1,fmt='%3.1f',fontsize=10,colors='k')
    cbar = m.colorbar(cf,location='right',pad="10%")
    cbar.ax.set_title('Wave (m)')
    
    plt.title('Waves at ' + time[t].strftime("%H:%M %d/%m/%Y"))
    plt.savefig('Waves.png',dpi=100,facecolor='white',bbox_inches='tight')
    
    
    
       
def convert_ncdate(nc):

    #####nc = netCDF4.Dataset('Mosambik_10y_6h_30E_46E_30S_10S.nc')
    #####time_format = nc.variables['time']
    
    from datetime import datetime, timedelta
    
    time_format = nc.variables['time']
    time_values = time_format[:]
    ref = time_format.units
    lst = ref.split() # ['hours', 'since', '1900-01-01', '00:00:00.0']
    ref2 = lst[2].split('-')
    
    refDate = datetime(int(ref2[0]), int(ref2[1]), int(ref2[2]))
    if lst[0] == 'hours' :
        time = [refDate + timedelta(hours=float(t)) for t in time_values] # convert to float, sometimes no need 
    elif lst[0] == 'seconds' :
        time = [refDate + timedelta(seconds=float(t)) for t in time_values] # convert to float, sometimes no need 
    else:
        print ('please check the reference date')
        
    return time
    
    
     
    
def find_position(x0, y0, x, y): # this is for roms 
    dist = (x-x0)**2 + (y-y0)**2
    # find point with minimum distance
    yi, xi = nd.minimum_position(dist)
    return yi, xi
    
    
    
def find_nearest_index(array,value):
    
    idx = (np.abs(array-value)).argmin()
    
    return idx
    
def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx  
    
    
    
    
### READ DATA      
#########################################################################################
#########################################################################################
#########################################################################################

def getNORA10(lonS,latS,start_time,end_time):

    path = '/vol/hindcast4/magnarr/ndpdata/'
    #path = '/lustre/storeB/project/fou/om/NORA10/ndpfiler/';
    
    # get all files in given directory
    files = os.listdir(path=path)
    
    # remove files without key words
    keywords = 'NORA10*.txt'
    #keywords = 'NORA10X*.txt'
    words = keywords.split('*')
    for file in files:        
        for word in words:
            if word not in file:
                files.remove(file)
            
    # from file names to list of coordinates
    lst = [] # list of coordinates 
    for file in files: 
        lat = float(file[7:11])/100
        lon = float(file[13:17])/100
        if file[17] == 'W':
            lon = -lon
        lst.append((lat,lon))
    
    # find closest grid point  
    location = np.array([latS,lonS]) # lat, lon 
    grid=np.array(lst) # convert list to array to calculate distance 
    distance = ((grid[:,0] - location[0]) ** 2 + (grid[:,1] - location[1]) ** 2) ** 0.5  # calculate distance
    closest_idx = np.argmin(distance) # get the index of closest coordinate
    
    # read nearest data point 
    file = path+files[closest_idx]
    #print ('Data file: ',file)
    with open(file) as f:
        content = f.readlines()
    print ('Data start from '+content[4][:10]+' to '+content[-1][:10])
        
    start = start_time
    end = end_time
    start_idx = 0
    end_idx = 0
    for i in range(4,len(content)):
        date = content[i][:10]
        y = date[0:4]
        m = date[5:7]
        d = date[8:10]
        
        m2 = str(int(m)).zfill(2)
        d2 = str(int(d)).zfill(2)
        new_date = y+'-'+m2+'-'+d2
        if new_date==start:
            start_idx=i-7 # it repeats until the last hour step of the day 
        if new_date==end:
            end_idx=i
            
    # write to new file 
    output=files[closest_idx][:-3]+start_time+'.'+end_time+'.txt'
    #print(output)
    output_file = open(output,"w+")
    output_file.write(content[0])
    output_file.write(content[1])
    output_file.write(content[2])
    output_file.write(content[3])
    for i in range(start_idx,end_idx+1):
        output_file.write(content[i])
    output_file.close()
    
    return output
    
def readExtractionFile(file):
      
    with open(file) as f:
        content = f.readlines()
    line3 = content[3]
    heading = line3.split()
    
    data = np.empty((len(content)-4,27))
    data[:] = np.nan
    for row in range(4,len(content)): 
        line = content[row]
        words = line.split()
        for column in range(len(words)):
            data[row-4,column]=float(words[column])
    
    Y = data[:,0]
    M = data[:,1]
    D = data[:,2]
    H = data[:,3]
    W10 = data[:,8] 
    D10 = data[:,13]
    Hs = data[:,16]
    Tp = data[:,17]
    DirP = data[:,19]  
    DirM = data[:,20]  
    
    return Y,M,D,H,W10,D10,Hs,Tp,DirP,DirM     
    
def readTXT(file):
    with open(file) as f:
        content = f.readlines()

    data = np.empty((len(content)-1,2)) # ROWS, COLUMNS, ELIMINATE FIRST ROW 
    data[:] = np.nan
    for row in range(1,len(content)): # ELIMINATE FIRST ROW 
        line = content[row]
        words = line.split()
        for column in range(len(words)):
            data[row-1,column]=float(words[column])
    
    Tp = data[:,0]
    Hs = data[:,1] 
    
    return Hs,Tp  
        
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles 
    
# get depth 
def getDepth(infile,lon0,lat0): # for roms file 
    nc = Dataset(infile)
    lon = nc.variables['lon_rho'][:]
    lat = nc.variables['lat_rho'][:]
    h = nc.variables['h'][:]
    eta_rho, xi_rho = find_position(lon0, lat0, lon, lat) 
    depth = h[eta_rho, xi_rho]
    return depth 
    
def downloadSVIM(name,lon0,lat0) : 
    # this will download data at location 'name', coordinates lon0,lat0
    # then save data as u_'name', v_'name', time to pickle files
    # move to folder named 'name'
    
    ## get all files, then sort
    files = getListOfFiles('/lustre/storeA/project/fou/hi/legacy-ocean/thredds/SVIM')
    files.sort()
    
    ### get daily files, then sort
    daily_files = []
    for file in files:        
        if len(file)==83: 
            daily_files.append(file)
    daily_files.sort() # sort them out 
    
    ## get full years
    N=732 # Jan 1960 to Dec 2020: 61 years 
    files = daily_files[:732]
    
    
    
    infile = 'SVIM/Data/SVIM.angle_h.nc'
    y0, x0 = getIndexes(infile,lon0,lat0)
    
    for i in range(len(files)): 
        file2read = Dataset(files[i])
        dictionary = {}
        dictionary['ocean_time'] = file2read['ocean_time'][:]
        dictionary['u_'+name] = file2read['u'][:,:,y0,x0]
        dictionary['v_'+name] = file2read['v'][:,:,y0,x0]
        outf = files[i][-22:-4]+'.pkl'
        with open(outf, 'wb') as f:
            pickle.dump(dictionary, f)
            shutil.move(outf, name)
            
    return     
   
   
def plot_Domain(infile,lon0,lat0) : ## for ROMS (SVIM,NK800)
    
    nc = Dataset(infile)
    lon = nc.variables['lon_rho'][:]
    lat = nc.variables['lat_rho'][:]
    eta_rho, xi_rho = lon.shape
    
    ### MAP 
    lon1 = np.min(lon)
    lon2 = np.max(lon)
    lat1 = np.min(lat)
    lat2 = np.max(lat)

    fig = plt.figure(figsize=(8,10))
    m = Basemap(llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2, resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.fillcontinents(color='0.6')
    parallels = np.arange(0.,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0.,360.,2)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    ### Plotting
    for eta in range(0,eta_rho,20):
        for xi in range(0,xi_rho,50):
            if lon[eta,xi] < lon2 and lat[eta,xi] < lat2 : 
                m.plot(lon[eta,xi],lat[eta,xi],'k+',markersize=1)
    
    m.plot(lon0,lat0,'ro',markersize=5)
    
    #m.plot(3.28,61.07,'ko',markersize=5)
    #font0 = FontProperties()
    #font = font0.copy()
    #font.set_weight('bold')
    #font.set_size('xx-small') #sizes = ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
    #plt.text(lon0-1,lat0+0.2,'Horatio', fontproperties=font) 
    
    #plt.savefig('NK800.domain.png',dpi=100,facecolor='white',bbox_inches='tight')
    
    return 
    
def plot_Domain_newNK800(infile,lon0,lat0) : ## for ROMS (SVIM,NK800)
    
    nc = Dataset(infile)
    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]
    eta_rho, xi_rho = lon.shape
    
    ### MAP 
    lon1 = np.min(lon)
    lon2 = np.max(lon)
    lat1 = np.min(lat)
    lat2 = np.max(lat)

    fig = plt.figure(figsize=(8,10))
    m = Basemap(llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2, resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.fillcontinents(color='0.6')
    parallels = np.arange(0.,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0.,360.,2)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    ### Plotting
    for eta in range(0,eta_rho,20):
        for xi in range(0,xi_rho,50):
            if lon[eta,xi] < lon2 and lat[eta,xi] < lat2 : 
                m.plot(lon[eta,xi],lat[eta,xi],'k+',markersize=1)
    
    m.plot(lon0,lat0,'ro',markersize=5)
    
    #m.plot(3.28,61.07,'ko',markersize=5)
    #font0 = FontProperties()
    #font = font0.copy()
    #font.set_weight('bold')
    #font.set_size('xx-small') #sizes = ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
    #plt.text(lon0-1,lat0+0.2,'Horatio', fontproperties=font) 
    
    #plt.savefig('NK800.domain.png',dpi=100,facecolor='white',bbox_inches='tight')
    
    return 
    
def plot_Domain_details(infile,lon0,lat0) : ## for ROMS (SVIM,NK800)
    
    nc = Dataset(infile)
    lon = nc.variables['lon_rho'][:]
    lat = nc.variables['lat_rho'][:]
    eta_rho, xi_rho = lon.shape
    
    ### MAP 
    lon1 = lon0-0.1
    lon2 = lon0+0.1
    lat1 = lat0-0.1
    lat2 = lat0+0.1

    fig = plt.figure(figsize=(6,9))
    m = Basemap(llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2, resolution='i')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.fillcontinents(color='0.6')
    parallels = np.arange(0.,90,0.1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(0.,360.,0.1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    
    ### Plotting
    m.plot(lon0,lat0,'ro',markersize=1)
    print ('lon,lat requested : ',lon0,lat0)
    eta0,xi0 = getIndexes(infile,lon0,lat0)
    print ('eta,xi: ',eta0,xi0,'Nearest')
    print ()
    #for eta in range(eta0-1,eta0+2):
    #    for xi in range(xi0-1,xi0+1): #OK 
    #        print (eta,xi)
    #        m.plot(lon[eta,xi],lat[eta,xi],'k+',markersize=1)
    eta=578
    xi=2173
    m.plot(lon[eta,xi],lat[eta,xi],'k+',markersize=1)
    print ('eta,xi: ',eta,xi)
    print ('lon[eta,xi],lat[eta,xi]: ',np.round(lon[eta,xi],3),np.round(lat[eta,xi],3))
    print ()
    
    eta=579
    xi=2173
    m.plot(lon[eta,xi],lat[eta,xi],'k+',markersize=1)
    print ('eta,xi: ',eta,xi)
    print ('lon[eta,xi],lat[eta,xi]: ',np.round(lon[eta,xi],3),np.round(lat[eta,xi],3))
    print ()
    
    eta=579
    xi=2174
    m.plot(lon[eta,xi],lat[eta,xi],'k+',markersize=1)
    print ('eta,xi: ',eta,xi)
    print ('lon[eta,xi],lat[eta,xi]: ',np.round(lon[eta,xi],3),np.round(lat[eta,xi],3))
    print ()
    
    eta=580
    xi=2174
    m.plot(lon[eta,xi],lat[eta,xi],'k+',markersize=1)
    print ('eta,xi: ',eta,xi)
    print ('lon[eta,xi],lat[eta,xi]: ',np.round(lon[eta,xi],3),np.round(lat[eta,xi],3))
    print ()

    
    #m.plot(3.28,61.07,'ko',markersize=5)
    #font0 = FontProperties()
    #font = font0.copy()
    #font.set_weight('bold')
    #font.set_size('xx-small') #sizes = ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
    #plt.text(lon0-1,lat0+0.2,'Horatio', fontproperties=font) 
    
    #plt.savefig('NK800.domain.png',dpi=100,facecolor='white',bbox_inches='tight')
    
    return 

    
def find_NORA10_depth(lon0, lat0):
    # this find nearst coordinates (lat, lon) and depth
    import numpy as np
    
    with open('ndp_bottom.txt') as f:
        content = f.readlines()
           
    lst = [] # list of coordinates 
    depth = []
    for row in range(len(content)): # ELIMINATE FIRST ROW 
        line = content[row]
        words = line.split() #latitude, longitude, depth
        lat = float(words[0])
        lon = float(words[1])
        lst.append((lat,lon))
        depth.append(float(words[2]))
            
    # find closest grid point  
    location = np.array([lat0,lon0]) # lat, lon 
    grid=np.array(lst) # convert list to array to calculate distance 
    distance = ((grid[:,0] - location[0]) ** 2 + (grid[:,1] - location[1]) ** 2) ** 0.5  # calculate distance
    closest_idx = np.argmin(distance) # get the index of closest coordinate

    return depth[closest_idx],lst[closest_idx]
    
    
    def download_NK800_2(lon0,lat0,path):
    # this download u_eastward,v_northward and save to file 
    try:
        os.mkdir(path) 
    except:
        pass
    
    ## getting all files then download data at all depth layers, save to a folder 
    files = getListOfFiles('/lustre/storeB/project/copernicus/sea/romsnorkystv2/zdepths1h')
    files.sort()
    
    nc = Dataset(files[0])
    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]
    
    eta, xi = getIndexes_newNK800(files[0],lon0,lat0)
    
    for i in range(len(files)): # 1297: 2021.01.01 and 2026: 2022.12.31
        file2read = Dataset(files[i])
        dictionary = {}
        dictionary['time'] = file2read['time'][:]
        dictionary['u_eastward'] = file2read['u_eastward'][:,:,eta,xi]
        dictionary['v_northward'] = file2read['v_northward'][:,:,eta,xi]
        
        outf = files[i][-13:]+'.pkl'
        with open(outf, 'wb') as f:
            pickle.dump(dictionary, f)
            shutil.move(outf, path)
            
def load_NK800_2(path):
    # load u and v and merge
    files = getListOfFiles(path)
    files.sort()
    
    u1 = 'u_eastward'
    v1 = 'v_northward'
    
    with open(files[0], 'rb') as f: # save first one 
        loaded_dict = pickle.load(f)
        time2 = loaded_dict['time']
        u = loaded_dict[u1]
        v = loaded_dict[v1]
    for i in range(1,len(files)): # add next ones to the first one
        with open(files[i], 'rb') as f:
            loaded_dict = pickle.load(f)
            time2_ = loaded_dict['time']
            time2 = np.concatenate((time2, time2_))
            a = loaded_dict[u1]
            u = np.concatenate((u, a), axis=0)
            a = loaded_dict[v1]
            v = np.concatenate((v, a), axis=0)
            
    refDate = datetime(1970, 1, 1)
    time = [refDate + timedelta(seconds=t) for t in time2]
    depth = np.array([0, 3, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 500, 1000, 2000, 3000])
    
    return time, u, v, depth
