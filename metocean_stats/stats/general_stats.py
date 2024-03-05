import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from math import floor,ceil

from .aux_funcs import convert_latexTab_to_csv

def calculate_scatter(data: pd.DataFrame, var1: str, step_var1: float, var2: str, step_var2: float) -> pd.DataFrame:
    """
    Create scatter table of two variables (e.g, var1='hs', var2='tp')
    step_var: size of bin e.g., 0.5m for hs and 1s for Tp
    The rows are the upper bin edges of var1 and the columns are the upper bin edges of var2
    """
    dvar1 = data[var1]
    v1min = np.min(dvar1)
    v1max = np.max(dvar1)
    if step_var1 > v1max:
        raise ValueError(f"Step size {step_var1} is larger than the maximum value of {var1}={v1max}.")

    dvar2 = data[var2]
    v2min = np.min(dvar2)
    v2max = np.max(dvar2)
    if step_var2 > v2max:
        raise ValueError(f"Step size {step_var2} is larger than the maximum value of {var2}={v2max}.")

    # Find the the upper bin edges
    max_bin_1 = ceil(v1max / step_var1)
    max_bin_2 = ceil(v2max / step_var2)
    min_bin_1 = ceil(v1min / step_var1)
    min_bin_2 = ceil(v2min / step_var2)

    offset_1 = min_bin_1 - 1
    offset_2 = min_bin_2 - 1

    var1_upper_bins = np.arange(min_bin_1, max_bin_1 + 1, 1) * step_var1
    var2_upper_bins = np.arange(min_bin_2, max_bin_2 + 1, 1) * step_var2

    row_size = len(var1_upper_bins)
    col_size = len(var2_upper_bins)
    occurences = np.zeros([row_size, col_size])

    for v1, v2 in zip(dvar1, dvar2):
        # Find the correct bin and sum up
        row = floor(v1 / step_var1) - offset_1
        col = floor(v2 / step_var2) - offset_2
        occurences[row, col] += 1

    return pd.DataFrame(data=occurences, index=var1_upper_bins, columns=var2_upper_bins)

def scatter_diagram(data: pd.DataFrame, var1: str, step_var1: float, var2: str, step_var2: float, output_file):
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    Plot scatter diagram (heatmap) of two variables (e.g, var1='hs', var2='tp')
    step_var: size of bin e.g., 0.5m for hs and 1s for Tp
    cmap: colormap, default is 'Blues'
    outputfile: name of output file with extrensition e.g., png, eps or pdf 
     """

    sd = calculate_scatter(data, var1, step_var1, var2, step_var2)

    # Convert to percentage
    tbl = sd.values
    var1_data = data[var1]
    tbl = tbl/len(var1_data)*100

    # Then make new row and column labels with a summed percentage
    sumcols = np.sum(tbl, axis=0)
    sumrows = np.sum(tbl, axis=1)

    sumrows = np.around(sumrows, decimals=1)
    sumcols = np.around(sumcols, decimals=1)

    bins_var1 = sd.index
    bins_var2 = sd.columns
    lower_bin_1 = bins_var1[0] - step_var1
    lower_bin_2 = bins_var2[0] - step_var2

    rows = []
    rows.append(f'{lower_bin_1:04.1f}-{bins_var1[0]:04.1f} | {sumrows[0]:04.1f}%')
    for i in range(len(bins_var1)-1):
        rows.append(f'{bins_var1[i]:04.1f}-{bins_var1[i+1]:04.1f} | {sumrows[i+1]:04.1f}%')

    cols = []
    cols.append(f'{int(lower_bin_2)}-{int(bins_var2[0])} | {sumcols[0]:04.1f}%')
    for i in range(len(bins_var2)-1):
        cols.append(f'{int(bins_var2[i])}-{int(bins_var2[i+1])} | {sumcols[i+1]:04.1f}%')

    rows = rows[::-1]
    tbl = tbl[::-1,:]
    dfout = pd.DataFrame(data=tbl, index=rows, columns=cols)
    hi = sns.heatmap(data=dfout.where(dfout>0), cbar=True, cmap='Blues', fmt=".1f")
    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return hi

def table_var_sorted_by_hs(data, var, var_hs='hs', output_file='var_sorted_by_Hs.txt'):
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    This will sort variable var e.g., 'tp' by 1 m interval og hs
    then calculate min, percentile 5, mean, percentile 95 and max
    data : panda series 
    var  : variable 
    output_file: extension .txt for latex table or .csv for csv table
    """
    Hs = data[var_hs]
    Var = data[var]
    binsHs = np.arange(0.,math.ceil(np.max(Hs))+0.1) # +0.1 to get the last one   
    temp_file = output_file.split('.')[0]

    Var_binsHs = {}
    for j in range(len(binsHs)-1) : 
        Var_binsHs[str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))] = [] 
    
    N = len(Hs)
    for i in range(N):
        for j in range(len(binsHs)-1) : 
            if binsHs[j] <= Hs.iloc[i] < binsHs[j+1] : 
                Var_binsHs[str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))].append(Var.iloc[i])
    
    with open(temp_file, 'w') as f:
        f.write('\\begin{tabular}{l p{1.5cm}|p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
        f.write('& & \multicolumn{5}{c}{'+var+'} \\\\' + '\n')
        f.write('Hs & Entries & Min & 5\% & Mean & 95\% & Max \\\\' + '\n')
        f.write('\hline' + '\n')
    
        for j in range(len(binsHs)-1) : 
            Var_binsHs_temp = Var_binsHs[str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))]
            
            Var_min = round(np.min(Var_binsHs_temp), 1)
            Var_P5 = round(np.percentile(Var_binsHs_temp , 5), 1)
            Var_mean = round(np.mean(Var_binsHs_temp), 1)
            Var_P95 = round(np.percentile(Var_binsHs_temp, 95), 1)
            Var_max = round(np.max(Var_binsHs_temp), 1)
            
            hs_bin_temp = str(int(binsHs[j]))+'-'+str(int(binsHs[j+1]))
            f.write(hs_bin_temp + ' & '+str(len(Var_binsHs_temp))+' & '+str(Var_min)+' & '+str(Var_P5)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_max)+' \\\\' + '\n')
        hs_bin_temp = str(int(binsHs[-1]))+'-'+str(int(binsHs[-1]+1)) # +1 for one empty row 
        f.write(hs_bin_temp + ' & 0 & - & - & - & - & - \\\\' + '\n')
        
        # annual row 
        Var_min = round(np.min(Var), 1)
        Var_P5 = round(np.percentile(Var , 5), 1)
        Var_mean = round(np.mean(Var), 1)
        Var_P95 = round(np.percentile(Var, 95), 1)
        Var_max = round(np.max(Var), 1)
        
        hs_bin_temp = str(int(binsHs[0]))+'-'+str(int(binsHs[-1]+1)) # +1 for one empty row 
        f.write(hs_bin_temp + ' & '+str(len(Var))+' & '+str(Var_min)+' & '+str(Var_P5)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_max)+' \\\\' + '\n')
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')
    
    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)
    

    return    

def table_monthly_percentile(data,var,output_file='var_monthly_percentile.txt'):  
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    this function will sort variable var e.g., hs by month and calculate percentiles 
    data : panda series 
    var  : variable 
    output_file: extension .txt for latex table or .csv for csv table
    """

    Var = data[var]
    varName = data[var].name
    Var_month = Var.index.month
    M = Var_month.values
    temp_file = output_file.split('.')[0]
    
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    monthlyVar = {}
    for i in range(len(months)) : 
        monthlyVar[months[i]] = [] # create empty dictionaries to store data 
    
    
    for i in range(len(Var)) : 
        m_idx = int(M[i]-1) 
        monthlyVar[months[m_idx]].append(Var.iloc[i])  
        
        
    with open(temp_file, 'w') as f:
        f.write('\\begin{tabular}{l | p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm}}' + '\n')
        f.write('& \multicolumn{5}{c}{' + varName + '} \\\\' + '\n')
        f.write('Month & 5\% & 50\% & Mean & 95\% & 99\% \\\\' + '\n')
        f.write('\hline' + '\n')
    
        for j in range(len(months)) : 
            Var_P5 = round(np.percentile(monthlyVar[months[j]],5),1)
            Var_P50 = round(np.percentile(monthlyVar[months[j]],50),1)
            Var_mean = round(np.mean(monthlyVar[months[j]]),1)
            Var_P95 = round(np.percentile(monthlyVar[months[j]],95),1)
            Var_P99 = round(np.percentile(monthlyVar[months[j]],99),1)
            f.write(months[j] + ' & '+str(Var_P5)+' & '+str(Var_P50)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_P99)+' \\\\' + '\n')
        
        # annual row 
        f.write('Annual & '+str(Var_P5)+' & '+str(Var_P50)+' & '+str(Var_mean)+' & '+str(Var_P95)+' & '+str(Var_P99)+' \\\\' + '\n')
    
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')
    
    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)
    
    return   


def table_monthly_min_mean_max(data, var,output_file='montly_min_mean_max.txt') :  
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    It calculates monthly min, mean, max based on monthly maxima. 
    data : panda series
    var  : variable 
    output_file: extension .txt for latex table or .csv for csv table
    """
        
    var = data[var]
    temp_file  = output_file.split('.')[0]
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    monthly_max = var.resample('M').max() # max in every month, all months 
    minimum = monthly_max.groupby(monthly_max.index.month).min() # min, sort by month
    mean = monthly_max.groupby(monthly_max.index.month).mean() # mean, sort by month
    maximum = monthly_max.groupby(monthly_max.index.month).max() # max, sort by month
    
    with open(temp_file, 'w') as f :
        f.write('\\begin{tabular}{l | c c c }' + '\n')
        f.write('Month & Minimum & Mean & Maximum \\\\' + '\n')
        f.write('\hline' + '\n')
        for i in range(len(months)):
            f.write(months[i] + ' & ' + str(minimum.values[i]) + ' & ' + str(round(mean.values[i],1)) + ' & ' + str(maximum.values[i]) + ' \\\\' + '\n')
        
        ## annual row 
        annual_max = var.resample('Y').max()
        min_year = annual_max.min()
        mean_year = annual_max.mean()
        max_year = annual_max.max()
        f.write('Annual Max. & ' + str(min_year) + ' & ' + str(round(mean_year,1)) + ' & ' + str(max_year) + ' \\\\' + '\n')
            
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')


    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)

    return


def RVE_of_EXP_GEV_GUM_LoNo(df,period,distribution='EXP',method='default',threshold='default'):
    
    # df : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # distribution: 'EXP', 'GEV', 'GUM' and 'LoNo'
    # method: 'default', 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 
    
    import scipy.stats as stats
    from pyextremes import get_extremes
    
    # get data for fitting 
    if method == 'default' : # all data 
        data = df.values
    elif method == 'AM' : # annual maxima
        annual_maxima = df.resample('Y').max() # get annual maximum 
        data = annual_maxima
    elif method == 'POT' : # Peak over threshold 
        if threshold == 'default' :
            annual_maxima = df.resample('Y').max() 
            threshold=annual_maxima.min()
        data = get_extremes(df, method="POT", threshold=threshold, r="48H")
    else:
        print ('Please check the method of filtering data')
    
    # Return periods in K-th element 
    try:
        for i in range(len(period)) :
            if period[i] == 1 : 
                period[i] = 1.5873
    except:
        if period == 1 : 
            period = 1.5873
            
    duration = (df.index[-1]-df.index[0]).days + 1 
    length_data = data.shape[0]
    interval = duration*24/length_data # in hours 
    period = period*365.2422*24/interval # years is converted to K-th
    
    # Fit a distribution to the data
    if distribution == 'EXP' : 
        from scipy.stats import expon
        loc, scale = expon.fit(data)
        value = expon.isf(1/period, loc, scale)
        #value = expon.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'GEV' :
        from scipy.stats import genextreme
        shape, loc, scale = genextreme.fit(data) # fit data   
        value = genextreme.isf(1/period, shape, loc, scale)
        #value = genextreme.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'GUM' :
        from scipy.stats import gumbel_r 
        loc, scale = gumbel_r.fit(data) # fit data
        value = gumbel_r.isf(1/period, loc, scale)
        #value = gumbel_r.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'LoNo' :
        from scipy.stats import lognorm 
        shape, loc, scale = lognorm.fit(data)
        value = lognorm.isf(1/period, shape, loc, scale)
        #value = lognorm.ppf(1 - 1 / period, shape, loc, scale)
    else:
        print ('Please check the distribution')    
    
    return value  



def RVE_Weibull(df,period,method_weibull='3P',method_data='default',threshold='default'):
    # df : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # method_weibull: '2P', '3P'
    # method_data: 'default', 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 
    
    import scipy.stats as stats
    from pyextremes import get_extremes
    
    # get data for fitting 
    if method_data == 'default' : # all data 
        data = df.values
    elif method_data == 'AM' : # annual maxima
        annual_maxima = df.resample('Y').max() # get annual maximum 
        data = annual_maxima
    elif method_data == 'POT' : # Peak over threshold 
        if threshold == 'default' :
            annual_maxima = df.resample('Y').max() 
            threshold=annual_maxima.min()
        data = get_extremes(df, method="POT", threshold=threshold, r="48H")
    else:
        print ('Please check the method of filtering data')
    
    # Return periods in K-th element 
    try:
        for i in range(len(period)) :
            if period[i] == 1 : 
                period[i] = 1.5873
    except:
        if period == 1 : 
            period = 1.5873
    
    duration = (df.index[-1]-df.index[0]).days + 1 
    length_data = data.shape[0]
    interval = duration*24/length_data # in hours 
    period = period*365.2422*24/interval # years is converted to K-th
    
    # Fit a Weibull distribution to the data
    if method_weibull == '3P' : 
        shape, loc, scale = stats.weibull_min.fit(data) # (ML)
    elif method_weibull == '2P' :
        shape, loc, scale = stats.weibull_min.fit(data, floc=0) # (ML)
    else:
        print ('Please the Weibull distribution must be 2P or 3P')    
        
    #value = stats.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    value = stats.weibull_min.isf(1/period, shape, loc, scale)
    
    return value

def readNora10File(file):
      
    df = pd.read_csv(file, delim_whitespace=True, header=3) # sep=' ', header=None,0,1,2,3
    df.index= pd.to_datetime(df.YEAR*1000000+df.M*10000+df.D*100+df.H,format='%Y%m%d%H')
    df['hs'] = df['HS']
    df['tp'] = Tp_correction(df.TP.values)
    df['w10'] = df['W10']
    df['d10'] = df['D10'] 
    df2 = df.filter(['hs', 'tp','w10','d10']) # ['HS', 'TP','W10','D10'], axis=1
    
    #df2 = df.filter(['HS', 'TP','W10','D10']) # , axis=1
    
    #df2=df
    
    return df2


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
from scipy.stats import weibull_min
import scipy.io as io
import seaborn as sns
import scipy.stats as stats
from pyextremes import get_extremes
from scipy.stats import expon
from scipy.io import savemat
from datetime import datetime, timedelta
from matplotlib.font_manager import FontProperties
from inspect import getsource as source # print(source(readNora10File))
from scipy.signal import find_peaks # index = find_peaks(a) # a[index[0]]



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










###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
def RVE_Weibull(df,period,method_weibull='3P',method_data='default',threshold='default'):
    # df : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # method_weibull: '2P', '3P'
    # method_data: 'default', 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 
    
    import scipy.stats as stats
    from pyextremes import get_extremes
    
    # get data for fitting 
    if method_data == 'default' : # all data 
        data = df.values
    elif method_data == 'AM' : # annual maxima
        annual_maxima = df.resample('Y').max() # get annual maximum 
        data = annual_maxima
    elif method_data == 'POT' : # Peak over threshold 
        if threshold == 'default' :
            annual_maxima = df.resample('Y').max() 
            threshold=annual_maxima.min()
        data = get_extremes(df, method="POT", threshold=threshold, r="48H")
    else:
        print ('Please check the method of filtering data')
    
    # Return periods in K-th element 
    # Return periods in K-th element 
    try:
        for i in range(len(period)) :
            if period[i] == 1 : 
                period[i] = 1.5873
    except:
        if period == 1 : 
            period = 1.5873
    
    duration = (df.index[-1]-df.index[0]).days + 1 
    length_data = data.shape[0]
    interval = duration*24/length_data # in hours 
    period = period*365.2422*24/interval # years is converted to K-th
    
    # Fit a Weibull distribution to the data
    if method_weibull == '3P' : 
        shape, loc, scale = stats.weibull_min.fit(data) # (ML)
    elif method_weibull == '2P' :
        shape, loc, scale = stats.weibull_min.fit(data, floc=0) # (ML)
    else:
        print ('Please the Weibull distribution must be 2P or 3P')    
        
    #value = stats.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    value = stats.weibull_min.isf(1/period, shape, loc, scale)
    
    return value




def RVE_of_EXP_GEV_GUM_LoNo(df,period,distribution='EXP',method='default',threshold='default'):
    
    # df : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # distribution: 'EXP', 'GEV', 'GUM' and 'LoNo'
    # method: 'default', 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 
    
    import scipy.stats as stats
    from pyextremes import get_extremes
    
    # get data for fitting 
    if method == 'default' : # all data 
        data = df.values
    elif method == 'AM' : # annual maxima
        annual_maxima = df.resample('Y').max() # get annual maximum 
        data = annual_maxima
    elif method == 'POT' : # Peak over threshold 
        if threshold == 'default' :
            annual_maxima = df.resample('Y').max() 
            threshold=annual_maxima.min()
        data = get_extremes(df, method="POT", threshold=threshold, r="48H")
    else:
        print ('Please check the method of filtering data')
    
    # Return periods in K-th element 
    try:
        for i in range(len(period)) :
            if period[i] == 1 : 
                period[i] = 1.5873
    except:
        if period == 1 : 
            period = 1.5873
            
    duration = (df.index[-1]-df.index[0]).days + 1 
    length_data = data.shape[0]
    interval = duration*24/length_data # in hours 
    period = period*365.2422*24/interval # years is converted to K-th
    
    # Fit a distribution to the data
    if distribution == 'EXP' : 
        from scipy.stats import expon
        loc, scale = expon.fit(data)
        value = expon.isf(1/period, loc, scale)
        #value = expon.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'GEV' :
        from scipy.stats import genextreme
        shape, loc, scale = genextreme.fit(data) # fit data   
        value = genextreme.isf(1/period, shape, loc, scale)
        #value = genextreme.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'GUM' :
        from scipy.stats import gumbel_r 
        loc, scale = gumbel_r.fit(data) # fit data
        value = gumbel_r.isf(1/period, loc, scale)
        #value = gumbel_r.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'LoNo' :
        from scipy.stats import lognorm 
        shape, loc, scale = lognorm.fit(data)
        value = lognorm.isf(1/period, shape, loc, scale)
        #value = lognorm.ppf(1 - 1 / period, shape, loc, scale)
    else:
        print ('Please check the distribution')    
    
    return value



def RVE_ALL(df,period=100,distribution='Weibull3P',method='default',threshold='default'):
    
    # df : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # distribution: 'EXP', 'GEV', 'GUM', 'LoNo', 'Weibull2P' or 'Weibull3P'
    # method: 'default', 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 
    
    import scipy.stats as stats
    from pyextremes import get_extremes
    
    # get data for fitting 
    if method == 'default' : # all data 
        data = df.values
    elif method == 'AM' : # annual maxima
        annual_maxima = df.resample('Y').max() # get annual maximum 
        data = annual_maxima
    elif method == 'POT' : # Peak over threshold 
        if threshold == 'default' :
            annual_maxima = df.resample('Y').max() 
            threshold=annual_maxima.min()
        data = get_extremes(df, method="POT", threshold=threshold, r="48H")
    else:
        print ('Please check the method of filtering data')
    
    # Return periods in K-th element 
    try:
        for i in range(len(period)) :
            if period[i] == 1 : 
                period[i] = 1.5873
    except:
        if period == 1 : 
            period = 1.5873
            
    duration = (df.index[-1]-df.index[0]).days + 1 
    length_data = data.shape[0]
    interval = duration*24/length_data # in hours 
    period = period*365.2422*24/interval # years is converted to K-th
    
    # Fit a distribution to the data
    if distribution == 'EXP' : 
        loc, scale = stats.expon.fit(data)
        value = stats.expon.isf(1/period, loc, scale)
        #value = stats.expon.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'GEV' :
        shape, loc, scale = stats.genextreme.fit(data) # fit data   
        value = stats.genextreme.isf(1/period, shape, loc, scale)
        #value = stats.genextreme.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'GUM' :
        loc, scale = stats.gumbel_r.fit(data) # fit data
        value = stats.gumbel_r.isf(1/period, loc, scale)
        #value = stats.gumbel_r.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'LoNo' :
        shape, loc, scale = stats.lognorm.fit(data)
        value = stats.lognorm.isf(1/period, shape, loc, scale)
        #value = stats.lognorm.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'Weibull2P' :
        shape, loc, scale = stats.weibull_min.fit(data, floc=0) # (ML)
        value = stats.weibull_min.isf(1/period, shape, loc, scale)
        #value = stats.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'Weibull3P' : 
        shape, loc, scale = stats.weibull_min.fit(data) # (ML)
        value = stats.weibull_min.isf(1/period, shape, loc, scale)
        #value = stats.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    else:
        print ('Please check the distribution')    
        
    return value
    
    
    
def pdf_all(data, bins=70):
    
    import matplotlib.pyplot as plt
    from scipy.stats import expon
    from scipy.stats import genextreme
    from scipy.stats import gumbel_r 
    from scipy.stats import lognorm 
    from scipy.stats import weibull_min
    
    
    x=np.linspace(min(data),max(data),bins)
    
    param = weibull_min.fit(data) # shape, loc, scale
    pdf_weibull = weibull_min.pdf(x, param[0], loc=param[1], scale=param[2])
    
    param = expon.fit(data) # loc, scale
    pdf_expon = expon.pdf(x, loc=param[0], scale=param[1])
    
    param = genextreme.fit(data) # shape, loc, scale
    pdf_gev = genextreme.pdf(x, param[0], loc=param[1], scale=param[2])
    
    param = gumbel_r.fit(data) # loc, scale
    pdf_gumbel = gumbel_r.pdf(x, loc=param[0], scale=param[1])
    
    param = lognorm.fit(data) # shape, loc, scale
    pdf_lognorm = lognorm.pdf(x, param[0], loc=param[1], scale=param[2])
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, pdf_expon, label='pdf-expon')
    ax.plot(x, pdf_weibull,label='pdf-Weibull')
    ax.plot(x, pdf_gev, label='pdf-GEV')
    ax.plot(x, pdf_gumbel, label='pdf-GUM')
    ax.plot(x, pdf_lognorm, label='pdf-lognorm')
    ax.hist(data, density=True, bins=bins, color='tab:blue', label='histogram', alpha=0.5)
    #ax.hist(data, density=True, bins='auto', color='tab:blue', label='histogram', alpha=0.5)
    ax.legend()
    ax.grid()
    
    return 





























        
        
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
        
        # sorted by direction, get max in each year, and statistical values 
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
    
    plt.figure(figsize=(12,8))
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
    
    
    
    if len(outfile) > 0 :
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
    
def readNora10File(file):
      
    df = pd.read_csv(file, delim_whitespace=True, header=3) # sep=' ', header=None,0,1,2,3
    df.index= pd.to_datetime(df.YEAR*1000000+df.M*10000+df.D*100+df.H,format='%Y%m%d%H')
    df['hs'] = df['HS']
    df['tp'] = Tp_correction(df.TP.values)
    df['w10'] = df['W10']
    df['d10'] = df['D10'] 
    df2 = df.filter(['hs', 'tp','w10','d10']) # ['HS', 'TP','W10','D10'], axis=1
    
    #df2 = df.filter(['HS', 'TP','W10','D10']) # , axis=1
    
    #df2=df
    
    return df2
    
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
    

def best_fit(data):

    # fit lognormal distribution
    sigma, loc, scale = stats.lognorm.fit(data)
    # evaluate lognormal distribution
    lognormal_test = stats.kstest(data, stats.lognorm.cdf, args=(sigma, loc, scale))
    
    
    # fit gamma distribution
    shape, loc, scale = stats.gamma.fit(data)
    # evaluate gamma distribution
    gamma_test = stats.kstest(data, stats.gamma.cdf, args=(shape, loc, scale))
    
    # fit beta distribution
    a, b, loc, scale = stats.beta.fit(data)
    # evaluate beta distribution
    beta_test = stats.kstest(data, stats.beta.cdf, args=(a, b, loc, scale))
    
    
    if lognormal_test.statistic < gamma_test.statistic and lognormal_test.statistic < beta_test.statistic:
        print("Lognormal distribution is the best fit.")
    elif gamma_test.statistic < lognormal_test.statistic and gamma_test.statistic < beta_test.statistic:
        print("Gamma distribution is the best fit.")
    else:
        print("Beta distribution is the best fit.")
    
    return
    
    
def sort_by_month(df):  # disadvantage of this is not store time 
    # df: dataframe with time index
    # out: list of 12-month dataframe
    
    # extract data  
    time = df.index
    data = df.values
    M = df.index.month.values 
    
    # get monthly names 
    month_name = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(month_name)) : 
        month_name[i] = month_name[i][:3] # get the three first letters 
    
    # create empty dictionaries to store data 
    monthly_t = {}
    monthly_X = {}
    for i in range(len(month_name)) : 
        monthly_t[month_name[i]] = [] 
        monthly_X[month_name[i]] = [] 
    
    # store data
    for i in range(len(M)) : 
        m_idx = int(M[i]-1) 
        monthly_t[month_name[m_idx]].append(time[i])  # fill data 
        monthly_X[month_name[m_idx]].append(data[i])  # fill data 

    lst_out = []
    # convert to dataframe 
    for i in range(len(month_name)) : 
        df_monthly = pd.DataFrame()
        df_monthly.index = monthly_t[month_name[i]]
        df_monthly['data'] = monthly_X[month_name[i]]
        lst_out.append(df_monthly)
        
    return lst_out
    
    
def sort_by_direction(df, direction='direction', intensity='speed'): 
    # this take dataframe and sort by direction, return a dictionary of direction sectors 
    
    # sorted by sectors/directions, keep time for the next part
    bins_dir = np.arange(0,360,30) # 0,30,...,300,330
    time = df.index
    inten = df[intensity] 
    dirt = df[direction] 
    
    dic_inten = {}
    dic_time = {}
    dic_out = {}
    for i in range(len(bins_dir)) : 
        dic_inten[str(int(bins_dir[i]))] = [] 
        dic_time[str(int(bins_dir[i]))] = [] 
        dic_out[str(int(bins_dir[i]))] = [] 
    
    for i in range(len(inten)): 
        if 345 <= dirt[i] : 
            dic_time[str(int(bins_dir[0]))].append(time[i])
            dic_inten[str(int(bins_dir[0]))].append(inten[i]) 
        else: 
            for j in range(len(bins_dir)): 
                if bins_dir[j]-15 <= dirt[i] < bins_dir[j] + 15 : 
                    dic_time[str(int(bins_dir[j]))].append(time[i])
                    dic_inten[str(int(bins_dir[j]))].append(inten[i]) 
                    
    for j in range(len(bins_dir)):
        df_dir = pd.DataFrame()
        df_dir.index = dic_time[str(int(bins_dir[j]))]
        df_dir[intensity] = dic_inten[str(int(bins_dir[j]))]
        dic_out[str(int(bins_dir[j]))].append(df_dir) 
        
    return dic_out
    
    
   
def Hs_Tp_curve(data,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t,X=100):
    # RVE of X years 
    period=X*365.2422*24/3
    shape, loc, scale = weibull_min.fit(data) # shape, loc, scale
    rve_X = stats.weibull_min.isf(1/period, shape, loc, scale)
    #print (value)
    
    # Find index of Hs=value
    epsilon = abs(h - rve_X)
    param = find_peaks(1/epsilon) # to find the index of bottom
    index = param[0][0]     # the  index of Hs=value
    #print (index)
    
    # Find peak of pdf at Hs=RVE of X year 
    pdf_Hs_Tp_X = pdf_Hs_Tp[index,:] # Find pdf at RVE of X year 
    param = find_peaks(pdf_Hs_Tp_X) # find the peak
    index = param[0][0]
    f_Hs_Tp_100=pdf_Hs_Tp_X[index]
    
    #f_Hs_Tp_100 = weibull_min.pdf(value, shape, loc=0, scale=scale)
    #f_Hs_Tp_100=1/(100*365.2422*24/3)
    
    h1=[]
    t1=[]
    t2=[]
    for i in range(len(h)):
        f3_ = f_Hs_Tp_100/pdf_Hs[i]
        f3 = f_Hs_Tp[i,:]
        epsilon = abs(f3-f3_) # the difference 
        para = find_peaks(1/epsilon) # to find the bottom
        index = para[0]
        if t[index].shape[0] == 2 :
            h1.append(h[i])
            t1.append(t[index][0])
            t2.append(t[index][1])
    
    h1=np.asarray(h1)
    t1=np.asarray(t1)
    t2=np.asarray(t2)
    t3 = np.concatenate((t1, t2[::-1])) # to get correct circle order 
    h3 = np.concatenate((h1, h1[::-1])) # to get correct circle order 
    t3 = np.concatenate((t3, t1[0:1])) # connect the last to the first point  
    h3 = np.concatenate((h3, h1[0:1])) # connect the last to the first point  

    return t3,h3,X
    
    
def RVE_Hs_Tp(file): 
    from scipy.stats import lognorm, weibull_min
    
    df = readNora10File(file)
    
    # calculate lognormal and weibull parameters and plot the PDFs 
    mu = np.mean(np.log(df.hs.values)) # mean of ln(Hs)
    std = np.std(np.log(df.hs.values)) # standard deviation of ln(Hs)
    alpha = mu
    sigma = std
    
    h = np.linspace(start=0.001, stop=20, num=2000)
    pdf_Hs1 = h*0
    pdf_Hs2 = h*0
    
    pdf_Hs1 = 1/(np.sqrt(2*np.pi)*alpha*h)*np.exp(-(np.log(h)-sigma)**2/(2*alpha**2))
    param = weibull_min.fit(df.hs.values) # shape, loc, scale
    pdf_Hs2 = weibull_min.pdf(h, param[0], loc=param[1], scale=param[2])
    
    
    # Find the index where two PDF cut, between P60 and P99 
    for i in range(len(h)):
        if abs(h[i]-np.percentile(df.hs.values,60)) < 0.1:
            i1=i
            
        if abs(h[i]-np.percentile(df.hs.values,99)) < 0.1:
            i2=i
            
    epsilon=abs(pdf_Hs1[i1:i2]-pdf_Hs2[i1:i2])
    param = find_peaks(1/epsilon)
    index = param[0][0]
    index = index + i1
    
    
    # Merge two functions and do smoothing around the cut 
    eta = h[index]
    pdf_Hs = h*0
    for i in range(len(h)):
        if h[i] < eta : 
            pdf_Hs[i] = pdf_Hs1[i]
        else:
            pdf_Hs[i] = pdf_Hs2[i]
            
    for i in range(len(h)):
        if eta-0.5 < h[i] < eta+0.5 : 
            pdf_Hs[i] = np.mean(pdf_Hs[i-10:i+10])
    
            
    #####################################################
    # calcualte a1, a2, a3, b1, b2, b3 
    # firstly calcualte mean_hs, mean_lnTp, variance_lnTp 
    Tp = df.tp.values
    Hs = df.hs.values
    maxHs = max(Hs)
    if maxHs<2 : 
        intx=0.05
    elif maxHs>=2 and maxHs<3 :
        intx=0.1
    elif maxHs>=3 and maxHs<4 :
        intx=0.2
    elif maxHs>=4 and maxHs<10 :
        intx=0.5
    else : 
        intx=1.0;
    
    mean_hs = []
    variance_lnTp = []
    mean_lnTp = []
    
    hs_bin = np.arange(0,maxHs+intx,intx)
    for i in range(len(hs_bin)-1):
        idxs = np.where((hs_bin[i]<=Hs) & (Hs<hs_bin[i+1]))
        if Hs[idxs].shape[0] > 15 : 
            mean_hs.append(np.mean(Hs[idxs]))
            mean_lnTp.append(np.mean(np.log(Tp[idxs])))
            variance_lnTp.append(np.var(np.log(Tp[idxs])))
    
    mean_hs = np.asarray(mean_hs)
    mean_lnTp = np.asarray(mean_lnTp)
    variance_lnTp = np.asarray(variance_lnTp)
    
    # calcualte a1, a2, a3
    from scipy.optimize import curve_fit
    def Gauss3(x, a1, a2):
        y = a1 + a2*x**0.36
        return y
    parameters, covariance = curve_fit(Gauss3, mean_hs, mean_lnTp)
    a1 = parameters[0]
    a2 = parameters[1]
    a3 = 0.36
    
    # calcualte b1, b2, b3 
    def Gauss4(x, b2, b3):
        y = 0.005 + b2*np.exp(-x*b3)
        return y
    start = 1
    x = mean_hs[start:]
    y = variance_lnTp[start:]
    parameters, covariance = curve_fit(Gauss4, x, y)
    b1 = 0.005
    b2 = parameters[0]
    b3 = parameters[1]
    
    
    # calculate pdf Hs, Tp 
    t = np.linspace(start=0.001, stop=35, num=3000)
    
    f_Hs_Tp = np.zeros((len(h), len(t)))
    pdf_Hs_Tp=f_Hs_Tp*0
    
    for i in range(len(h)):
        mu = a1 + a2*h[i]**a3
        std2 = b1 + b2*np.exp(-b3*h[i])
        std = np.sqrt(std2)
        
        f_Hs_Tp[i,:] = 1/(np.sqrt(2*np.pi)*std*t)*np.exp(-(np.log(t)-mu)**2/(2*std2))
        pdf_Hs_Tp[i,:] = pdf_Hs[i]*f_Hs_Tp[i,:]
        
        
        
    ## Plot data 
    param1 = Hs_Tp_curve(df.hs.values,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t,X=1)
    param50 = Hs_Tp_curve(df.hs.values,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t,X=50)
    param100 = Hs_Tp_curve(df.hs.values,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t)
    param500 = Hs_Tp_curve(df.hs.values,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t,X=500)


    plt.figure(figsize=(8,6))
    plt.plot(param1[0],param1[1],'b',label=str(param1[2])+'-year')
    plt.plot(param50[0],param50[1],'y',label=str(param50[2])+'-year')
    plt.plot(param100[0],param100[1],'b',label=str(param100[2])+'-year')
    plt.plot(param500[0],param500[1],'r',label=str(param500[2])+'-year')
    plt.scatter(df.tp.values,df.hs.values,s=5)
    plt.xlabel('Tp - Peak Period[s]')
    plt.ylabel('Hs - Significant Wave Height[m]')
    plt.grid()
    plt.legend() 
    
