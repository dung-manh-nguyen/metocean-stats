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
