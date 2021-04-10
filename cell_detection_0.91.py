import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import sqlite3
import numpy as np
import itertools
from datetime import datetime
import argparse
import os
import sys
import argparse

def create_graph(df,fn="Zelldiagramm", customer="Generic", show=True, sdev = 4, df_statistics = [], excel=False, dir="./data/", thick = 1):
    fig, axs = plt.subplots(len(df)-1,2, figsize=(16, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .4, wspace=.1)
    axs = axs.ravel()
    n = -1 # skip last row with count for x-axis
    half = (len(df))/2 # determines what is battery data first half, outlier second half preparing subplot title
    for i in range(len(df)):
        for k in df[i].columns[:n]: # n shows if last column must be skipped or not, due to preparedf dataframes or anomalie
            axs[i].plot(df[i][k].index,df[i][str(k)], label = str(k), linewidth= thick )
        if i < half:
            axs[i].set_title("battery " +str(i))
            if excel:
                df[i].to_excel(dir +customer+"_battery_" +str(i)+ ".xlsx")
        else:
            axs[i].set_title("battery " +str(i % 2) + " " + "anomalie with sd = " +str(sdev))
            if excel:
                df[i].to_excel(dir + customer+"_battery_" +str(i % 2)+ "_anomalie_with_sd_" +str(sdev)+ ".xlsx")
        if len(df[i].columns) < 30:
            axs[i].legend(loc="upper right", title="Cell(s) to examine ", bbox_to_anchor=(1, 1), fontsize = 5)
    cell_text = []
    for row in range(len(df_statistics)):
        cell_text.append(df_statistics.iloc[row])
    column_labels = df_statistics.columns
    axs[i+1].axis('tight')
    axs[i+1].axis('off')
    axs[i+1].table(cellText=cell_text,colLabels=column_labels,loc="center",cellLoc ='left', colLoc='left')
    if i % 2 == 1:
        fig.delaxes(axs[i+2]) # remove empty drawing if printet figures are uneven
    plt.suptitle(customer,fontsize=20)
    plt.savefig(dir + fn+ ".pdf")
    if show:
        plt.show()
    if excel:
        df_statistics.to_excel(dir + customer + "_stats.xlsx", index=False)


def prepare_df(df):
    df["date_time"] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = (df.reset_index().pivot(index='date_time', columns='UID', values='U_V'))
    df.reset_index(drop=True, inplace=True)
    df['mean'] = df.iloc[:, 0:len(df.columns)].mean(axis=1) 
    df['date_time'] = df.index
    df.columns = df.columns.astype(str)
    #print(len(df.columns))
    return df

def read_db(db):
    #con = sqlite3.connect(db)
    #j = 0 # counter for battery numbers
    ldf = []

    try:
        con = sqlite3.connect(db)
    except sqlite3.Error as e:
        print(e)
    cur = con.cursor()
    cur.execute("SELECT distinct battery FROM scl_data")
    rows = cur.fetchall()  
    print("sql1")
    cur.execute("""
        SELECT 
            Battery,
            min(rt.date || " " || rt.time),
            max(rt.date || " " || rt.time),
            count( *),
            count( distinct Addr)         
        FROM   record_time rt
        INNER JOIN scl_data sd 
                    ON rt.record = sd.record 
        GROUP BY sd.Battery	   
    """)
    rows2 =cur.fetchall()
    print("sql2")
    for count, value  in enumerate(rows):
        df = pd.read_sql_query("""SELECT rt.date, 
        rt.time, 
        sd.battery, 
        sd.addr, 
        sd.u_v, 
        su.uid
        FROM   record_time rt 
        INNER JOIN scl_data sd 
                ON rt.record = sd.record 
        INNER JOIN scl_uid su 
                ON su.battery = sd.battery 
                    AND su.addr = sd.addr 
        WHERE  sd.u_v > 0 and su.battery = """ +str(count) ,   con)
        #j += 1
        ldf.append(df)
    print("sql3")
    return ldf,rows2 #rows2 = min max datetime per battery

def find_outliers(dfT,sdev=4):
    res = []
    for i in dfT.columns:
        ana = find_anomalies(dfT[i],sdev)
        if len(ana)>0:
            for j in ana:
                res.append(dfT.index[dfT[str(i)] == j].tolist())   # check index -> should be UID
    return res 


def find_anomalies(data, sdev=4):
    #define a list to accumlate anomalies
    anomalies = []
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(data,ddof = 1) # 
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * sdev 
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

def ret_anomalies_as_list_index(df,sdev=4):
    dfT = df.iloc[:,0:-2] # copy
    dfT = dfT.T
    dfT['mean'] = dfT.mean(axis=1)
    dfT.columns = dfT.columns.astype(str)
    dfret = find_outliers(dfT,sdev) 
    return dfret

def create_data_lists(ldf,sdev):
    for j in range(len(ldf)):
        ldft.append(prepare_df(ldf[j]))
    #ldft[0].to_excel("create_data")
    for j in range(len(ldf)):
        test = ret_anomalies_as_list_index(ldft[j],sdev)
        #print(test)
        test = list(set(list(itertools.chain(*test)))) 
        test.append('mean') # help columns to have same format as main dataframes with these columns
        test.append('date_time')
        ldft.append(ldft[j][test]) # filter dataframe with list
    return ldft

# def create_statistics(ldft,ldf, min_max, sdev=4, customer = 'Generic'):
#     lcols = [] # columns with UID
#     batt_list = []
#     kind_list = []
#     value_list = [] 

#     df_half = int(len(ldft)/2) # second half is outlier
#     for i in range(df_half, len(ldft)): #get outlier uids from columnnames
#         lcols.extend(ldft[i].columns)

#     lcols = list(set(lcols))
#     lcols.remove("mean")
#     lcols.remove("date_time")

    
#     date_list = [item for t in min_max for item in t]
    
#     kind_list.append("Customer")
#     value_list.append(customer)
#     kind_list.append("Creation date")
#     value_list.append(date_list[1][:10])
#     batt_list.append("-")
#     kind_list.append("Finish date")
#     value_list.append(date_list[2][:10])
#     batt_list.append("-")
#     kind_list.append("Start time")
#     value_list.append(date_list[1][11:])
#    # batt_list.append("-")
#     kind_list.append("Stop time")
#     value_list.append(date_list[2][11:])
#     dt_start = datetime. strptime(date_list[1], '%d.%m.%Y %H:%M:%S')
#     dt_end = datetime. strptime(date_list[2], '%d.%m.%Y %H:%M:%S')
#     dt_diff = dt_end - dt_start  
#     minutes = divmod(dt_diff.total_seconds(), 60) 
#     #batt_list.append("-")
#     kind_list.append("Duration minutes:")
#     value_list.append(str(minutes[0]))
#     kind_list.append("Duration seconds:")
#     value_list.append(str(minutes[1]))
#     #batt_list.append("-")
#     kind_list.append("Cells per battery")
#     value_list.append(date_list[4])
#     #batt_list.append("-")
#     kind_list.append("# Measurements")
#     value_list.append(date_list[3])
#     kind_list.append("Threshold anomalie detection with sd")
#     value_list.append(sdev)
#     if len(lcols) == 0:
#         kind_list.append("Anomalies")
#         value_list.append(0)
#     else:
#         for i in sorted(lcols):
#             kind_list.append("Detected UID")
#             value_list.append(i)

#     df_statistics = pd.DataFrame(list(zip(kind_list, value_list)), 
#                columns =[ 'Name', "Value"]) 
    
#     return df_statistics
def create_statistics(ldft, min_max, sdev=4, customer = 'Generic'):
    
    #lcols = [] # columns with UID
    batt_list = []
    kind_list = []
    value_list = [] 
    x = {}
    df_half = int(len(ldft)/2) # second half is outlier
    #for i in range(df_half, len(ldft)):
    #    lcols.extend(ldft[i].columns)
    x = { "Battery " + str(i%2): ldft[i].columns.to_list() for i in range(df_half, len(ldft)) }
    #print(x)
    # lcols = list(set(lcols))
    # lcols.remove("mean")
    # lcols.remove("date_time")

    
    date_list = [item for t in min_max for item in t]
    
    kind_list.append("Customer")
    value_list.append(customer)
    kind_list.append("Creation date")
    value_list.append(date_list[1][:10])
    batt_list.append("-")
    kind_list.append("Finish date")
    value_list.append(date_list[2][:10])
    batt_list.append("-")
    kind_list.append("Start time")
    value_list.append(date_list[1][11:])
   # batt_list.append("-")
    kind_list.append("Stop time")
    value_list.append(date_list[2][11:])
    dt_start = datetime. strptime(date_list[1], '%d.%m.%Y %H:%M:%S')
    dt_end = datetime. strptime(date_list[2], '%d.%m.%Y %H:%M:%S')
    dt_diff = dt_end - dt_start  
    minutes = divmod(dt_diff.total_seconds(), 60) 
    #batt_list.append("-")
    kind_list.append("Duration [min:s]:")
    value_list.append(str(int(minutes[0]))+":"+str(int(minutes[1])))
    #batt_list.append("-")
    kind_list.append("Cells per battery")
    value_list.append(date_list[4])
    #batt_list.append("-")
    kind_list.append("# Measurements")
    value_list.append(date_list[3])
    kind_list.append("Threshold sd")
    value_list.append(sdev)
    if len(x) == 0:
        kind_list.append("Anomalies")
        value_list.append(0)
    else:
      kind_list.append("Detected UIDs")
      value_list.append(" ")
      for key in x:
          for i in x[key][:-2]: # skip mean and date
            kind_list.append(key)
            value_list.append(i)

    df_statistics = pd.DataFrame(list(zip(kind_list, value_list)), 
               columns =[ 'Name', "Value"]) 
    
    return df_statistics

def create_data_dir(directory="./data"):
    if not os.path.exists(directory):
        os.makedirs(directory)
 ########## end functions ##############  

parser = argparse.ArgumentParser()
parser.add_argument('db_path_name', type=str,help='Path including file to sqlite db')
parser.add_argument('--excel', '-e', action='store_true', help="if set statistics, all cells and detected cells are exported per battery bank ")
parser.add_argument('--customer', '-c', type=str, help="sets customer, if omitted generic")
parser.add_argument('--pdfn', '-p', type=str, help="sets pdf output name, if omitted Zelldiagramm")
parser.add_argument('--output', '-o', action='store_true', help="if set output pdf will be directly invoked")
parser.add_argument('--sdev', '-s', type=int, help="sets pstandard deviation, if omitted sdvev = 5")
parser.add_argument('--dir', '-d', type=str, help="sets output directory, if omitted ./data (created if not excistent)")
parser.add_argument('--thick', '-t', type=str, help="set line thickness of plot, if omitted = 1")
args = parser.parse_args()
path = args.db_path_name
excel_output = args.excel
show_output =  args.output
if args.customer:
    customer = args.customer
else:
     customer = "generic"
if args.pdfn:
    pdf_name = args.pdfn
else:
     pdf_name = "Zelldiagrammn"
if args.sdev:
    sdev = args.sdev
else:
     sdev = 5
if args.dir:
    output_dir = args.dir
else:
    output_dir = "./tmp/"
if args.thick:
    thickness = args.thick
else:
    thickness = 1
create_data_dir(directory=output_dir)
create_data_dir(directory='./data/')
ldft = []
ldf, rows = read_db(path)
ldft = create_data_lists(ldf,sdev)
#show_output = True
df_statistics = create_statistics(ldft,rows, sdev, customer)
customer = customer + " "+ str(df_statistics.iloc[1]["Value"]) 
if pdf_name == "Zelldiagrammn":
    pdf_name = pdf_name +" " + customer 
#create_graph(ldft,"Zelldiagramm","Salzatours 16.3",True,sdev, df_statistics, True)

create_graph(df=ldft,fn=pdf_name, customer=customer, show=show_output, sdev = sdev, df_statistics = df_statistics, excel=excel_output, dir=output_dir, thick = thickness)
