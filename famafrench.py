时间范围：2017年5月至2018年4月
范围分割：我们将上述时间范围划分为四个阶段。第一阶段为2017年5月至2017年7月，第二阶段为2017年8月至2017年10月，第三阶段为2017年11月至2018年1月，第四阶段为2018年2月至2018年4月
基金名称：中欧时代先锋A（001938.OF）
股票池：中证800指数成分股（000906.SH）
无风险利率（Rf）:中国境内商业银行一年期整存整取日利率：1.75%÷12÷30
市场收益率（Rm）:中证800指数日收益率

import pandas as pd
import numpy as np
#import pyodbc
import datetime
import datetime
import dateutil

from dateutil.parser import parse
import os,pymssql
import re

#链接SQL Sever
def get_data(sql1):
    server=XXXX
    user=XXXX
    password=XXXX
    conn=pymssql.connect(server,user,password,database="XXXX",charset='utf8')
    cursor=conn.cursor()
    cursor.execute(sql1)
    row=cursor.fetchall()
    conn.close()
    data =pd.DataFrame(row,columns=zip(*cursor.description)[0])
    data = l2gbyR(data)
    return data

def latin2gbk(s):
    if type(s)==unicode:
        s = s.encode('latin1').decode('gbk')
    elif s is None:
        s = np.nan
    return s

def l2gbyR(data):
    for i in data.columns:    
        try:
            data[i] = data[i].apply(lambda s: latin2gbk(s))
        except:
            continue
    return data
def re_s(s,strr):
    pattern = re.compile(strr)
    m = pattern.search(s)
    if m is not None:
        return m.group()
    else:
        return np.nan
    
def decimal_to_float(df,s):
    for i in s:
        df[i] = df[i].apply(lambda s:float(s))
    return df
    
    
from  datetime  import  * 
import time

#获取基金数据
Rf=1.75/12/30
dataall=pd.read_excel('001938All.xlsx')
dataall['TradingDate']=pd.to_datetime(dataall['TradingDate'])

#获取指数（中证800）数据
index_data=get_data("SELECT c.SecuCode,c.SecuAbbr\
                    FROM LC_IndexComponentsWeight a, SecuMain b,SecuMain c\
                    WHERE a.EndDate='2017-04-28' and a.IndexCode=b.InnerCode\
                    and b.SecuCode='000906' and a.InnerCode=c.InnerCode\
                    and b.SecuCategory=4")
                    
seculist=index_data['SecuCode'].tolist()
data_all=pd.DataFrame()
for i in range(800):
    code=seculist[i]
    data=get_data("SELECT a.SecuCode,b.PB,c.TotalMV \
                  FROM SecuMain a,LC_IndicesForValuation b \
                  LEFT join QT_Performance c \
                  on b.InnerCode=c.InnerCode \
                  where a.InnerCode=b.InnerCode and a.SecuCode='"+
                  code+"'and b.EndDate='2016-12-31' \
                  and c.TradingDay=(select MAX(TradingDay) \
                  from QT_Performance where TradingDay<'2017-05-01')")
    
    data_all=data_all.append(data)
    
data_all=data_all.reset_index(drop=True)

data_all['TotalMV']=data_all['TotalMV'].astype('float')
data_all['1/PB']=1/data_all['PB']  

#对股票池数据进行分组
#按市值大小分为两组，存为列表
ME50=np.percentile(data_all['TotalMV'],50)
S=data_all[data_all['TotalMV']<=ME50]['SecuCode'].tolist()                                
B=data_all[data_all['TotalMV']>ME50]['SecuCode'].tolist()

#按1/PB大小分为三组
data_all['1/PB']=data_all['1/PB'].astype('float')
BP30=np.percentile(data_all['1/PB'],30)
BP70=np.percentile(data_all['1/PB'],70)
L=data_all[data_all['1/PB']<=BP30]['SecuCode','TotalMV'].tolist()                                     
H=data_all[data_all['1/PB']>BP70]['SecuCode'].tolist()
M=list(set(data_all['SecuCode'].tolist()).difference(set(L+H)))

#两两取交集，生成股票组合列表
SL=list(set(S).intersection(set(L)))                                       
SM=list(set(S).intersection(set(M)))
SH=list(set(S).intersection(set(H)))
BL=list(set(B).intersection(set(L)))
BM=list(set(B).intersection(set(M)))
BH=list(set(B).intersection(set(H)))

#获取2017-05-01至2018-04-30的交易日
df_tradingday=get_data("select TradingDay from QT_Performance\ 
                       where TradingDay<'2018-05-01' and \
                       TradingDay>'2017-04-30' and InnerCode=1120")
                       
#计算各个组别的收益率                       
#取SL组收益率数据
df_SL_all=pd.DataFrame({'TradingDay':pd.Series(),'DailyReturn_SL':pd.Series()})


for day in df_tradingday['TradingDay']:
    df_SL=pd.DataFrame()
    day=str(day)
    
    codes = [str(s) for s in SL]
  
        
    data=get_data("select a.TradingDay, b.SecuCode, c.TotalMV,a.ChangePCT \
    from QT_Performance a, SecuMain b, QT_Performance c where a.InnerCode=b.InnerCode \
    and c.InnerCode=b.InnerCode and b.SecuCode in {} \
    and a.TradingDay='".format(tuple(codes)) + day +"' and c.TradingDay='2017-04-28'")

    df_SL=df_SL.append(data)
        
        
    df_SL['Weight']=df_SL['TotalMV']/sum(df_SL['TotalMV'])
    DailyReturn_SL=sum(df_SL['Weight']*df_SL['ChangePCT'])
    
    df1=pd.DataFrame({'TradingDay':day,'DailyReturn_SL':DailyReturn_SL},index=[0])
    df_SL_all=df_SL_all.append(df1).reset_index(drop=True)
    
    
    
    
    
    
    
    
    
    
    
#取SM组收益率数据
df_SM_all=pd.DataFrame({'TradingDay':pd.Series(),'DailyReturn_SM':pd.Series()})


for day in df_tradingday['TradingDay']:
    df_SM=pd.DataFrame()
    day=str(day)
    
    codes = [str(s) for s in SM]
  
        
    data=get_data("select a.TradingDay, b.SecuCode, c.TotalMV,a.ChangePCT \
    from QT_Performance a, SecuMain b, QT_Performance c where a.InnerCode=b.InnerCode \
    and c.InnerCode=b.InnerCode and b.SecuCode in {} \
    and a.TradingDay='".format(tuple(codes)) + day +"' and c.TradingDay='2017-04-28'")

    df_SM=df_SM.append(data)
        
        
    df_SM['Weight']=df_SM['TotalMV']/sum(df_SM['TotalMV'])
    DailyReturn_SM=sum(df_SM['Weight']*df_SM['ChangePCT'])
    
    df1=pd.DataFrame({'TradingDay':day,'DailyReturn_SM':DailyReturn_SM},index=[0])
    df_SM_all=df_SM_all.append(df1).reset_index(drop=True)




#取SH组收益率数据
df_SH_all=pd.DataFrame({'TradingDay':pd.Series(),'DailyReturn_SH':pd.Series()})


for day in df_tradingday['TradingDay']:
    df_SH=pd.DataFrame()
    day=str(day)
    
    codes = [str(s) for s in SH]
  
        
    data=get_data("select a.TradingDay, b.SecuCode, c.TotalMV,a.ChangePCT \
    from QT_Performance a, SecuMain b, QT_Performance c where a.InnerCode=b.InnerCode \
    and c.InnerCode=b.InnerCode and b.SecuCode in {} \
    and a.TradingDay='".format(tuple(codes)) + day +"' and c.TradingDay='2017-04-28'")

    df_SH=df_SH.append(data)
        
        
    df_SH['Weight']=df_SH['TotalMV']/sum(df_SH['TotalMV'])
    DailyReturn_SH=sum(df_SH['Weight']*df_SH['ChangePCT'])
    
    df1=pd.DataFrame({'TradingDay':day,'DailyReturn_SH':DailyReturn_SH},index=[0])
    df_SH_all=df_SH_all.append(df1).reset_index(drop=True)



#取BL组收益率数据
df_BL_all=pd.DataFrame({'TradingDay':pd.Series(),'DailyReturn_BL':pd.Series()})


for day in df_tradingday['TradingDay']:
    df_BL=pd.DataFrame()
    day=str(day)
    
    codes = [str(s) for s in BL]
  
        
    data=get_data("select a.TradingDay, b.SecuCode, c.TotalMV,a.ChangePCT \
    from QT_Performance a, SecuMain b, QT_Performance c where a.InnerCode=b.InnerCode \
    and c.InnerCode=b.InnerCode and b.SecuCode in {} \
    and a.TradingDay='".format(tuple(codes)) + day +"' and c.TradingDay='2017-04-28'")

    df_BL=df_BL.append(data)
        
        
    df_BL['Weight']=df_BL['TotalMV']/sum(df_BL['TotalMV'])
    DailyReturn_BL=sum(df_BL['Weight']*df_BL['ChangePCT'])
    
    df1=pd.DataFrame({'TradingDay':day,'DailyReturn_BL':DailyReturn_BL},index=[0])
    df_BL_all=df_BL_all.append(df1).reset_index(drop=True)
    
    
#取BM组收益率数据
df_BM_all=pd.DataFrame({'TradingDay':pd.Series(),'DailyReturn_BM':pd.Series()})


for day in df_tradingday['TradingDay']:
    df_BM=pd.DataFrame()
    day=str(day)
    
    codes = [str(s) for s in BM]
  
        
    data=get_data("select a.TradingDay, b.SecuCode, c.TotalMV,a.ChangePCT \
    from QT_Performance a, SecuMain b, QT_Performance c where a.InnerCode=b.InnerCode \
    and c.InnerCode=b.InnerCode and b.SecuCode in {} \
    and a.TradingDay='".format(tuple(codes)) + day +"' and c.TradingDay='2017-04-28'")

    df_BM=df_BM.append(data)
        
        
    df_BM['Weight']=df_BM['TotalMV']/sum(df_BM['TotalMV'])
    DailyReturn_BM=sum(df_BM['Weight']*df_BM['ChangePCT'])
    
    df1=pd.DataFrame({'TradingDay':day,'DailyReturn_BM':DailyReturn_BM},index=[0])
    df_BM_all=df_BM_all.append(df1).reset_index(drop=True)
    
    
#取BH组收益率数据
df_BH_all=pd.DataFrame({'TradingDay':pd.Series(),'DailyReturn_BH':pd.Series()})


for day in df_tradingday['TradingDay']:
    df_BH=pd.DataFrame()
    day=str(day)
    
    codes = [str(s) for s in BH]
  
        
    data=get_data("select a.TradingDay, b.SecuCode, c.TotalMV,a.ChangePCT \
    from QT_Performance a, SecuMain b, QT_Performance c where a.InnerCode=b.InnerCode \
    and c.InnerCode=b.InnerCode and b.SecuCode in {} \
    and a.TradingDay='".format(tuple(codes)) + day +"' and c.TradingDay='2017-04-28'")

    df_BH=df_BH.append(data)
        
df_factor_all=pd.merge(df_SL_all,df_SM_all,how='left',on='TradingDay')
df_factor_all=pd.merge(df_SL_all,df_SM_all,how='left',on='TradingDay')
df_factor_all=pd.merge(df_factor_all,df_SH_all,how='left',on='TradingDay')
df_factor_all=pd.merge(df_factor_all,df_BL_all,how='left',on='TradingDay')
df_factor_all=pd.merge(df_factor_all,df_BM_all,how='left',on='TradingDay')
df_factor_all=pd.merge(df_factor_all,df_BH_all,how='left',on='TradingDay')


#计算SMB和HML因子，存为list
df_factor_all['SMBr']=(df_factor_all.DailyReturn_SL+df_factor_all.DailyReturn_SM
                       +df_factor_all.DailyReturn_SH)/3-(df_factor_all.DailyReturn_BL+
                                                         df_factor_all.DailyReturn_BM+
                                                         df_factor_all.DailyReturn_BH)/3                         
df_factor_all['HMLr']=(df_factor_all.DailyReturn_SH+df_factor_all.DailyReturn_BH
                      )/2-(df_factor_all.DailyReturn_SL+df_factor_all.DailyReturn_BL)/2 
                      
                      
df_alldata=pd.merge(df_fund,df_factor_all,how='outer',on='TradingDay')

df_alldata.set_index(['TradingDay'],inplace=True)

#构建市场因子（Rm-Rf）
df_mktfactor=get_data("SELECT a.TradingDay, a.ChangePCT FROM QT_IndexQuote a, SecuMain b\ 
                      WHERE a.InnerCode=b.InnerCode and b.SecuCode='000906'and b.SecuCategory=4\
                      and a.TradingDay>'2017-04-30' and a.TradingDay<'2018-05-01'\
                      order by a.TradingDay asc")
df_mktfactor.set_index(['TradingDay'],inplace=True)


df_mktfactor['ChangePCT']=df_mktfactor['ChangePCT'].astype(float)
df_mktfactor['ChangePCT']=df_mktfactor['ChangePCT']-Rf
df_mktfactor=df_mktfactor.rename(columns={'ChangePCT':'Rm_Rf'})
df_alldata=df_alldata.rename(columns={'Return':'Rp_Rf'})

#将所有因子合并为result
result=pd.merge(df_mktfactor,df_alldata,how='outer',left_index=True,right_index=True) 

result_factor=result[['Rm_Rf','SMBr','HMLr']]

#CAPM回归
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

a1=np.zeros(4)   #a项
b1=np.zeros(4)   #市场因子项系数
R2_1=np.zeros(4)   #R2相关系数平方
tb1=np.zeros(4)
ta1=np.zeros(4)
ap1=np.zeros(4)  #a显著性检验的P值，下面类同
bp1=np.zeros(4)


for i in range(4):
    x=result_all[i]['Rm_Rf']
    X=sm.add_constant(x)   #添加常数项
    y=result_all[i]['Rp_Rf']
    model=sm.OLS(y,X)
    results = model.fit()
    a1[i] = results.params[0]
    b1[i] = results.params[1]
    ap1[i]=results.pvalues[0]
    bp1[i]=results.pvalues[1]
    R2_1[i] = results.rsquared
    ta1[i] = results.tvalues[0]
    tb1[i] = results.tvalues[1]
    
result_CAPM=pd.DataFrame({'alpha':a1,'beta':b1,'R2':R2_1})


#famafrench三因子回归
a_fama=np.zeros(4)   #a项
b1_fama=np.zeros(4)   #市场因子项系数
b2_fama=np.zeros(4)
b3_fama=np.zeros(4)
R2_fama=np.zeros(4)   #R2相关系数平方
ta_fama=np.zeros(4)
tb1_fama=np.zeros(4)
tb2_fama=np.zeros(4)
tb3_fama=np.zeros(4)
pa_fama=np.zeros(4)
pb1_fama=np.zeros(4)
pb2_fama=np.zeros(4)
pb3_fama=np.zeros(4)


for i in range(4):
    x=result_all[i][['Rm_Rf','SMBr','HMLr']]

    X=sm.add_constant(x)   #添加常数项
    y=result_all[i]['Rp_Rf']
    model=sm.OLS(y,X)
    results = model.fit()
    a_fama[i] = results.params[0]
    b1_fama[i] = results.params[1]
    b2_fama[i] = results.params[2]
    b3_fama[i] = results.params[3]
    pa_fama[i]=results.pvalues[0]
    pb1_fama[i]=results.pvalues[1]
    pb2_fama[i]=results.pvalues[2]
    pb3_fama[i]=results.pvalues[3]
    R2_fama[i] = results.rsquared
    ta_fama[i] = results.tvalues[0]
    tb1_fama[i] = results.tvalues[1]
    tb2_fama[i] = results.tvalues[2]
    tb3_fama[i] = results.tvalues[3]
    
result_FamaFrench=pd.DataFrame({'alpha':a_fama,'coeff1':b1_fama,'coeff2':b2_fama,'coeff3':b3_fama,'R2':R2_fama})

#计算因子收益
from functools import reduce

for i in range(4):
    result_all[i]['a']=(result_all[i]['Rm_Rf'])/100+1
    mkt[i]=(reduce(lambda x,y:x*y,result_all[i]['a'])-1)*100
    
    result_all[i]['b']=(result_all[i]['SMBr'])/100+1
    size[i]=(reduce(lambda x,y:x*y,result_all[i]['b'])-1)*100
    
    result_all[i]['c']=(result_all[i]['HMLr'])/100+1
    value[i]=(reduce(lambda x,y:x*y,result_all[i]['c'])-1)*100
    
    result_all[i]['d']=(result_all[i]['Rp_Rf'])/100+1
    fund_return[i]=(reduce(lambda x,y:x*y,result_all[i]['d'])-1)*100
    
result_FamaFrench_Return=pd.DataFrame({'mkt_return':mkt,'size_return':size,'value_return':value,'fund_return':fund_return})

#计算各因子贡献

mkt_contribution=np.zeros(4)
size_contribution=np.zeros(4)
value_contribution=np.zeros(4)
ind_contribution=np.zeros(4)

for i in range(4):
    mkt_contribution[i]=result_FamaFrench_Return['mkt_return'][i]*result_FamaFrench['coeff1'][i]
    size_contribution[i]=result_FamaFrench_Return['size_return'][i]*result_FamaFrench['coeff2'][i]
    value_contribution[i]=result_FamaFrench_Return['value_return'][i]*result_FamaFrench['coeff3'][i]
    ind_contribution[i]=result_FamaFrench_Return['fund_return'][i]-mkt_contribution[i]-size_contribution[i]-value_contribution[i]
    
result_contribution=pd.DataFrame({'fund_return':fund_return,'mkt_contribution':mkt_contribution,
                                  'size_contribution':size_contribution,'value_contribution':value_contribution,
                                  'ind_contribution':ind_contribution})

    
