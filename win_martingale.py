import pandas as pd
import numpy as np
import pandas_ta as ta
#import Strategy,Backtest from backtesting
from backtesting import Strategy,Backtest


df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/euro_us.csv')
print(df)


#Convert the 'date' column to datetime if it's not already in that format
df['date'] = pd.to_datetime(df['Local time'])
df['date'] = pd.to_datetime(df['date'], utc=True) # Convert the 'date' column to datetime if it's not already in that format
#print(df)


'''
# Convert the 'date' column to datetime if it's not already in that format
df['date'] = pd.to_datetime(df['Local time'])

#print(df)

# Convert the 'date' column to datetime if it's not already in that format
df['date'] = pd.to_datetime(df['date'], utc=True)
#print(df)

#Extract the year from the 'date' column
df['year'] = df['date'].dt.year
df['month']= df['date'].dt.month
print(df)

df.to_csv('/Users/akashipra/Desktop/Forex/MarketData/2020_TO_2023_5M_yearly.csv', index=False)
'''
# Filter the dataframe for a particular year
'''
'''
#desired_year = 2023
#df = df[df['year'] == desired_year]


#Extract the year from the 'date' column
df['year'] = df['date'].dt.year
df['month']= df['date'].dt.month
print(df)

#Check if NA values are in data
df=df[df['Volume']!=0]
df.reset_index(drop=True, inplace=True)
df.isna().sum()
df.tail()


#BollingerBands Signal function
def indicator(data):
    bbands=ta.bbands(close=data.Close.s,std=0.5)
    #print(bbands.to_numpy)
    return bbands.to_numpy().T[:3]
   

#Extending Strategy Class
class Martingale(Strategy):  

    initsize = 0.005
    count=0

    def takeprofitcalc(self, trades, current_price):

        sum_trade = 0
        # Calculate the new take profit as the average difference 
        # between the entry prices of the existing trades and current market price
        if self.trades[-1].is_long:
            '''for i in range(len(self.trades)):
                sum_trade +=  self.trades[i].entry_price - current_price
            
            avg_sum = sum_trade/len(self.trades)
            take_profit = current_price + avg_sum
            '''

            if len(self.trades)==1:
                take_profit=current_price+4e-4
            elif len(self.trades)==2:
                take_profit=current_price+6e-4
            elif len(self.trades)==3:
                take_profit=current_price+7e-4
            elif len(self.trades)==4:
                take_profit=current_price+8e-4
            elif len(self.trades)==5:
                take_profit=current_price+10e-4
            elif len(self.trades)==6:
                take_profit=current_price+11e-4
            elif len(self.trades)==7:
                take_profit=current_price+12e-4
            elif len(self.trades)==8:
                take_profit=current_price+14e-4
            elif len(self.trades)==9:
                take_profit=current_price+16e-4
            elif len(self.trades)==10:
                take_profit=current_price+17e-4
            elif len(self.trades)==11:
                take_profit=current_price+18-4
            elif len(self.trades)==12:
                take_profit=current_price+19-4
            elif len(self.trades)==13:
                take_profit=current_price+20-4
            elif len(self.trades)==14:
                take_profit=current_price+22-4
            elif len(self.trades)==15:
                take_profit=current_price+24-4
            elif len(self.trades)==16:
                take_profit=current_price+26e-4
            elif len(self.trades)==17:
                take_profit=current_price+28e-4
            elif len(self.trades)==18:
                take_profit=current_price+30e-4 
            elif len(self.trades)==19:
                take_profit=current_price+32e-4
                self.count+=1
                print('####over 20 trades###: ',self.count,self.data.Volume[-1])
                print(self.trades)
            elif len(self.trades)==20:
                take_profit=current_price+35e-4
                
                
            
        
        else:
            '''
            for i in range(len(self.trades)):
                sum_trade += current_price - self.trades[i].entry_price
            
            avg_sum = sum_trade/len(self.trades)              
            take_profit = current_price + avg_sum
            '''
            if len(self.trades)==1:
                take_profit=current_price-4e-4
            elif len(self.trades)==2:
                take_profit=current_price-6e-4
            elif len(self.trades)==3:
                take_profit=current_price-7e-4
            elif len(self.trades)==4:
                take_profit=current_price-8e-4
            elif len(self.trades)==5:
                take_profit=current_price-10e-4
            elif len(self.trades)==6:
                take_profit=current_price-11e-4
            elif len(self.trades)==7:
                take_profit=current_price-12e-4
            elif len(self.trades)==8:
                take_profit=current_price-14e-4
            elif len(self.trades)==9:
                take_profit=current_price-16e-4
            elif len(self.trades)==10:
                take_profit=current_price-17e-4
            elif len(self.trades)==11:
                take_profit=current_price-18e-4
            elif len(self.trades)==12:
                take_profit=current_price-19e-4
            elif len(self.trades)==13:
                take_profit=current_price-20e-4
            elif len(self.trades)==14:
                take_profit=current_price-22e-4
            elif len(self.trades)==15:
                take_profit=current_price-24e-4
            elif len(self.trades)==16:
                take_profit=current_price-26e-4
            elif len(self.trades)==17:
                take_profit=current_price-28e-4
            elif len(self.trades)==18:
                take_profit=current_price-30e-4 
            elif len(self.trades)==19:
                take_profit=current_price-32e-4
                self.count+=1
                print('####over 20 trades###: ',self.count,self.data.Date[-1])
            elif len(self.trades)==20:
                take_profit=current_price-35e-4
                
                

        return take_profit
    

    def sizecalc(self, trades,current_price):
        
        
        if self.trades[-1].is_long:
            if len(self.trades)==1:
                s=self.initsize
            elif len(self.trades)==2:
                s=self.initsize
            elif len(self.trades)==3:
                s=0.00064
            elif len(self.trades)==4:
                s=0.00076
            elif len(self.trades)==5:
                s=0.00092
            elif len(self.trades)==6:
                s=0.00112
            elif len(self.trades)==7:
                s=0.00132
            elif len(self.trades)==8:
                s=0.0016
            elif len(self.trades)==9:
                s=0.00192
            elif len(self.trades)==10:
                s=0.00228 
            elif len(self.trades)==11:
                s=0.00276
            elif len(self.trades)==12:
                s=0.00332
            elif len(self.trades)==13:
                s=0.00390
            elif len(self.trades)==14:
                s=0.00470
            elif len(self.trades)==15:
                s=0.00560
            elif len(self.trades)==16:
                s=0.00670
            elif len(self.trades)==17:
                s=0.0081
            elif len(self.trades)==18:
                s=0.0097 
            elif len(self.trades)==19:
                s=0.012
            elif len(self.trades)==20:
                s=0.016
            
        elif self.trades[-1].is_short:
            if len(self.trades)==1:
                s=self.initsize
            elif len(self.trades)==2:
                s=self.initsize
            elif len(self.trades)==3:
                s=0.00064
            elif len(self.trades)==4:
                s=0.00076
            elif len(self.trades)==5:
                s=0.00092
            elif len(self.trades)==6:
                s=0.00112
            elif len(self.trades)==7:
                s=0.00132
            elif len(self.trades)==8:
                s=0.0016
            elif len(self.trades)==9:
                s=0.00192
            elif len(self.trades)==10:
                s=0.00228 
            elif len(self.trades)==11:
                s=0.00276
            elif len(self.trades)==12:
                s=0.00332
            elif len(self.trades)==13:
                s=0.00390
            elif len(self.trades)==14:
                s=0.00470
            elif len(self.trades)==15:
                s=0.00560
            elif len(self.trades)==16:
                s=0.00670
            elif len(self.trades)==17:
                s=0.0081
            elif len(self.trades)==18:
                s=0.0097 
            elif len(self.trades)==19:
                s=0.012
            elif len(self.trades)==20:
                s=0.016
        
        return s




    def init(self):
        self.bbands=self.I(indicator,self.data)


        
    def next(self):
        upper_band=self.bbands[2]
        lower_band=self.bbands[0]

        #Martingale + Bollinger Bands

        #if no trade exists go long or short or do nothing per BB
        if len(self.trades)==0:

            #long order if <lower band
            if self.data.Close[-1] < lower_band:
                tp1 = self.data.Close[-1] + 7e-4
                self.buy(tp=tp1, size=self.initsize)
                
            #short order if >upper band
            elif self.data.Close[-1] > upper_band:
                tp1 = self.data.Close[-1] - 7e-4
                self.sell(tp=tp1, size=self.initsize)
            
    
        #if trades exists and market is in opposite direction above your threshold apply Martingale
        elif len(self.trades)!=0 :

            #if long trades exists and <20 in count
            if self.trades[-1].is_long and len(self.trades) < 11:
                if self.trades[-1].entry_price - self.data.Close[-1] > 5e-4:
                    self.mysize=self.sizecalc(self.trades,self.data.Close[-1])
                    tp1 = self.takeprofitcalc(self.trades,self.data.Close[-1])
                    for i in range(len(self.trades)):
                        self.trades[i].tp=tp1
                    self.buy(tp=tp1, size=self.mysize)
                   
             
            #if Short trades exists and <20 in count
            elif self.trades[-1].is_short and len(self.trades) < 11:
                
                if self.data.Close[-1]-self.trades[-1].entry_price  > 5e-4:
                    self.mysize=self.sizecalc(self.trades,self.data.Close[-1])
                    tp1 = self.takeprofitcalc(self.trades,self.data.Close[-1])
                    for i in range(len(self.trades)):
                        self.trades[i].tp=tp1
                    self.sell(tp=tp1, size=self.mysize) 
                    
            
               
            
          

#initializing backtest
bt=Backtest(df,Martingale,cash=25000,margin=1/1000)

#running backtest
stats=bt.run()
print(stats)

#plot backtest
#bt.plot()