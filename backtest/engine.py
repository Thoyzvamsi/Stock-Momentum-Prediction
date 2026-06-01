import pandas as pd
from models.direction_model import Direction_Model

class Strategy:
    def strategy_execution(data,market_type ,capital):
        data = Direction_Model(data).training_model()

        trade_on = 0
        entry = 0
        trade = []


        for i in range(len(data)-4):
            if (data["pred"].iloc[i] == 1 and data["prob"].iloc[i] > 0.55 and data["close"].iloc[i+4]) and trade_on == 0 :
                entry = data["close"].iloc[i+1]
                entry_date = data["date"].iloc[i+1]
                entry_time = data["time"].iloc[i+1]
                qnt = capital//entry

                if qnt <= 0:
                    continue
                trade_on = 5
            
            elif trade_on == 1:
                trade_on = 0

                exit_price = data["close"].iloc[i]
                charges = Strategy.charges_calulation(entry, exit_price, qnt ,market_type)
                exit_date = data["date"].iloc[i]
                exit_time = data["time"].iloc[i]
                PnL = ((data["close"].iloc[i+1] - entry)*qnt) #- charges
                capital += PnL
                    
                trade.append({
                        "Entry" : entry ,
                        "Exit" : exit_price ,
                        "Quantity" : qnt,
                        #"Charges" : charges,
                        "Entry_date" : entry_date,
                        "Entry_time" : entry_time,
                        "Exit_date" : exit_date,
                        "Exit_time" : exit_time,
                        "PnL" : PnL ,
                        "Capital" : capital
                    })

            elif trade_on > 1 :
                trade_on -= 1
           
        trade_df = pd.DataFrame(trade)

        return trade_df

    
    def charges_calulation(entry, exit, qnt ,market_type):
        charges = 0
        brokerage = 5
        if market_type == 0:
            #Brokerage
            if brokerage > 0.1 * (entry*qnt):
                charges += 10
            elif (0.1 * (entry*qnt)) < 20:
                charges += (0.1 * (entry*qnt)) + (0.1 * (exit*qnt))
            else:
                charges += 40

            #Dp charges
            charges += 20 + (0.18*20)
            #STT (Gov tax)
            charges += (0.01*entry) + (0.01*exit)
            #Stamp duty
            charges += 0.00015*entry*qnt

            if (exit - entry)*0.0325 > 0:
                charges += (exit - entry)*0.0325
            
            return charges
        
        if market_type == 1:
            #Brokerage
            if brokerage > 0.1 * (entry*qnt):
                charges += 5*2 
            elif 0.1 * (entry*qnt) < 20:
                charges += 0.1 * (entry*qnt)
            else:
                charges += 20 * 2

            #STT
            #charges += 0.025*exit
            charges += 0.00003*entry*qnt
            if ((exit - entry)*qnt)*0.0325 > 0:
                charges = ((exit - entry)*qnt)*0.0325

            return charges