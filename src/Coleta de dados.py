ENUM_TIMEFRAMES timeframes[]={PERIOD_M1,PERIOD_M5,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H4,PERIOD_D1};

string TimeframeToString(ENUM_TIMEFRAMES tf)
{
 switch(tf)
 {
   case PERIOD_M1:return("M1");break;
   case PERIOD_M5:return("M5");break;
   case PERIOD_M15:return("M15");break;
   case PERIOD_M30:return("M30");break;
   case PERIOD_H1:return("H1");break;
   case PERIOD_H4:return("H4");break;
   case PERIOD_D1:return("D1");break;
   default:return("");break;
 }
 return("");
}

void Write()
{//write data file
   for (int i=0;i<ArraySize(timeframes);i++)
   {
      string datafile=Symbol()+"_"+TimeframeToString(timeframes[i])+".csv";
      Print(datafile);
      int fhandle=FileOpen(datafile,FILE_TXT|FILE_CSV|FILE_WRITE,";");
   
      if(fhandle!=INVALID_HANDLE) 
      {
         MqlRates rt[];
         ArraySetAsSeries(rt,false);
         datetime starttime=D'2013.01.01 00:00',endtime=D'2026.12.31 00:00';
         int rates=CopyRates(Symbol(),timeframes[i],starttime,endtime,rt);
         Print(rates);
         FileWrite(fhandle,
               "time",
               "open",
               "high",
               "low",
               "close",
               "tick_volume",
               "real_volume"
               );
         for(int j=0;j<rates;j++)
         {
            FileWrite(fhandle,
                  rt[j].time,
                  rt[j].open,
                  rt[j].high,
                  rt[j].low,
                  rt[j].close,
                  rt[j].tick_volume,
                  rt[j].real_volume
                  );
         }
   
         FileClose(fhandle); 
      }
   } 
   return;
}

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   Write();
//---
   
  }
//+------------------------------------------------------------------+