###
#Make sure you clean the download folder in the project before running
###
###Scrape stock prices Yahoo
setwd("D:\\OneDrive\\SMU\\6306 doing data science\\Project 2")
#Stock list
st <- c('AXP'  
         ,'AAPL', 'BA',  'CAT',    'CSCO',   'CVX',  'DIS',  'DWDP',    'GS'  , 'HD', 'IBM', 'INTC',
         'JNJ',  'JPM', 'KO', 'MCD'  ,'MMM',   'MRK', 'MSFT',  'NKE' , 'PFE' ,'PG' , 'TRV' , 'UNH', 'UTX' ,
         'V' ,'VZ', 'WBA', 'WMT' , 'XOM'
        )

#Download stock prices
for (s in 1:length(st)){
  url<- paste("https://query1.finance.yahoo.com/v7/finance/download/",
              st[s],
              "?period1=1104566400&period2=1532847600&interval=1d&events=history&crumb=sW.zPVRhSGW", sep="")
  utils::browseURL(url)
  Sys.sleep(time = 5)
  dest<-paste("D:\\OneDrive\\SMU\\6306 doing data science\\Project 2\\download\\",st[s],".csv", sep="")
  file.rename(paste("C:\\Users\\Ivan\\Downloads\\",st[s],".csv", sep=""), dest)
  
  quote<-read.csv(dest)
  
  
  #Strategy
  quote$pos[1:nrow(quote)]<-0
  quote$buy[1:nrow(quote)]<-0
  quote$sell[1:nrow(quote)]<-0
  
  #Strategy 1: multiple downs with a gap up 
  # for (i in 5:nrow(quote)) {
  #   if ( (quote$Close[i-1]<quote$Open[i-1])&&(quote$Close[i-2]<quote$Open[i-2])&&
  #        (quote$Close[i-3]<quote$Open[i-3])&&
  #        #(quote$Close[i-4]<quote$Open[i-4])&&
  #        (quote$Close[i-3]>quote$Close[i-2])&&
  #        (quote$Close[i-2]>quote$Close[i-1])&&
  #        (quote$Open[i]>quote$Close[i-1]) 
  #   )
      
  #Strategy 2: breakout
  for (i in 5:nrow(quote)) {
    if ( ((quote$Close[i-1]-quote$Open[i-1])<quote$Open[i-1]*0.01)&&
         ((quote$Close[i-2]-quote$Open[i-2])<quote$Open[i-1]*0.01)&&
         ((quote$Close[i-3]-quote$Open[i-3])<quote$Open[i-1]*0.01)&&
         ((quote$Close[i-4]-quote$Open[i-4])<quote$Open[i-1]*0.01)&&
         #(quote$Close[i-4]<quote$Open[i-4])&&
         ((quote$Close[i-4]-quote$Close[i-3])<quote$Open[i-1]*0.01)&&
         ((quote$Close[i-3]-quote$Close[i-2])<quote$Open[i-1]*0.01)&&
         ((quote$Close[i-2]-quote$Close[i-1])<quote$Open[i-1]*0.01)&&
         (quote$Open[i]>quote$High[i-1])
    )

      
      
    {if (quote$pos[i]!=1)
      
    {
      quote$buy[i]<- 1
      for (j in i:nrow(quote)) {
        quote$pos[j]<-1
        if ( (quote$Open[j]>(quote$Open[i]*1.05))||(quote$Open[j]<(quote$Open[i]*0.98))
        )
        {quote$sell[j]<-1
        break
        }
        else {quote$sell[j]<-0}   
      }
    }
      
    }
    else {quote$buy[i]<- 0}
  }
  
  #Attach results for this stock to the overall result
  quote$Company<-st[s]
  if (s==1){allquote<-quote}
  else { allquote<-rbind(allquote,quote)}
  
  #plot

#  plot (quote$Date, quote$Close, col=ifelse(quote$pos==1, "red", "yellow"))
}


#Save results
write.csv(allquote, file="allstocks.csv")