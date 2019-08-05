
library("ggplot2")

setwd('C:/Users/dat/Documents')

par (mfrow=c(1,3))
for (category in c('bp','mf','cc')) {
  df = read.table(paste0("CountGoInTrain-",category,".tsv"),sep="\t",header=T)
  print (category)
  print (dim(df))
  hist(df$count,main=paste0('GO frequency in train data ',category),breaks=50, xlab='Frequency', ylab='Num. of GO')
  print (summary(df$count))
}

