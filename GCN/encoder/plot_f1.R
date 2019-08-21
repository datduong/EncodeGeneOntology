
library(tidyr)
library(ggplot2)
library(gridExtra)
library(ggpubr)

fin = read.table("GOEncoderF1.txt",header=T,stringsAsFactors=F)

# data_long = gather(fin, f1threshold, f1value, factor_key=TRUE)

fin$name = paste0(fin$method,fin$metric_type,fin$type,fin$range)
PlotList = list()
metric_type = 'macroRec'

metricMap = list() 
metricMap[['macroRec']] = 'Macro recall'

quantileMap = list()
quantileMap[['quant25']] = '<25% quantile'
quantileMap[['quant75']] = '>75% quantile'
quantileMap[['betweenQ25Q75']] = '25-75% quantile'

for (quantile_range in c( "quant25"   ,    "betweenQ25Q75"    ,   "quant75" ) ) {

for (ontotype in c("bp","mf","cc")) {

  fin2 = subset(fin,fin$type==ontotype & fin$metric==metric_type & fin$range==quantile_range)

  data_long = gather(fin2, f1threshold, f1value, X0.1:X0.9, factor_key=TRUE)

  PlotList[[ paste0(ontotype,quantile_range) ]] = ggplot(data=data_long, aes(x=f1threshold, y=f1value, group=factor(name) )) +
  geom_smooth(aes(color=factor(name),linetype=factor(name)),size=1.5, se=F) +
  scale_x_discrete( name="Rounding cutoff", labels=seq(0.1,.9,.1) ) +
  scale_y_continuous( name=paste0(metricMap[[metric_type]]) ) + # ylim(0, .5) +
  scale_color_brewer(name = "", labels = c("BERT-AS", "BERT-AveToken+CLS+SEP","BERT-AveToken", "BERT-CLS","BiLSTM","GCN","None","Onto2vec"), palette="Set2" ) +
  theme(legend.position="left",plot.title = element_text(hjust = 0.5)) +
  ggtitle(paste (toupper(ontotype),quantileMap[[quantile_range]]) ) + theme_linedraw() + theme_light() + 
  guides(colour = guide_legend(override.aes = list(size=4))) + 
  guides(linetype = FALSE)

}

}

# windows()

# grid.arrange(PlotList[['bpquant25']], PlotList[['bpbetweenQ25Q75']], PlotList[['bpquant75']], 
# PlotList[['mfquant25']], PlotList[['mfbetweenQ25Q75']], PlotList[['mfquant75']],
# PlotList[['ccquant25']], PlotList[['ccbetweenQ25Q75']], PlotList[['ccquant75']],
# ncol=3, nrow=3)

ggarrange(PlotList[['bpquant25']], PlotList[['bpbetweenQ25Q75']], PlotList[['bpquant75']], 
PlotList[['mfquant25']], PlotList[['mfbetweenQ25Q75']], PlotList[['mfquant75']],
PlotList[['ccquant25']], PlotList[['ccbetweenQ25Q75']], PlotList[['ccquant75']],  common.legend = TRUE, ncol=3, nrow=3)

