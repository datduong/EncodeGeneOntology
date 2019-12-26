
library(tidyr)
library(ggplot2)
library(gridExtra)
library(ggpubr)

fin = read.table("recallprecision25.txt",header=T,stringsAsFactors=F)

# data_long = gather(fin, f1threshold, f1value, factor_key=TRUE)
Method2get = c('Baseline','BertAsService','BertGONameDim768','bilstm768','Onto2vec768','GCN768','BertAveWordClsSepLayer12Dim768','BertWeighted11+12AveWordClsSepLayer12Dim768','BertClsLayer12Dim768','Elmo768')

print (sort(Method2get))

label_name = gsub('Dim768','',sort(Method2get))
label_name = gsub('768','',label_name)

# label_name = c('Baseline2','BERT As Service', 'BERT Average Layer 12','BERT CLS token','BERT GO name','BiLSTM','GCN','Onto2vec')
label_name = c('Baseline2','BERTas', 'BERT12','BERTCLS','BERTname','BERT11+12','BiLSTM','ELMO','GCN','Onto2vec')
# label_name
# [1] "Baseline"                 "BertAsService"            "BertAveWordClsSepLayer12"
# [4] "BertClsLayer12"           "BertGOName"               "bilstm"
# [7] "GCN"                      "Onto2vec"

fin$name = paste0(fin$Method,fin$metric,fin$ontology)
PlotList = list()
metric_type = 'rec'

metricMap = list()
metricMap[['rec']] = 'Recall@k'
metricMap[['prec']] = 'Precision@k'

quantileMap = list()
quantileMap[['quant25']] = '<25% quantile'
quantileMap[['quant75']] = '>75% quantile'
quantileMap[['betweenQ25Q75']] = '25-75% quantile'
quantileMap[['all']] = 'Complete data'

cbbPalette = c("#000000", "gold2", "deepskyblue", "peachpuff3", "#D55E00", 'deeppink2', "chartreuse1","gray50","darkorchid4","#009E73")

# quantile_range="quant25"
for (quantile_range in c( "all", "quant25"   ,    "betweenQ25Q75"    ,   "quant75" ) ) {

  for (ontotype in c("bp","mf","cc")) {

    fin2 = subset(fin,fin$ontology==ontotype & fin$metric==metric_type & fin$Method%in%Method2get & quantile==quantile_range)

    data_long = gather(fin2, f1threshold, f1value, X10:X25, factor_key=TRUE)

    PlotList[[ paste0(ontotype,quantile_range) ]] = ggplot(data=data_long, aes(x=f1threshold, y=f1value, group=factor(name) )  ) +
    # stat_smooth(aes(color=factor(name),linetype=factor(name)),size=1.1, se=F, geom='line') +
    geom_line(aes(color=factor(name),linetype=factor(name)),size=1.1,alpha=0.8) +
    scale_x_discrete( name="k", labels=seq(10,25,5) ) +
    scale_y_continuous( name=paste0(metricMap[[metric_type]])) + # + ylim(0, .65) + # , limits=c(0,.65)
    scale_colour_manual(name = "", labels = label_name, values=cbbPalette ) + # palette = 'Set1',direction=-1, type = 'div'
    theme(legend.position="left",plot.title = element_text(hjust = 0.5)) +
    ggtitle(paste (toupper(ontotype),quantileMap[[quantile_range]]) ) + theme_linedraw() + theme_light() +
    guides(colour = guide_legend(override.aes = list(size=5))) +
    guides(linetype = FALSE) +
    theme(legend.text=element_text(size=16))

  }

}

# windows()

# grid.arrange(PlotList[['bpquant25']], PlotList[['bpbetweenQ25Q75']], PlotList[['bpquant75']],
# PlotList[['mfquant25']], PlotList[['mfbetweenQ25Q75']], PlotList[['mfquant75']],
# PlotList[['ccquant25']], PlotList[['ccbetweenQ25Q75']], PlotList[['ccquant75']],
# ncol=3, nrow=3)

ggarrange(
PlotList[['bpall']], PlotList[['bpquant25']], PlotList[['bpbetweenQ25Q75']], PlotList[['bpquant75']],
PlotList[['mfall']], PlotList[['mfquant25']], PlotList[['mfbetweenQ25Q75']], PlotList[['mfquant75']],
PlotList[['ccall']], PlotList[['ccquant25']], PlotList[['ccbetweenQ25Q75']], PlotList[['ccquant75']],  common.legend = TRUE, ncol=4, nrow=3)

