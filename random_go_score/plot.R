
rm(list=ls())

library(tidyr)
library(ggplot2)
library(gridExtra)
library(ggpubr)
library(dplyr)


filter_positive = function(fin){
  ## too many positive cases 
  negative = subset(fin,fin$label=='not_entailment')
  positive = subset(fin,fin$label=='entailment')
  set.seed(2019)
  positive2 = sample_n(positive, nrow(negative))
  return (rbind(negative,positive2))
}

setwd("C:/Users/dat/Documents/RandomGOAnalysis")


file_list = c('AIC/Human_go_analysis_all.AIC.degree.txt',
'BiLSTM/cosine.bilstm768/go_analysis_all.degree.txt',
# 'Elmo/cosine.1024/go_analysis_all.degree.txt',
'BertAsService/cosine.precompvec.300Vec/go_analysis_all.BertAsService.degree.txt',
'BERT/cosine.AveWordClsSep768.Linear768.Layer12/go_analysis_all.degree.txt', 
'BERT/cosine.Cls768.Linear768/go_analysis_all.degree.txt',
'BertGOName768/go_analysis_all.degree.txt', 
'GCN/cosine.768/go_analysis_all.degree.txt', 
'Onto2vec/cosine.768/go_analysis_all.degree.txt' )

# plot_name = c('AIC','BiLSTM','ELMo', 'BERT1', 'BERT2', 'BERT3', 'BERT4', 'GCN', 'Onto2vec')
plot_name = c('AIC','BiLSTM', 'BERT1', 'BERT2', 'BERT3', 'BERT4', 'GCN', 'Onto2vec')


dfmain = read.table(file_list[1],header=T,sep="\t",row.names=NULL)
dfmain = subset (dfmain, dfmain$ic1>=1 & dfmain$ic2>=1)
dfmain = subset(dfmain , dfmain$degree1<98 & dfmain$degree2<98) # 480

# dftemp = dfmain[c('go1','ic1','degree1')]
# dftemp = dftemp [ ! duplicated(dftemp), ]
# cor (dftemp$ic1,dftemp$degree1)

dfmain = dfmain[ c('go1','go2','ic1','ic2') ]
dfmain = dfmain [ ! duplicated(dfmain), ]

choice = 'ic' ## degree or ic

label_name = list() 
label_name[['ic']] = 'Range of min IC of GO pair' 
label_name[['degree']] = 'Range of max degree of GO pair'

allplot = list()

for (i in seq(1,length(plot_name))) {

  df = read.table(file_list[i],header=T,sep="\t",row.names=NULL)
  # colnames(df) = c("go1","go2","label","type","score","degree1","degree2","MeanDegree")

  df = subset(df , df$degree1<98 & df$degree2<98) ## may not need this, because of @dfmain
  df$MaxDegree = pmax(df$degree1,df$degree2)
  if (plot_name[i]=='AIC') {
    df$ic1=NULL
    df$ic2=NULL
  }

  df = df [! duplicated(df), ]
  df = merge (df, dfmain, by=c('go1','go2') , all = F) ## overlap set only 
  df = df [! duplicated(df), ]
  print (dim(df))
  df$MinIc = pmin(df$ic1,df$ic2)
  df = subset(df , df$ic1>=1 & df$ic2>=1) ## may not need this, because of @dfmain

  df = filter_positive(df)
  print (table(df$label))

  if (choice == 'ic'){
    xlabel = levels ( cut_width(df$MinIc,.5) ) ## use 5 for MaxDegree
    plot1 = ggplot(df) +
    geom_boxplot(aes(x=cut_width(MinIc,.5),y=score,fill=label), outlier.size=0.5) + # group = cut_width(carat, 0.5) # MinIc
    geom_smooth(method = "lm", se=FALSE, aes(x=cut_width(MinIc,.5),y=score,group=label, color=label) ) + 
    scale_color_manual( values=c("red","blue"), name = "", labels = "") 
  }

  if (choice == 'degree') {
    xlabel = levels ( cut_width(df$MaxDegree,5) ) ## use 5 for MaxDegree
    plot1 = ggplot(df) +
    geom_boxplot(aes(x=cut_width(MaxDegree,5),y=score,fill=label), outlier.size=0.5) +
    geom_smooth(method = "lm", se=FALSE, aes(x=cut_width(MaxDegree,5),y=score,group=label, color=label) ) + 
    scale_color_manual( values=c("red","blue"), name = "", labels = "") 
  }

  plot1 = plot1 +
  scale_x_discrete( name=label_name[[choice]], labels=xlabel ) + # , breaks=seq(0, 420, 10), labels=seq(0,420,10)
  # scale_y_continuous( name="Score scaled", breaks=seq(0, 1, .1) ) +
  scale_fill_manual(values=c("gray44","gray86"),name = "", labels = c("Child-Parent", "Random"), ) +
  theme_linedraw() + theme_light() + 
  theme(legend.position="bottom",plot.title = element_text(hjust = 0.5),
  axis.text.x = element_text(angle = 25, hjust = 1, size=9) ) +
  ggtitle(plot_name[i]) + 
  guides(color=FALSE)+ ## turn off color legend
  theme(legend.text=element_text(size=14), legend.key.size = unit(1.25, "cm") )

  if ( ! plot_name[i] %in% c('AIC','ELMo','BERT1','BERT4','Onto2vec') ) {
    # plot1 = plot1 + geom_hline(yintercept=0.5, size=1)
    plot1 = plot1 + ylim(-1,1) 
  }

  # if (i != 6){
  #   plot1 = plot1 + theme(axis.title.x = element_blank(), axis.text.x = element_blank())
  # }

  # ggsave(paste0(plot_name[i],'_GO_pairs_',choice,'.PNG'),device='png',width=12,height=3,units='in')

  allplot[[i]] = plot1
}



pp = ggarrange(allplot[[1]],allplot[[2]],allplot[[3]],allplot[[4]],
          allplot[[5]],allplot[[6]],allplot[[7]],
          allplot[[8]],
          # allplot[[9]],
          # allplot[[10]],
          common.legend = TRUE, ncol=2, nrow=4) 


ggsave( paste0(choice,".pdf"), pp , device='pdf',width=14,height=10,units='in' )


