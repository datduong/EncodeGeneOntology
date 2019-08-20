
rm(list=ls())

library('ggplot2')

setwd("C:/Users/dat/Documents/RandomGOAnalysis")


file_list = c('AIC/Human_go_analysis_all.AIC.degree.txt', 'GCN/cosine.300/go_analysis_all.GCN.degree.txt', 'BERT/cosine.768.reduce300ClsVec/go_analysis_all.BERT.degree.txt', 'BiLSTM/cosine.bilstm.300Vec/go_analysis_all.BiLSTM.degree.txt', 'BERT/cosine.768.reduce300SecondLayerVec/go_analysis_all.BERT.degree.txt' )

plot_name = c('AIC','GCN','BERT [CLS]','BiLSTM', 'BERT ave(2nd last layer)')


dfmain = read.table(file_list[1],header=T,sep="\t",row.names=NULL)
dfmain = subset (dfmain, dfmain$ic1>0 & dfmain$ic2>0)
dfmain = subset(dfmain , dfmain$degree1<480 & dfmain$degree2<480)

# dftemp = dfmain[c('go1','ic1','degree1')]
# dftemp = dftemp [ ! duplicated(dftemp), ]
# cor (dftemp$ic1,dftemp$degree1)

dfmain = dfmain[ c('go1','go2','ic1','ic2') ]
dfmain = dfmain [ ! duplicated(dfmain), ]

choice = 'degree' ## degree or ic

for (i in seq(1,length(plot_name))) {

  df = read.table(file_list[i],header=T,sep="\t",row.names=NULL)
  # colnames(df) = c("go1","go2","label","type","score","degree1","degree2","MeanDegree")

  df = subset(df , df$degree1<480 & df$degree2<480)
  df$MaxDegree = pmax(df$degree1,df$degree2)
  if (plot_name[i]=='AIC') {
    df$ic1=NULL
    df$ic2=NULL
  }

  df = df [! duplicated(df), ]
  df = merge (df, dfmain, by=c('go1','go2') , all = FALSE)
  df = df [! duplicated(df), ]
  print (dim(df))
  df$MinIc = pmin(df$ic1,df$ic2)


  # windows()

  if (choice == 'ic'){
    xlabel = levels ( cut_width(df$MinIc,.5) ) ## use 5 for MaxDegree
    plot1 = ggplot(df) +
    geom_boxplot(aes(x=cut_width(MinIc,.5),y=score,fill=label), outlier.size=0.5) # group = cut_width(carat, 0.5) # MinIc
  }

  if (choice == 'degree') {
    xlabel = levels ( cut_width(df$MaxDegree,5) ) ## use 5 for MaxDegree
    plot1 = ggplot(df) +
    geom_boxplot(aes(x=cut_width(MaxDegree,5),y=score,fill=label), outlier.size=0.5) # group = cut_width(carat, 0.5) # MinIc
  }


  plot1 = plot1 +
  scale_x_discrete( name="Range of max degree of GO pair", labels=xlabel ) + # , breaks=seq(0, 420, 10), labels=seq(0,420,10)
  # scale_y_continuous( name="Score scaled", breaks=seq(0, 1, .1) ) +
  scale_fill_discrete(name = "Relation", labels = c("Child-Parent", "Random")) +
  theme(legend.position="bottom",plot.title = element_text(hjust = 0.5),
  axis.text.x = element_text(angle = 45, hjust = 1) ) +
  ggtitle(plot_name[i])

  if (plot_name[i]!='AIC') {
    plot1 = plot1 + geom_hline(yintercept=0.5, size=1)
  }

  ggsave(paste0(plot_name[i],'_GO_pairs_',choice,'.PNG'),device='png',width=12,height=3,units='in')

}


