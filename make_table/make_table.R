

library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)


fin = read.table('OrthologBert.txt',header=T)
fin2 = fin [ grep ("Mean",fin$method ) , ] 
# fin2$method = c('AIC', 'BiLSTM','BertAsService','BertAveWordClsSep','BertCls','BertGOName','GCN','Onto2vec')
rownames(fin2)= c('AIC', 'BiLSTM', 'ELMo','BERTas','BERT12','BERTCLS','GCN','Onto2vec','BERTname')
fin2$method=NULL 
fin2 = fin2*100
fin2 = round(fin2,digits=2)

## model type
# fin2$type = c(rep('IC',2), 'TF-IDF', rep('GO def',5), rep('GO pos',2))

kable(fin2, "latex", longtable = F, booktabs = T, caption = "Set comparison") %>%
add_header_above(c(" "=1, "Orthologs data" = 3, "PPI data" = 2)) %>%
kable_styling(latex_options = c("hold_position",position = "center"),font_size =10) %>%
pack_rows("Info Content", 1, 1,latex_gap_space = "0.4em") %>% 
# pack_rows("TF-IDF", 3,3,latex_gap_space = "0.4em") %>%
pack_rows("Defintion encoders", 2,6,latex_gap_space = "0.4em") %>%
pack_rows("Entity encoders", 7,9,latex_gap_space = "0.4em") 


### 
library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('OrthologBertGoSummary.txt',header=T)
# rownames(fin2)=fin2$Name
# fin2$Name=NULL 

## model type
# fin2$type = c(rep('IC',2), 'TF-IDF', rep('GO def',5), rep('GO pos',2))

kable(fin2, "latex", longtable = F, booktabs = T, caption = "Set comparison") %>%
add_header_above(c(" "=1, "Negative set" = 3, "Positive set" = 3)) %>%
kable_styling(latex_options = c("repeat_header","hold_position",position = "center")) %>%
pack_rows("Dataset 1", 1, 2,latex_gap_space = "0.4em") %>% 
pack_rows("Dataset 2", 3,4,latex_gap_space = "0.4em") %>%
pack_rows("Dataset 3", 5,6,latex_gap_space = "0.4em") 


### 
fin2 = read.table('RecallPrecisionDeepGo1GoQ75.txt',header=T)
fin2$metric=NULL
fin2$metric.1=NULL
fin2$ontology=NULL


# rownames(fin2)=fin2$Name
# fin2$Name=NULL 

## model type
# fin2$type = c(rep('IC',2), 'TF-IDF', rep('GO def',5), rep('GO pos',2))

kable(fin2, "latex", longtable = F, booktabs = T, caption = "Recall@k and Precision@k for all GO terms using original DeepGO data.") %>%
add_header_above(c(" "=1, "Recall@k" = 5, "Precision@k" = 5)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("BP ontology", 1, 8,latex_gap_space = "0.3em") %>% 
pack_rows("MF ontology", 9,16,latex_gap_space = "0.3em") %>%
pack_rows("CC ontology", 17,24,latex_gap_space = "0.3em")

#  %>%
# landscape()



### 
fin2 = read.table('AucDeepGo1AllGo.txt',header=T)
fin2$metric=NULL
fin2$metric.1=NULL
fin2$ontology=NULL

kable(fin2, "latex", longtable = F, booktabs = T, caption = "Evaluation for all GO terms using original DeepGO data.") %>%
add_header_above(c(" "=2, "AUC" = 2, " "=1,"AUC" = 2, " "=1,"AUC" = 2 )) %>%
add_header_above(c(" "=1, "BP" = 3, "MF" = 3, "CC"=3)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("DeepGO baseline", 1, 3,latex_gap_space = "0.3em") %>% 
pack_rows("GO definition", 4,9,latex_gap_space = "0.3em") %>%
pack_rows("GO position", 10,12,latex_gap_space = "0.3em")  

%>%

row_spec(3:5, bold = T, color = "white", background = "black")



#### long table with standard and zeroshot

library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('GOEncoderResult3Jan2020.txt',sep='\t',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 100
# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=TRUE ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "BP AUC" = 2, "MF AUC" = 2, "CC AUC"=2)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>% 
pack_rows("Train and test DeepGOSeqFlat on DeepGO data", 1, 14, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Train and test DeepGOZero on DeepGO data", 15, 27, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Train on larger DeepGO data, and test on original labels", 28, 28, label_row_css = "background-color: #666; color: #fff;") %>%
pack_rows("Train on larger DeepGO data, and test on added labels", 29, 29, label_row_css = "background-color: #666; color: #fff;") %>%
pack_rows("Train DeepGOZero on DeepGO data, and test on unseen labels", 30, 41, label_row_css = "background-color: #666; color: #fff;") %>%
add_indent( c(5:10)) %>%
add_indent( c(12:14)) %>%
add_indent( c(18:23)) %>%
add_indent( c(25:27)) %>%
add_indent( c(32:37)) %>%
add_indent( c(39:41)) 


pack_rows("DeepGO baseline", 1, 3,latex_gap_space = "0.3em") %>% 
pack_rows("GO definition", 4,9,latex_gap_space = "0.3em") %>%
pack_rows("GO position", 10,12,latex_gap_space = "0.3em")  


#### make table of method on deepgo data original one, without larger labels. 
library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('GOEncoderResult28Jan2020.txt',sep='\t',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 100
# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=F ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=1, "BP AUC" = 2, "MF AUC" = 2, "CC AUC"=2)) %>%
pack_rows("Baseline", 1, 3, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Defintion encoders", 4, 8, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Entity encoders", 9, 11, label_row_css = "background-color: #666; color: #fff;")


#### zeroshot using concat of 4 vectors.
library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('GOEncoderZeroshotResult28Jan2020.txt',sep='\t',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 100
# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=F ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=1, "BP AUC" = 1, "MF AUC" = 1, "CC AUC"=1)) %>%
pack_rows("Baseline", 1, 2, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Defintion encoders", 3, 7, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Entity encoders", 8, 10, label_row_css = "background-color: #666; color: #fff;")





%>%


%>%
add_header_above(c(" "=1, "Macro/Micro AUC" = 2,"AUC" = 2,"AUC" = 2 )) %>%

kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("DeepGO baseline", 1, 3,latex_gap_space = "0.3em") %>% 
pack_rows("GO definition", 4,9,latex_gap_space = "0.3em") %>%
pack_rows("GO position", 10,12,latex_gap_space = "0.3em")  


