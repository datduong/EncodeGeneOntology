
# /u/flashscratch/d/datduong/YeastPPI3ontology/score3ontologyYeastPPI.txt
# /u/flashscratch/d/datduong/geneOrtholog/YeastMouseScore/YeastMouse3ontology.txt

library('MKmisc')

library ("ROCR")
computeAUC = function(dataIn1,dataIn2,methodName){
	
	# @dataIn2 is the "negative" data
	# stupid @performance needs both sets to be the same size. 

	if (nrow(dataIn1)<nrow(dataIn2)){
		dataIn2 = dataIn2[1:nrow(dataIn1), ]
		negativeSet = dataIn2
	} else{
		dataIn1 = dataIn1[1:nrow(dataIn2), ]
		positiveSet = dataIn1
	}

	dataUsedInAuc = rbind (positiveSet,negativeSet) ## actual data used 

	label1 = rep(1,nrow(dataIn1))
	label2 = 1-label1 ## flip label (it's ok to do this because both set are same size)
	
	pred1 = t( dataIn1[,methodName] )
	pred2 = t( dataIn2[,methodName] )

	predValue = c(pred1,pred2)
	labelValue = c(label1,label2)

	pred = prediction(predValue,labelValue)
	areaUnderCurve = performance(pred, "auc")
	perf = performance(pred,"tpr","fpr")
	# plot(perf,...)
	return ( list( perf,areaUnderCurve,predValue,labelValue,dataUsedInAuc ) )
}

path = '/u/flashscratch/d/datduong/HumanPPI3ontology/'
setwd(path)
# 'GCN.cosine.768' , 'cosine.AveWordClsSep768.Linear768.Layer12' , 'cosine.bilstm768' , 'cosineCls768Linear768'
model_type = c( 'BertGOName768' )

dataUsedInAuc = NULL 
ret = NULL 
for (ontology in c("all3") ) { ## "BP","CC","MF",
	print (ontology) 
	
	result = read.table(paste0('scoreHumanPPI',ontology,'wSimDef.txt'),header=F) # wSimDef
	
	stringName1 = c('resnik','aic','w2v','inferSent','simDef') # w2v
	name1 = c() 
	for (i in stringName1) {
		name1 = c(name1, paste0( i,c("Min","Mean") ) )
	}	

	stringName2 = c('aic','w2v','inferSent','simDef') 
	name2 = c()
	for (i in 1:3) {
		for (j in (i+1):4) {
			name2 = c(name2, paste0( paste0(stringName2[i],stringName2[j]) ,c("Min","Mean") ) )
		}
	}

	colnames(result) = c("g1","g2",name1,name2,"label")
	types = c(name1,name2)

	print ( dim(result) ) 
	for (label in types ) { 
		index2rm = which ( result[,label]<0 )
		if (length(index2rm) > 0 ) {
			result = result [-1*index2rm, ] 
		}
		# print ( dim(result) ) 
	}
	print ( dim(result) ) 

	## append 
	counter = 1
	for (next_data in model_type) {
		print (next_data)
		next_data_file = read.table(paste0('qnliFormatData17/',next_data,'/GeneScore.tsv'),header=F,sep="\t")
		colnames(next_data_file) = c('g1','g2','label1',paste0('min',next_data),paste0('mean',next_data)) 
		result = merge ( result, next_data_file, by=c('g1','g2'))
		print ( dim (result) )
		types = c(types,paste0('min',next_data),paste0('mean',next_data))
	}

	
	positiveSet = subset ( result , result$label==1 ) 
	negativeSet = subset ( result , result$label==0 ) 

	set.seed(2222)

	positiveSet = result [ which(result[,'label']==1), ] 
	print ('positiveSet size')
	print ( dim(positiveSet) )
	positiveSet <- positiveSet[sample(nrow(positiveSet)),]

	negativeSet = result [ which(result[,'label']==0), ]
	print ('negativeSet size')
	print ( dim(negativeSet) ) 
	negativeSet <- negativeSet[sample(nrow(negativeSet)),]


	aucOutput = list() 
	aucArr = NULL 
	for (type in types ){
		out = computeAUC(positiveSet,negativeSet,type)
		aucOutput[[type]] = out
		aucArr = c ( aucArr, unlist(out[[2]]@y.values) ) 
		dataUsedInAuc = out[[5]] 
	}
	
	numType = length(types)
	outputAucTest = matrix(-1,numType,numType)
	for ( i in 1:numType ) {
		typeI = aucOutput[[ types[i] ]][[3]]
		label = aucOutput[[ types[i] ]][[4]]
		for (j in i:numType) {
			typeJ = aucOutput[[ types[j] ]][[3]]
			testResult = AUC.test( typeI, label, typeJ, label )
			# indicator = 0
			# if (testResult$Test[[2]] < .05) {
				# indicator = 1
			# }
			outputAucTest[i,j] = testResult$Test[[2]] # indicator
			outputAucTest[j,i] = outputAucTest[i,j]
		}
	}
	colnames(outputAucTest)=types
	rownames(outputAucTest)=types

	if (is.null(ret)){
		ret = data.frame(types,aucArr)
	} else {
		ret = cbind(ret,aucArr)
	}
}

print (ret)


dataUsedInAuc = dataUsedInAuc[c('g1','g2','label')]
colnames(dataUsedInAuc) = c('gene1','gene2','label')
write.table( dataUsedInAuc, 'dataUsedInAuc.csv', sep=" ", quote=F, row.names=FALSE)

