
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

for ( data_type in c( 'HumanMouseScore','HumanFlyScore','MouseFlyScore','FlyWormScore' ) )  {

path = paste0( '/u/scratch/d/datduong/geneOrtholog/',data_type )
setwd(path)
ret = NULL
dataUsedInAuc = NULL 

for (ontology in c("all3") ) { ## "BP","CC","MF",

	print (ontology)
	## append
	fin = 'qnliFormatData17/BertGOName768/GeneScore.txt'
	next_data_file = read.table(fin,header=F,sep='\t')
	# gene1,gene2,trueLabel,score1,score2

	result = read.table(paste0(data_type,ontology,'.txt'),header=F)

	stringName1 = c('resnik','aic','w2v','inferSent') # w2v 'simDef'
	name1 = c()
	for (i in stringName1) {
		name1 = c(name1, paste0( i,c("Min","Mean") ) )
	}

	stringName2 = c('aic','w2v','inferSent') #'simDef'
	name2 = c()
	end = length(stringName2)
	for (i in 1: (end-1)) {
		for (j in (i+1):end) {
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
	}
	print ( dim(result) )


	colnames(next_data_file) = c('g1','g2','label','minBMA','meanBMA')
	next_data_file$label = NULL

	result = merge ( result, next_data_file, by=c('g1','g2'))
	print ( dim (result) )
	types = c(types,'minBMA','meanBMA')


	positiveSet = subset ( result , result$label==1 )
	negativeSet = subset ( result , result$label==0 )

	print ('using seed to keep same subsample')
	set.seed(2222)

	positiveSet = result [ which(result[,'label']==1), ]
	print ('positiveSet size')
	print ( dim(positiveSet) )
	positiveSet = positiveSet[sample(nrow(positiveSet)),]

	negativeSet = result [ which(result[,'label']==0), ]
	print ('negativeSet size')
	print ( dim(negativeSet) )
	negativeSet = negativeSet[sample(nrow(negativeSet)),]


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
	for ( i in 1:numType ) { ## for each method type
		typeI = aucOutput[[ types[i] ]][[3]]
		label = aucOutput[[ types[i] ]][[4]]
		for (j in i:numType) {
			typeJ = aucOutput[[ types[j] ]][[3]]
			testResult = AUC.test( typeI, label, typeJ, label )
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

print (data_type)
print (ret)

# dataUsedInAuc = dataUsedInAuc[c('g1','g2','label')]
# colnames(dataUsedInAuc) = c('gene1','gene2','label')
# write.table( dataUsedInAuc, 'dataUsedInAuc.csv', sep=" ", quote=F, row.names=FALSE)

}


