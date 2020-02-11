


library("GOSemSim") ## **** use R/3.3.0. We installed this lib some times ago, some IC might have changed in recent times.

getIC = function (database,ont='Bp'){
	goDatabase = godata(database, ont=ont)
	IC2write = goDatabase@IC
	# IC2write = IC2write[IC2write<Inf] ## write only GO with valid IC
	ICspec1 = data.frame ( names(IC2write), IC2write )
	colnames(ICspec1) = c("go","ic")
	return (ICspec1)
}


dataName = c ( 'org.Ce.eg.db' , 'org.Mm.eg.db', 'org.Hs.eg.db', 'org.Sc.sgd.db' , 'org.Dm.eg.db' )
specName = c( 'Worm', "Mouse", 'Human', 'Yeast', 'Fly')

for (ont in c('BP','MF','CC')) {
  for (i in seq(1,length((dataName)))) {
    spec1 = getIC(dataName[i], ont )
    name1 = specName[i]
    write.table(  as.matrix( spec1[,c("go","ic")] , ncol=2 ) , sep="\t",
                  file=paste0("/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/ICdata/",name1,"Ic",ont,".txt"),quote=F,row.names=FALSE,col.names=FALSE)

  }
}


