
BertAveWordClsSepLayer12/cosine.768.reduce300LastLayerCLS2Vec

/local/auppunda/auppunda/goAndGeneAnnotationMar2017/RandomGOAnalysis/Elmo_gs/cosine.Elmo768.Linear768
/local/auppunda/auppunda/goAndGeneAnnotationMar2017/RandomGOAnalysis/BERT/cosine.Cls768.11+12

encoder='BERT'
name='cosine.Cls768.11+12'
cd /local/datdb/goAndGeneAnnotationMar2017/RandomGOAnalysis/$encoder/$name
cat ParentChild_go_analysis_cc.$encoder.degree.txt ParentChild_go_analysis_mf.$encoder.degree.txt ParentChild_go_analysis_bp.$encoder.degree.txt random_go_analysis_cc.$encoder.degree.txt random_go_analysis_mf.$encoder.degree.txt random_go_analysis_bp.$encoder.degree.txt > go_analysis_all.$encoder.degree.txt



encoder='AIC'
cd /u/scratch/d/datduong/goAndGeneAnnotationMar2017/RandomGOAnalysis/$encoder/

cat Human_ParentChild_go_analysis_cc.degree.tsv Human_ParentChild_go_analysis_mf.degree.tsv Human_ParentChild_go_analysis_bp.degree.tsv Human_random_go_analysis_cc.degree.tsv Human_random_go_analysis_mf.degree.tsv Human_random_go_analysis_bp.degree.tsv > Human_go_analysis_all.$encoder.degree.txt


go1\tgo2\tlabel\ttype\tscore\tdegree1\tdegree2\tMeanDegree

go1\tgo2\tic1\tic2\tlabel\ttype\tscore\tdegree1\tdegree2\tMeanDegree


encoder='AIC'
cd /u/scratch/d/datduong/goAndGeneAnnotationMar2017/RandomGOAnalysis/$encoder/

cat Fly_ParentChild_go_analysis_cc.degree.tsv Fly_ParentChild_go_analysis_mf.degree.tsv Fly_ParentChild_go_analysis_bp.degree.tsv Fly_random_go_analysis_cc.degree.tsv Fly_random_go_analysis_mf.degree.tsv Fly_random_go_analysis_bp.degree.tsv > Fly_go_analysis_all.$encoder.degree.txt
