
cd /u/home/c/chelseaj/project/GO_KB/Onto2Vec
file_name='VecResults_human512.lst'
perl -pi -e's/\n//gs' $file_name # first remove all the newline
perl -pi -e's/\]/\n/gs' $file_name # second, replace all the "]" with newline
perl -pi -e's/\[//gs' $file_name #third, remove all the "["

