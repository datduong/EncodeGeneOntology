
# Encode Gene Ontology terms using their definitions or positions on the GO tree.

## The methods applied are: 

* **Defintion encoder**
  1. BiLSTM 
  2. ELMo
  3. Transformer based on BERT strategy. 
  
* **Position encoder**
  1. GCN
  2. Onto2vec

## The key objective is to capture the relatedness of GO terms by encoding them into similar vectors. 

Consider the example here. 

![GoTermExampl|50%](Figure/GoTermExample.png)

