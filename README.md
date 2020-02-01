
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

Consider the example below. We would expect child-parent terms to have similar vector embeddings; whereas, two unrelated terms should have different embeddings. 

![GoTermExampl](Figure/GoTermExample.png)

### Defintion encoder 

We embed the [definition](https://www.ebi.ac.uk/QuickGO/term/GO:0075295) of a term. The key is that child-parent terms often have simlar defintions, so that we can embed them into comparable vectors. 

All models are already trained, and ready to be used. You can download the embeddings here. 


