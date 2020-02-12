

### Demo scripts

In this folder, we provide shell scripts to demonstrate how to run BERT and BiLSTM. 

1. To run BERT, you need to do 2 steps. In Phase 1, you need to fine-tune the pretrained Google BERT on the Gene Ontology which has 2 objectives (1) masked language model and (2) next sentence prediction. Observe that the model will not specifically made to produce a GO embeddings. You can download our already trained [Phase 1 model here](https://drive.google.com/drive/folders/129UObLlhnp0RK6MQAS7waUF-k4SuGV-u), otherwise you can follow the script [here](https://github.com/datduong/EncodeGeneOntology/blob/master/BERT/PretrainBertPhase1/run.sh). In Phase 2, you continue to train the model to specifically produce GO embeddings. For Phase 2, we have the following scripts
  1. [BERT as-a-service](https://github.com/auppunda/GeneOntologyEncoders/blob/master/BertAsAService/GetVecFile.sh)
  2. BERT Layer 12 is in this folder. 

2. BiLSTM script is in this folder

3. ELMO on Gene Ontology is [found here](https://github.com/auppunda/GeneOntologyEncoders/tree/master/Elmo/encoder)

