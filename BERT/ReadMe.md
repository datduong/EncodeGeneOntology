
### BERT has 2 phases to train parameters in the Transformer model.

#### Phase 1:
This phase requires us to train a masked language model, and next-sentence prediction.

#### Phase 2:
We train Transformer to specifically produce GO embeddings. We originally intended to jointly train Phase 1 and Phase 2, but some prelim results did not show significant improvement, and runtime was also too slow. 

