This directory contains the code to train the question-generation model.
The original model used for the paper experiments could not be directly loaded by the code after updating the AllenNLP and Transformers packages, so we have retrained the model here.

To retrain the model, run:
```
sh experiments/generation/setup.sh
sh experiments/generation/train.sh
```