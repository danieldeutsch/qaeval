# QAEval
This repository contains the code for the QAEval metric from [Towards Question-Answering as an Automatic Metric for Evaluating the Content Quality of a Summary](http://arxiv.org/abs/2010.00490).
We have included here only the minimal amount of code to run the metric and does not include the code to run the experiments in the paper.

The easiest way to run the metric end-to-end is to use the wrapper implementation included in [SacreROUGE](https://github.com/danieldeutsch/sacrerouge/blob/master/doc/metrics/qaeval.md).

The pretrained question generation model can be downloaded [here](https://drive.google.com/file/d/1vVhRgLtsQDAOmxYhY5PMPnxxHUyCOdQU/view?usp=sharing) and the pretrained question answering model can be downloaded [here](https://drive.google.com/file/d/1q2Z3FPP9AYNz0RJKHMlaweNhmLQoyPA8/view?usp=sharing).

## Known Differences from Paper
There are several known differences between the implementation here and the one we used for the experiments in the paper.

- For the paper, we used a string equals and ROUGE-1 F1 with stemming to calculate the EM and F1 scores between the QA model's predicted answer and the ground-truth answer.
This implementation uses the SQuAD EM/F1 implementations from the Transformers library.
We made this decision to not create a dependency on ROUGE.

- The AllenNLP version used here is 1.1.0.
For the paper it was 1.0.0rc3.
The 1.0.0rc3 version requires Transformers 2.8.0.
After upgrading the AllenNLP version, we can now use Transformers 3.0.2, but this made the question-generation model used for the paper incompatible, so it had to be retrained.
The retraining scripts are [here](experiments/generation/Readme.md).
The required changes to the code for this were to pass `use_cache=False` to the BART call.

## Citation
```
@misc{deutsch2020questionanswering,
      title={Towards Question-Answering as an Automatic Metric for Evaluating the Content Quality of a Summary}, 
      author={Daniel Deutsch and Tania Bedrax-Weiss and Dan Roth},
      year={2020},
      eprint={2010.00490},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
