# GenomePleiotropy

[PDF]() **Abstract**: Biological evolution shapes the body and brain of living creatures together over time. By contrast, in evolutionary
robotics, the co-optimization of these subsystems remains challenging. Conflicting mutations cause dissociation between
morphology and control, which leads to premature convergence and reduces diversity. The importance of genetic design in
this remains underexposed. Current approaches are divided between a single, pleiotropic genetic encoding and two
isolated encodings representing morphology and control. We performed a comparative analysis between these two
configurations on a soft robot locomotion task. Additionally, we incorporated two currently unexplored alternatives that
drive these configurations to their logical extremes. Our results demonstrate that pleiotropic representations yield
superior performance in fitness, morphological diversity, and robustness towards premature convergence. Moreover, we
showcase the importance of shared structure in the pleiotropic representation of robot morphology and control to achieve
this performance gain. These findings provide valuable insights into genetic encoding design, which supply practitioners
with a theoretical foundation to pursue efficient brain-body co-optimization.

## Usage
Create the anaconda environment:
```shell
conda env create -f environment.yml
```

Run an experiment:
```shell
python -m pleiotropy.main --name {experiment_name} --params pleiotropy/experiment_configs/{configuration}
```

## Citation
