# A Comparative Analysis on Genome Pleiotropy for Evolved Soft Robots

[PDF]() **Abstract**: Biological evolution shapes the body and brain of living creatures together over time. By
contrast, in evolutionary robotics, the co-optimization of these subsystems remains challenging. Conflicting mutations
cause dissociation between morphology and control, which leads to premature convergence. Recent works have proposed
algorithmic modifications to mitigate the impact of conflicting mutations. However, the importance of genetic design
remains underexposed. Current approaches are divided between a single, pleiotropic genetic encoding and two isolated
encodings representing morphology and control. This design choice is commonly made ad hoc, causing a lack of consistency
for practitioners. To standardize this design, we performed a comparative analysis between these two configurations on a
soft robot locomotion task. Additionally, we incorporated two currently unexplored alternatives that drive these
configurations to their logical extremes. Our results demonstrate that pleiotropic representations yield superior
performance in fitness and robustness towards premature convergence. Moreover, we showcase the importance of shared
structure in the pleiotropic representation of robot morphology and control to achieve this performance gain. These
findings provide valuable insights into genetic encoding design, which supply practitioners with a theoretical
foundation to pursue efficient brain-body co-optimization.


https://user-images.githubusercontent.com/28387178/163192651-d1e48d17-be0c-4409-90bc-155ee3822998.mp4


## Usage

Create the anaconda environment:

```shell
conda env create -f environment.yml
```

Run an experiment:

```shell
python -m pleiotropy.main --name {experiment_name} --params pleiotropy/experiment_configs/{configuration}
```

We refer to `pleiotropy/experiment_configs/` for the hyperparameters used in every configuration. 
Metrics are logged and automatically saved on the cloud via [Weights & Biases](https://wandb.ai/site). Genomes are saved locally.

## Citation
