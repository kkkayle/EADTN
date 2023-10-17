# Requirements

- numpy==1.20.3
- pandas==1.5.3
- scikit_learn==1.0.2
- torch==1.13.1
- tqdm==4.64.1

# Usage

```bash
python main.py --dataset 'BindingDB/Biosnap' \
               --scenario 'Random/Clustering' \
               --Clustering_basis 'drug/target'  # if scenario is Clustering