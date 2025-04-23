# Mlops1
Ce projet est un projet de Mlops

# MLOps Project Template

This project provides a complete structure for managing end-to-end machine learning workflows using MLOps principles.

## Structure

- `data/`: raw, processed and external data
- `src/`: scripts for data processing, modeling, monitoring
- `config/`: hyperparameters and paths
- `pipelines/`: orchestration scripts
- `notebooks/`: exploratory analysis
- `models/`: trained models
- `tests/`: unit tests

## Example Usage

```bash
# Prepare the data (Bank Marketing dataset)
python src/data/make_dataset.py

# Train the model
python src/models/train_model.py

# Monitor predictions
python src/monitoring/monitor_model.py
```

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Author
Ali Mijiyawa
