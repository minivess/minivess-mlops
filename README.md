# MiniVess MLOps

[![Docker (Env)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build-env_image.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build-env_image.yml)
[![Docker (Jupyter)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build-jupyter_image.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build-jupyter_image.yml)
[![Docker (Train)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build_train_image.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build_train_image.yml)
<br>[![Test EE (Data)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/test_dataload.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/test_dataload.yml)
[![Test EE (Train)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/test_train.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/test_train.yml)


_(in-progress)_ MLOPS for more end-to-end reproducible pipeline for the dataset published in the article:

Charissa Poon, Petteri Teikari _et al._ (2023):
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging",
Scientific Data 10, 141 doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8) see also: [Dagshub](https://dagshub.com/petteriTeikari/minivess_mlops)

## Tooling

* **Code:** Based on MONAI
* **Configuration management:** Hydra
* **Code Versioning:** Github
* **CI**: Github Actions
* **Containerization**: Docker
* **Data Versioning:** DVC
* **Experiment Tracking:** MLFlow (with WANDB as an extra option)
* **Model Store:** MLFlow
* **Serving:** BentoML (with optional FastAPI in front of it)
* **Training Orchestration:** Prefect

### Optional possibilities

* **IaC**: Cloud Development Kit for Terraform (CDKTF) or Pulumi
* **GUI Demo App**: Gradio/Streamlit with HuggingFace?
* **Monitoring**: Grafana (Prometheus from BentoML)
* **BentoML Docker orchestration**: Kubernetes (or something making k8 easier)

## Wiki

See some background for decisions, and tutorials: https://github.com/petteriTeikari/minivess_mlops/wiki

## TODO!

See cards on [Github Projects](https://github.com/orgs/minivess-mlops/projects/1)
