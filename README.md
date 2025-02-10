# PS3C-Self Supervised Learning

## Overview

Cervical cancer is the fourth most common cancer among women globally, with over 600,000 new cases and more than 300,000 deaths annually. Early detection through Pap smear screening is crucial in reducing mortality by identifying precancerous lesions. However, traditional methods of analyzing Pap smear samples are resource-intensive, time-consuming, and highly dependent on the expertise of cytologists. These challenges underscore the need for automation in cervical cancer screening, especially in resource-limited settings.

The **Pap Smear Cell Classification Challenge (PS3C)**, part of the ISBI 2025 Challenge Program, invites participants to tackle the automated classification of cervical cell images extracted from Pap smears. Using advanced machine learning techniques, participants will develop models to classify test images into one of three categories:

- **Healthy**: Normal cells without observable abnormalities.
- **Unhealthy**: Abnormal cells indicating potential pathological changes.
- **Rubbish**: Images unsuitable for evaluation due to artifacts or poor quality.

## Getting Started

### Installation

1. Clone this repository:

    ```
    git clone https://github.com/prabhashj07/PS3C-SSL.git
    cd PS3C-SSL
    ```

2. Set up a virtual environment and activate it:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```
    pip install -r requirements.txt
    ```

4. Add environment variables:

    ```
    cp .env.example .env
    ```

### Dataset Preparation

Dataset: [Kaggle](https://www.kaggle.com/competitions/pap-smear-cell-classification-challenge/data)

To prepare the dataset, run the following command:
It will download the dataset from Google Drive and extract it to the `data/` directory.

```bash
make dataset
```

### Running the Training Script  

To train the SimCLR model and fine-tune it on Pap smear images, use the following command:  

```bash
python main.py \
    --dataset_root path/to/dataset \
    --test_dir path/to/test_images \
    --pretrain_batch_size 32 \
    --ft_batch_size 64 \
    --test_batch_size 1 \
    --pretrain_epochs 100 \
    --ft_epochs 5 \
    --pretrain_lr 1e-4 \
    --ft_lr 1e-3 \
    --out_dim 128
```
#### Arguments  

| Argument               | Description                                            | Default Value |
|------------------------|--------------------------------------------------------|--------------|
| `--dataset_root`       | Path to the directory containing the training dataset  | `data`       |
| `--test_dir`           | Path to the directory containing test images           | `data`       |
| `--pretrain_batch_size`| Batch size for SimCLR pretraining                      | `32`         |
| `--ft_batch_size`      | Batch size for fine-tuning                             | `64`         |
| `--test_batch_size`    | Batch size for inference                               | `1`          |
| `--pretrain_epochs`    | Number of epochs for self-supervised pretraining       | `100`        |
| `--ft_epochs`         | Number of epochs for fine-tuning                       | `5`          |
| `--pretrain_lr`       | Learning rate for pretraining                          | `1e-4`       |
| `--ft_lr`             | Learning rate for fine-tuning                          | `1e-3`       |
| `--out_dim`           | Output dimension for SimCLR projection head            | `128`        |

## License

This project is licensed under the [MIT License](LICENSE).
