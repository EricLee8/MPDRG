# Codes and Data for ACL2023 Submission
This repository contains the official codes for our paper at ACL 2023: EM Pre-training for Multi-party Dialogue Response Generation.

## Environmental Settings
- GPU: TITAN RTX 24G
- CUDA: 11.7
- Python: 3.9.12
- Pytorch: 1.12.0
- Other dependencies: see requirements.txt

We strongly recommand that you run our codes on the same settings with Docker or Anaconda to ensure reproducibility. You can run `$ pip3 install -r requirements.txt` to install other dependencies.

## Running
First, you should download the pre-trained model from [the Google Drive](https://drive.google.com/file/d/1y6N7L13kHlkC6t-E0tgkDV2UPD5OGu-z/view?usp=sharing), then create a new folder named `pretrain_models` under the same path of this README file and put the downloaded model in this folder (`./pretrain_models/mpdrg.pth`).

Then, you should unzip the data.zip file to get the datset.

After that, you can run the following command to fine-tune the pre-trained model on the Ubuntu IRC benchmark:
```
$ bash run_finetune_ubuntu.sh [GPU_ID]
```

## Citation
If you find our paper and repository useful, please cite us in your paper:
```
@inproceedings{li-zhao-2023-em,
    title = "{EM} Pre-training for Multi-party Dialogue Response Generation",
    author = "Li, Yiyang  and
      Zhao, Hai",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.7",
    pages = "92--103",
}
```
