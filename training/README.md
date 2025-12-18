# training
## 概要

## 出力

## conda 環境の作成
```bash
conda env create -f training/environment.yml -n har-2way-model-training -y
```

## 実行
実行は、リポジトリのトップディレクトリで行う。

```bash
python -m training.main
```