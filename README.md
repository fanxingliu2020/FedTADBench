# Code for the HPCC-2022 paper "FedTADBench: Federated Time-series Anomaly Detection Benchmark"

Note: All the command in the code blocks are based on the root directory of this repository.

## Experiments Settings

* Time Series Anomaly Detection Methods: DeepSVDD, GDN, LSTM_AE, Tran_AD, USAD
* Federated Learning Methods: FedAvg, FedProx, SCAFFOLD, MOON
* Datasets: SMD, SMAP, PSM
* GPUs:
  * GDN: NVIDIA Geforce GTX 1080TI (11GB) GPU
  * Others: NVIDIA Geforce RTX 3090 (24GB) GPU
* Python Packages:
  * GDN: Python 3.7, PyTorch 1.4.0, torch-geometric 1.5.0
  * Others: Python 3.8, PyTorch 1.11.0, torch-geometric 2.1.0.post1

## Datasets

### SMD
* directory structure:
  * data/datasets/smd/SMD/raw
    * train
    * test
    * test_label
    * list.txt
* url: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset

### SMAP
* directory structure:
  * data/datasets/smd/SMAP/raw
    * train
    * test
    * test_labels
    * list.txt

### PSM
* directory structure:
  * data/datasets/smd/PSM/raw
    * train
    * test
    * test_labels
    * list.txt
* url: https://github.com/eBay/RANSynCoders/tree/main/data

## Usage

before running the code, please create directories:
* fltsad
  * pths
  * pths_average
  * scores
  * scores_average

### Experiments Corresponding to Table II

#### Centralized

```bash
# for Time Series Anomaly Detection Method: DeepSVDD
cd algorithms/DeepSVDD
python deepsvdd_exp_[Dataset(smd/smap/psm)].py
```
```bash
# for Time Series Anomaly Detection Method: GDN
cd algorithms/GDN
python gdn_exp_[Dataset(smd/smap/psm)].py
```
```bash
# for Time Series Anomaly Detection Method: LSTM_AE
cd algorithms/DeepSVDD
python lstmae_exp_[Dataset(smd/smap/psm)].py
```
```bash
# for Time Series Anomaly Detection Method: TranAD
cd algorithms/DeepSVDD
python tranad_exp_[Dataset(smd/smap/psm)].py
```
```bash
# for Time Series Anomaly Detection Method: USAD
cd algorithms/DeepSVDD
python usad_exp_[Dataset(smd/smap/psm)].py
```

#### Federated

```bash
# for Time Series Anomaly Detection Methods: GDN, LSTM_AE, Tran_AD, USAD
python main.py --alg [Federated Learning Method (fedavg/fedprox/scafold/moon)] --tsadalg [Time Series Anomaly Detection Method (gdn/lstm_ae/tran_ad/usad)] --dataset [Dataset Name(smd/smap/psm)]

# for Time Series Anomaly Detection Method: DeepSVDD
python main_svdd.py --alg [Federated Learning Method (fedavg/fedprox/scafold/moon)] --tsadalg deepsvdd --dataset [Dataset Name(smd/smap/psm)]
```

### Experiments Corresponding to Fig. 5.

```bash
python main.py --alg [Federated Learning Method (fedavg/fedprox/scafold/moon)] --tsadalg usad --dataset psm --beta [0.1/0.5/5/1000000]
```
Note: when setting beta >= 10000 in our code, it will be a uniform distribution instead of Dirichlet distribution with beta equals to that beta value.

### Experiments Corresponding to Fig. 6.

```bash
# for Time Series Anomaly Detection Methods: GDN, LSTM_AE, Tran_AD, USAD
python main_clients_average.py --alg [Federated Learning Method (fedavg/fedprox/scafold/moon)] --tsadalg [Time Series Anomaly Detection Method(gdn/lstm_ae/tran_ad/usad)] --dataset psm

# for Time Series Anomaly Detection Method: DeepSVDD
python main_clients_average_svdd.py --alg [Federated Learning Method (fedavg/fedprox/scafold/moon)] --tsadalg deepsvdd --dataset psm
```

### Results Corresponding to Table III, Table IV, Fig. 4. and Fig. 6.

After running the experiments before, the anomaly scores are stored in ``fltsad/scores`` and ``fltsad/scores_average``.
Run
```bash
python calc_precision_recall_f1_point_adjust.py
```
to get the Precision, Recall, F1 without/with point adjusting.

## Code Reference

* DeepSVDD: https://github.com/lukasruff/Deep-SVDD-PyTorch
* GDN: https://github.com/d-ailin/GDN
* LSTM_AE: https://github.com/astha-chem/mvts-ano-eval
* TranAD: https://github.com/imperial-qore/TranAD
* USAD: https://github.com/manigalati/usad
* SCAFFOLD：https://github.com/Xtra-Computing/NIID-Bench
* MOON: https://github.com/QinbinLi/MOON