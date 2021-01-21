<<<<<<< HEAD
# TimeGAN-Pytorch
Time generative adversarial network (Pytorch)   
Please change the branch to `master`
=======
# TimeGAN: A Pytorch Implementation
**This is a PyTorch implementation of the TimeGAN in "[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)" (Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, NIPS, 2019).**

## TimeGAN introduction
論文簡介:本篇論文主要的特點在於透過加入 embedding & recovery 的方式， generator 不是直接產生 time-series data，而是產生 syntheitc latent codes 的同時與 real data 的 latent codes 進行相似度比較，讓 generator 可以更好的學習 time-series data 的底層時間動態。


## TimeGAN architecture
![TimeGAN architecture](https://github.com/kent1201/TimeGAN-Pytorch/blob/main/TimeGAN%20architecture.jpg)

## Action base Datasets
使用 20 類人體下肢動作訊號數據集對 TimeGAN 進行訓練。總計 3796 筆，每筆動作包含三軸加速度，陀螺儀與四元數資料，經過轉換後總計 27 個維度。其中訓練資料 2847 筆，測試資料 949 筆。以下為 20 類動作資料筆數:
![Action base datasets](https://github.com/kent1201/TimeGAN-Pytorch/blob/master/src/image.png)

## Project architecture
`\Loss` 內含不同 train stages (encoder, decoder, generator, ...) 會用到的 loss。  

`\Network` 包含 `encoder.py`, `decoder.p`y, `generator.py`, `supervisor.py`, `discriminator.py` 五個不同部件，每個部件可用 rnn. lstm, gru, tcn 所替換。simple_discriminator.py, simple_predictor.py 則是用來評估 real data 與 syntheitc data 之間的差異所用的架構。 

`Configure.ini` 所有參數的設定檔。 

`requirements.txt` 套件要求。

`utils.py` 包含 train test data loader, random generator 等功能。  

`dataset.py, dataset_preprocess.py` 針對 Action base Datasets 的 Pytorch 操作。目前不提供 datasets。  

`train.py` TimeGAN 訓練部分。訓練部分分成三個階段: 
* `Stage 1` 訓練 encoder, decoder。
* `Stage 2` 訓練 encoder, supervisor, generator。
* `Stage 3` 聯合訓練 discriminator, generator, supervisor, encoder, decoder。  
訓練好模型將以日期作為劃分，儲存模型。

`generate_data.py` 經由訓練好的模型產生 synthetic data。 

`data_visualization.py` 對 real data 與 synthetic data 做 PCA, t-SNE 二維可視化圖。 

`test.py` 使用 discriminator 評估 real data 與 synthetic data 相似度 (以錯誤率當標準，越低越好)。使用 predictor 對 synthetic data 進行訓練，並在 real data 上進行預測(以 MAE 做標準，越低越好)。詳細標準可參考 "[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)"。 

`hyper_optimize.py` 開發測試中的功能，用於參數最佳化。

## Project results
* Original Accuracy: 77.76 %
<br/> </br>
![TRTR](https://github.com/kent1201/TimeGAN-Pytorch/blob/master/src/TRTR_7776.jpg)
<br/> </br>
* Train on syntheitc, test on real Accuracy: 76.73 %
<br/> </br>
![TSTR](https://github.com/kent1201/TimeGAN-Pytorch/blob/master/src/TSTR_7673.jpg)
<br/> </br>
* Train on mix, test on real Accuracy: 83.47 %
<br/> </br>
![TMTR](https://github.com/kent1201/TimeGAN-Pytorch/blob/master/src/TMTR_8347.jpg)
<br/> </br>
* t-SNE
<br/> </br>
![tSNE1-10](https://github.com/kent1201/TimeGAN-Pytorch/blob/master/src/tSNE1_10.jpg)
<br/> </br>
![tSNE11-20](https://github.com/kent1201/TimeGAN-Pytorch/blob/master/src/tSNE11_20.jpg)
<br/> </br>

## Requirements
* conda 4.8.2
```bash
conda install --yes --file requirements.txt
``` 
or
```bash
pip install -r requirements.txt
```
## How to use 
>Set the [configure.ini](https://github.com/kent1201/TimeGAN-Pytorch/blob/master/src/TimeGAN-Configure.pptx) 
>
>conda create your environment 
>
>conda activate your environment 
>
>pip install the requirments 
```python
python train.py
python generate_data.py
python data_visualization.py
python test.py
```
* **Notice** 無提供 Dataset，請自行根據使用的 dataset 自行調整程式內容。

>>>>>>> master
