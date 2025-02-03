***\*UniSAM: A universal multi-modal SAM framework for mirror and glass segmentation\****

SAIL (Statistic Analysis And Intelligent Learning) Lab of NWU

We provide related codes and configuration files to reproduce the "UniSAM: A universal multi-modal SAM framework for mirror and glass segmentation"



***\*Introduction:\****

We are the first to unify mirror and glass segmentation tasks into a single framework and introduce a universal multi-modal SAM framework for mirror and glass segmentation (UniSAM). UniSAM is a multi-modal segmentation framework that leverages an additional X modality (depth image or thermal image) to enhance SAMs performance in RGB images. The framework comprises two core components:  Control and Prompt Auto Generation Component (CPAG) for generating control features and prompts, and Feature Controlled Encoder Decoder Component (FCED) for controlling the encoder and decoder of SAM. Experimental results on public datasets demonstrate that UniSAM achieves state-of-the-art performance in mirror and glass segmentation.



***\*Schematic diagram of UniSAM:\****

![fig1](C:\Users\18329\Desktop\UniSAM_README\fig1.jpg)



![fig3](C:\Users\18329\Desktop\UniSAM_README\fig3.jpg)



***\*Example images：\****

![fig5](C:\Users\18329\Desktop\UniSAM_README\fig5.jpg)



![fig6](C:\Users\18329\Desktop\UniSAM_README\fig6.jpg)



***\*Train the model:\****

```python
python train_sam_self_mirror_416.py / train_sam_self_glass_480.py
```



***\*Inference Dataset (Please load pre-training weights in advance):\****

```python
python test_sam_self_mirror_416.py / test_sam_glass_self_480.py
```



***\*Predict maps (mirror & glass):\****

Link: https://pan.baidu.com/s/1BjuFtLi5Bd1huEXv7n7y0g?pwd=cgrj 

Extract code: cgrj



***\*Dataset (Mirror & Glass):\****

Link: https://pan.baidu.com/s/1mTEn-zOjQB9mIzkBqmnU5g?pwd=c9hs 

Extract code: c9hs



***\*UniSAM pretrained weights (mirror & glass):\****

Link: https://pan.baidu.com/s/1XKK9yblpEOklavNv_PPRxA?pwd=ghnr 

Extract code: ghnr



***\*Requirements:\****

```python
pip install -r requirements.txt
```

