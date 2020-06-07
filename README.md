#install apex 
  intsll apex in ./
  ```
  https://blog.csdn.net/jacke121/article/details/103252605
```
#Pipeline
## 0 train supernet

   if the gpu memory >15G: use single gpu is nenough
   
 ```  
   run      CUDA_VISIBLE_DEVICES=2,3  python train_autodeeplab.py --dataset 2d --no_val --filter_multiplier 2
 ```
##1 decode supernet
 The check point stored in ./run/2d/deeplab-resnet/experiment_xx（the last epoch）
  ```  
  CUDA_VISIBLE_DEVICES=0 python decode_autodeeplab.py --dataset 2d --resume ./run/2d/deeplab-resnet/experiment_16/checkpoint.pth.tar
 ```
 ## 2 retrain arch
   ```  
   CUDA_VISIBLE_DEVICES=0 python train.py --net_arch ./run/2d/deeplab-resnet/experiment_xx/network_path.npy  
   --cell_arch ./run/2d/deeplab-resnet/experiment_xx/genotype.npy  --filter_multiplier 2 --dataset 2d
 ```
 ##derive model
 ##retrain model


