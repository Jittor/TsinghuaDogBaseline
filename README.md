# TsinghuaDog Baseline 

### 使用 ResNet-50 进行细粒度识别 

#### 依赖环境:

```bash
Jittor 
Numpy 
tqdm
pillow
```



#### 使用方法：

从[官网](https://cg.cs.tsinghua.edu.cn/ThuDogs/ )下载 TsinghuaDog 数据集，并将 root_dir 指向 TsinghuaDog 数据集。

使用如下一行命令即可进行训练和测试

```bash
python main.py # （默认使用 GPU 执行）预计消耗显存 13G，如果显存不够可减小 batch_size 或者将代码中 jt.flags.use_cuda 改成 0，即可使用 cpu 执行。
```

 





