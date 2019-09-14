# Cross Model Retrieval

Pytorch implementation of cross model retrieval on dataset IAPR TC-12.

## Dataset Introduction



## Network Structure



可以整个网络一起训练

```python
#定义优化器
optimizer = optim.Adam(list(imageNet.parameters())+list(textExtractor.parameters())+list(textHashNet.parameters()), lr=args.lr, weight_decay=args.weight_decay)
```

```python
#训练的每个setp优化时
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

也可以分开imageNet和textNet两个部分，各自用不同的optimizer和lr分别训练

```python
#定义优化器
optimizer_image = optim.Adam(imageNet.parameters(), lr=args.image_lr, 	                weight_decay=args.weight_decay)
optimizer_text = optim.Adam(list(textExtractor.parameters())+
  list(textHashNet.parameters()), lr=args.text_lr,weight_decay=args.weight_decay)
```
```python
#训练的每个setp优化时
#backward计算网络的梯度，先更新imageNet部分，此时梯度已经用掉了。
#然后再backward计算一次梯度，更新textNet。
optimizer_image.zero_grad()
# retain_graph保留计算图, 接下面的backward，不然他直接就把图释放了
loss.backward(retain_graph=True) 
optimizer_image.step()

optimizer_text.zero_grad()
loss.backward()
optimizer_text.step()
```

但其实效果差不多，所以不如直接把ImageNet和TextNet合并成一个model，一起训练。