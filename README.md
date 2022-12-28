
## Usage

```python

splitter = NormalizedMultiTaskSplitter(2)

for x in dataloader:
    x = base_model(x)
    x1, x2 = splitter(x) # identity in the forward pass but combines gradients in the backward pass.
    loss1 = loss_model1(x1)
    loss2 = loss_model2(x2)
    loss = loss1 + loss2
    loss.backward()
    ... # can be used with any pytorch optimizer

```

## Results on Mulit-MNIST:
Two digits are plotted and the two tasks are to classify digits 1 and 2.
The dataloader is debugged from [github.com/WeiChengTseng/Pytorch-PCGrad.git](https://github.com/WeiChengTseng/Pytorch-PCGrad.git) and we compare our results to their implementations of pcgrad.  

<b>Preliminary results:</b> the normalized splitter is higher than pcgrad for both tasks, but probably not yet conclusive.
<b>Left:</b> accuracy for the left digit.
<b>Right:</b> accuracy for the right digit.
![MNIST result](results/summary.png)

<b> Invariance to loss scaling (no need to tune loss coeff):</b>
In this other task with assume the loss1 is multiplied by an unknown coefficient x1000. The normalized splitter is unaffected without re-weighting the losses. 

![MNIST result2](results/summary_imbalanced.png)
  
  

