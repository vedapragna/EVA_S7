# EVA_S7
#### Experiment:
      Build a Model on CIFAR10 dataset to achieve atleast 80% accuracy

#### Target:
      1. 4 Convolution blocks with 
            * Atleast One layer with Depthwise Separable Convolution
            * Atleast One layer withDilated Convolution
            * Three Max pooling layers 
      2. Number of paramaters should be leass than 1M
      3. Must use GAP layer
      4. Total Receptive field should be greater than 44


#### Result:
     * Total Parameters = 89,248
     * L2 Regularised model with weight decay of 0.005
     * Best Train Accuracy =  89.2%
     * Best Test Accuracy =  80.8%
     * Receptive field = 74
