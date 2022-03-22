# Multi-scale Selective Hierarchical biQuadratic Pooling(MSHQP)
 A implementation of Fine-grained image classification via multi-scale selective hierarchical biquadratic pooling(MSHQP)


## Requirements
- python 3.8
- pytorch 1.7.1

## Train


Step 1.
- Download the resnet pre-training parameters.

- Download the dataset.
[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
[Stanford-Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
[FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)


Step 2.
- Set the data path in the config.py.


Step 3. Train the fc layer only.(Take CUB-200-2011 as an example)
-Resnet-34:  python cub_net34_train_firststep.py
-Resnet-152: python cub_net152_train_firststep.py


Step 4. Fine-tune all layers.
-Resnet-34:  python cub_net34_train_finetune.py
-Resnet-152: python cub_net152_train_finetune.py


    | dataset       | Resnet-34|  Resnet-152|
    | --------      | :-----:  | :-----:    |
    | CUB-200-2011  | 88.5%    |   89.0%    |
    | Stanford-Cars | 94.4%    |   94.9%    |
    | FGVC-Aircraft | 92.8%    |   93.4%    |

## Resnet-50 and VGG
 We are still finishing the final code of Resnet-50 and VGG. Moreover, the model structure of Resnet-50 is very similar to that of Resnet-152, so you can freely modify it to get what you want.
