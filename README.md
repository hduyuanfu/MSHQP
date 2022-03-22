# Multi-scale Selective Hierarchical biQuadratic Pooling(MSHQP)
 A implementation of Fine-grained image classification via multi-scale selective hierarchical biquadratic pooling(MSHQP).[here](https://doi.org/10.1145/3492221)


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
 
## Trained model
- Download the trained model.
[cub_result_net34.pth](https://drive.google.com/file/d/1u44aGX9P5BclQnJWlWS3AE1-ZXbEXDcQ/view?usp=sharing)
[cub_result_net152.pth](https://drive.google.com/file/d/1vTFC_rmUXtZZks3PSKVvjis-cw3od8m3/view?usp=sharing)
[car_result_net34.pth](https://drive.google.com/file/d/1G3qFx_gGfye0C1vswVLiAStNF3u282uJ/view?usp=sharing)
[car_result_net152.pth](https://drive.google.com/file/d/1q190LaMp0IBZfJZXuB6X_PAI2zCqpbKo/view?usp=sharing)
[air_result_net34.pth](https://drive.google.com/file/d/1ioDw2DEdgz5dZ9R2QINw00pM4JVkwkp2/view?usp=sharing)
[air_result_net152.pth](https://drive.google.com/file/d/1uDNf4LvUybhA9rMiRc6hK2zgSrBpif2A/view?usp=sharing)
