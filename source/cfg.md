---
description: 설정
---

# /cfg

모델들의 구조를 모아 놓은 폴더 입니다.

* `alexnet`, `resnet`, `densenet`, `resxnet`, `vgg`, `darknet`, `rnn`, `gru`, `yolov1`, `yolov2`, `yolov3` 등 다양한 모델 구조를 담은 파일이 있습니다.

## structure

```
[net]
parameter1
parameter2
parameter3

[convolutional]
parameter1
parameter2
parameter3
```

* 대괄호(\[])안에 Deep Neural Network Layer의 이름이 들어가고 그 아래에는 해당 `Layer`에 필요한 파라미터 값이 추가 됩니다.
* 모델이 동작하는 순서대로 정의하면 됩니다.
