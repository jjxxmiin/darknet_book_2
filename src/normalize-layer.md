# normalize layer

## forward\_normalization\_layer

```c
void forward_normalization_layer(const layer layer, network net)
{
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    scal_cpu(w*h*c*layer.batch, 0, layer.squared, 1);

    for(b = 0; b < layer.batch; ++b){
        float *squared = layer.squared + w*h*c*b;
        float *norms   = layer.norms + w*h*c*b;
        float *input   = net.input + w*h*c*b;
        pow_cpu(w*h*c, 2, input, 1, squared, 1);

        const_cpu(w*h, layer.kappa, norms, 1);
        for(k = 0; k < layer.size/2; ++k){
            axpy_cpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
        }

        for(k = 1; k < layer.c; ++k){
            copy_cpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            if(prev >= 0)      axpy_cpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            if(next < layer.c) axpy_cpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
    mul_cpu(w*h*c*layer.batch, net.input, 1, layer.output, 1);
}
```

함수 이름: forward\_normalization\_layer

입력:&#x20;

* layer (layer 구조체)
* net (network 구조체)

동작:&#x20;

* 입력 데이터에 대한 정규화를 수행하는 함수입니다.&#x20;
* 입력 데이터는 layer 구조체와 network 구조체에서 가져옵니다.&#x20;
* 함수는 주어진 입력 데이터에서 입력 크기와 일치하는 정규화된 출력을 계산합니다.&#x20;
* 계산에는 입력 데이터의 각 채널에 대해 스퀘어드 값, 노름, 입력 데이터의 텐서 값을 사용합니다.&#x20;
* 계산이 완료되면 출력 데이터가 layer 구조체의 출력 포인터로 설정됩니다.

설명:

* layer: 정규화를 수행할 레이어의 정보를 담고 있는 layer 구조체입니다.
* net: 입력 데이터를 담고 있는 network 구조체입니다.
* w, h, c: layer 구조체의 너비, 높이, 채널 수입니다.
* batch: 입력 데이터의 배치 크기입니다.
* squared: 입력 데이터의 각 채널에 대한 스퀘어드 값입니다.
* norms: 입력 데이터의 각 채널에 대한 노름 값입니다.
* input: 입력 데이터의 텐서 값입니다.
* kappa: 노름 계산에 사용되는 값입니다.
* alpha: 노름 계산에 사용되는 값입니다.
* size: 노름 계산에 사용되는 필터 크기입니다.
* beta: 출력 데이터에 대한 승수 값입니다.
* output: 정규화된 출력 데이터의 포인터입니다.
* 입력 데이터의 각 채널에 대해 스퀘어드 값을 계산하고, 노름 값을 계산합니다. 그리고 나서 이 값들을 사용하여 입력 데이터를 정규화된 출력으로 계산합니다. 마지막으로, 계산된 출력 값이 layer 구조체의 출력 포인터로 설정됩니다.



## backward\_normalization\_layer

```c
void backward_normalization_layer(const layer layer, network net)
{
    // TODO This is approximate ;-)
    // Also this should add in to delta instead of overwritting.

    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
    mul_cpu(w*h*c*layer.batch, layer.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_normalization\_layer

입력:&#x20;

* layer (정규화 레이어 구조체)
* net (신경망 구조체)

동작:&#x20;

* 정규화 레이어의 역전파를 수행하고, 입력값에 대한 델타 값을 계산합니다.

설명:

* 이 함수는 정규화 레이어의 역전파를 수행하는 함수입니다.
* 입력으로는 정규화 레이어 구조체(layer)와 신경망 구조체(net)가 필요합니다.
* 함수 내부에서는 먼저 정규화 레이어의 베타 값에 대한 거듭제곱 계산을 수행합니다.
* 이후, 정규화 레이어의 델타 값과 거듭제곱 계산 결과를 곱한 뒤, 입력값에 대한 델타 값을 계산합니다.
* 계산된 결과는 net.delta에 덮어쓰기 되며, 정확도에는 약간의 오차가 있을 수 있습니다.



## resize\_normalization\_layer

```c
void resize_normalization_layer(layer *layer, int w, int h)
{
    int c = layer->c;
    int batch = layer->batch;
    layer->h = h;
    layer->w = w;
    layer->out_h = h;
    layer->out_w = w;
    layer->inputs = w*h*c;
    layer->outputs = layer->inputs;
    layer->output = realloc(layer->output, h * w * c * batch * sizeof(float));
    layer->delta = realloc(layer->delta, h * w * c * batch * sizeof(float));
    layer->squared = realloc(layer->squared, h * w * c * batch * sizeof(float));
    layer->norms = realloc(layer->norms, h * w * c * batch * sizeof(float));
}
```

함수 이름: resize\_normalization\_layer

입력:

* layer \*layer: normalization\_layer 구조체의 포인터
* int w: normalization\_layer의 가로 크기
* int h: normalization\_layer의 세로 크기

동작:

* 입력으로 받은 normalization\_layer의 가로, 세로 크기를 업데이트합니다.
* layer의 출력 크기 및 입력 크기를 업데이트합니다.
* layer의 출력, 델타, 제곱, norms 배열을 새로운 크기에 맞게 다시 할당합니다.

설명:

* 이 함수는 normalization layer의 크기를 조정하기 위해 사용됩니다.
* 입력으로 받은 가로, 세로 크기를 이용해 layer 구조체의 필드 값을 업데이트합니다.
* 이 함수는 입력 크기, 출력 크기 및 출력 배열의 크기를 재할당합니다.
* realloc 함수를 사용하여 메모리를 재할당하므로, 이전에 할당되어 있던 메모리를 자동으로 해제합니다.
* 이 함수는 입력으로 받은 normalization\_layer의 포인터를 직접 수정하므로, 반환 값은 없습니다.



## make\_normalization\_layer

```c
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
    fprintf(stderr, "Local Response Normalization Layer: %d x %d x %d image, %d size\n", w,h,c,size);
    layer layer = {0};
    layer.type = NORMALIZATION;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.kappa = kappa;
    layer.size = size;
    layer.alpha = alpha;
    layer.beta = beta;
    layer.output = calloc(h * w * c * batch, sizeof(float));
    layer.delta = calloc(h * w * c * batch, sizeof(float));
    layer.squared = calloc(h * w * c * batch, sizeof(float));
    layer.norms = calloc(h * w * c * batch, sizeof(float));
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    layer.forward = forward_normalization_layer;
    layer.backward = backward_normalization_layer;

    return layer;
}
```

함수 이름: make\_normalization\_layer

입력:

* int batch: 배치 크기
* int w: normalization\_layer의 가로 크기
* int h: normalization\_layer의 세로 크기
* int c: normalization\_layer의 채널 수
* int size: normalization을 수행하는 윈도우의 크기
* float alpha: 정규화의 강도를 결정하는 매개변수
* float beta: 정규화 상수
* float kappa: 정규화를 수행할 때 추가하는 값

동작:

* 입력으로 받은 값들을 이용해 normalization\_layer 구조체를 만듭니다.
* layer의 필드 값을 초기화합니다.
* 입력과 출력 배열을 할당합니다.
* layer의 forward, backward 함수를 설정합니다.

설명:

* 이 함수는 normalization layer를 만들기 위해 사용됩니다.
* 입력으로 받은 값들을 이용해 normalization\_layer 구조체를 만듭니다.
* layer의 type 필드는 NORMALIZATION으로 설정됩니다.
* layer의 출력, 델타, 제곱, norms 배열을 초기화합니다.
* layer의 forward, backward 함수를 설정합니다.
* 이 함수는 normalization\_layer 구조체를 반환합니다.

