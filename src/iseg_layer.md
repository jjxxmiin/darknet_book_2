# iseg\_layer

instance segmentation을 위한 layer입니다.

## forward\_iseg\_layer

```c
void forward_iseg_layer(const layer l, network net)
{

    double time = what_time_is_it_now();
    int i,b,j,k;
    int ids = l.extra;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

    for (b = 0; b < l.batch; ++b){
        // a priori, each pixel has no class
        for(i = 0; i < l.classes; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + i*l.w*l.h + k;
                l.delta[index] = 0 - l.output[index];
            }
        }

        // a priori, embedding should be small magnitude
        for(i = 0; i < ids; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                l.delta[index] = .1 * (0 - l.output[index]);
            }
        }


        memset(l.counts, 0, 90*sizeof(int));
        for(i = 0; i < 90; ++i){
            fill_cpu(ids, 0, l.sums[i], 1);

            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            // add up metric embeddings for each instance
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + c*l.w*l.h + k;
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    l.delta[index] = v - l.output[index];
                    axpy_cpu(ids, 1, l.output + b*l.outputs + l.classes*l.w*l.h + k, l.w*l.h, l.sums[i], 1);
                    ++l.counts[i];
                }
            }
        }

        float *mse = calloc(90, sizeof(float));
        for(i = 0; i < 90; ++i){
            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            for(k = 0; k < l.w*l.h; ++k){
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    int z;
                    float sum = 0;
                    for(z = 0; z < ids; ++z){
                        int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                        sum += pow(l.sums[i][z]/l.counts[i] - l.output[index], 2);
                    }
                    mse[i] += sum;
                }
            }
            mse[i] /= l.counts[i];
        }

        // Calculate average embedding
        for(i = 0; i < 90; ++i){
            if(!l.counts[i]) continue;
            scal_cpu(ids, 1.f/l.counts[i], l.sums[i], 1);
            if(b == 0 && net.gpu_index == 0){
                printf("%4d, %6.3f, ", l.counts[i], mse[i]);
                for(j = 0; j < ids; ++j){
                    printf("%6.3f,", l.sums[i][j]);
                }
                printf("\n");
            }
        }
        free(mse);

        // Calculate embedding loss
        for(i = 0; i < 90; ++i){
            if(!l.counts[i]) continue;
            for(k = 0; k < l.w*l.h; ++k){
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    for(j = 0; j < 90; ++j){
                        if(!l.counts[j])continue;
                        int z;
                        for(z = 0; z < ids; ++z){
                            int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                            float diff = l.sums[j][z] - l.output[index];
                            if (j == i) l.delta[index] +=   diff < 0? -.1 : .1;
                            else        l.delta[index] += -(diff < 0? -.1 : .1);
                        }
                    }
                }
            }
        }

        for(i = 0; i < ids; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                l.delta[index] *= .01;
            }
        }
    }

    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("took %lf sec\n", what_time_is_it_now() - time);
}
```

함수 이름: forward\_iseg\_layer&#x20;

입력:&#x20;

* layer 구조체와 network 구조체&#x20;

동작:&#x20;

* 이미지 분할을 위한 인스턴스 임베딩을 계산하고, 임베딩 손실을 계산하여 네트워크의 출력을 업데이트합니다.&#x20;

설명:&#x20;

* 이 함수는 YOLOv3-tiny 네트워크의 일부로 사용되는 이미지 분할 레이어를 수행합니다.&#x20;
* 이 함수는 입력 이미지의 크기와 분할된 클래스 수에 따라 출력 텐서의 크기를 결정합니다.&#x20;
* 이 함수의 핵심 기능은 이미지의 각 픽셀에 대한 인스턴스 임베딩을 계산하는 것입니다.&#x20;
* 이를 위해, 함수는 참값(truth)으로부터 각 인스턴스에 대한 임베딩을 추출합니다.&#x20;
* 추출한 임베딩과 네트워크의 출력 간의 차이를 계산하여 임베딩 손실을 계산하고, 이를 사용하여 네트워크의 가중치를 업데이트합니다.&#x20;
* 이 함수는 또한 임베딩 손실을 계산하기 위해 평균 제곱 오차(mse)를 계산합니다.&#x20;
* 함수는 또한 경계 상자와 함께 사용할 수 있는 좌표와 클래스 예측을 포함하는 출력 텐서를 생성합니다.



## backward\_iseg\_layer

```c
void backward_iseg_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_iseg\_layer&#x20;

입력:&#x20;

* const layer l
* network net&#x20;

동작:&#x20;

* l.delta와 net.delta를 더한 결과를 net.delta에 저장합니다.&#x20;

설명:&#x20;

* iSeg 레이어의 역전파(backward propagation)를 수행하는 함수입니다.&#x20;
* l.delta와 net.delta는 각각 iSeg 레이어와 연결된 레이어의 delta값과 네트워크 전체의 delta값을 저장하는 배열입니다.&#x20;
* 이 함수는 l.delta와 net.delta를 더한 결과를 net.delta에 저장합니다.&#x20;
* 이 과정은 연결된 레이어의 delta값을 이용하여 이전 레이어의 gradient를 계산하기 위해 필요합니다.



## resize\_iseg\_layer

```c
void resize_iseg_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->c;
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));
}
```

함수 이름: resize\_iseg\_layer&#x20;

입력:

* layer \*l : 레이어 구조체 포인터
* int w : 레이어의 새로운 너비
* int h : 레이어의 새로운 높이

동작:&#x20;

* 입력으로 받은 레이어 포인터를 이용하여 l->w와 l->h를 각각 w와 h로 변경하고, l->c와 w, h를 이용하여 l->outputs과 l->inputs을 다시 계산하여 업데이트합니다.&#x20;
* 그리고 l->output과 l->delta를 레이어의 새로운 크기에 맞게 재할당합니다.

설명:&#x20;

* 이 함수는 인풋 세그멘테이션 레이어를 리사이징할 때 사용됩니다.&#x20;
* 이 함수를 호출하면 레이어의 크기가 변경되며, 레이어의 아웃풋과 델타 배열도 리사이징된 크기에 맞게 재할당됩니다.&#x20;
* 이 함수를 통해 레이어의 크기를 적절히 조절하여 모델을 튜닝할 수 있습니다.



## make\_iseg\_layer

```c
layer make_iseg_layer(int batch, int w, int h, int classes, int ids)
{
    layer l = {0};
    l.type = ISEG;

    l.h = h;
    l.w = w;
    l.c = classes + ids;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.batch = batch;
    l.extra = ids;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*l.c;
    l.inputs = l.outputs;
    l.truths = 90*(l.w*l.h+1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));

    l.counts = calloc(90, sizeof(int));
    l.sums = calloc(90, sizeof(float*));
    if(ids){
        int i;
        for(i = 0; i < 90; ++i){
            l.sums[i] = calloc(ids, sizeof(float));
        }
    }

    l.forward = forward_iseg_layer;
    l.backward = backward_iseg_layer;

    fprintf(stderr, "iseg\n");
    srand(0);

    return l;
}
```

함수 이름: make\_iseg\_layer

입력:

* batch: int 타입, batch size
* w: int 타입, 입력 이미지의 너비 (width)
* h: int 타입, 입력 이미지의 높이 (height)
* classes: int 타입, segmentation 클래스 수
* ids: int 타입, 추가적인 segmentation ID 수

동작:&#x20;

* 입력으로 받은 파라미터를 이용하여, 인스턴스 분할(segmentation) 레이어를 생성하고 초기화한 후 반환한다.

설명:

* layer 구조체 변수 l을 초기화하고, 필요한 값들을 할당한다.
* l.type을 ISEG로 설정하고, l.h, l.w, l.c, l.out\_w, l.out\_h, l.out\_c, l.classes, l.batch, l.extra, l.outputs, l.inputs, l.truths, l.delta, l.output, l.counts, l.sums, l.cost 등의 변수를 설정한다.
* l.sums 배열의 메모리를 할당하고, ids가 0이 아니면, 각각의 원소마다 추가적인 메모리를 할당한다.
* l.forward와 l.backward 함수를 설정하고, "iseg"라는 문자열을 출력한다.
* 초기화된 layer 구조체 l을 반환한다.

