# darknet

```c
#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_regressor(int argc, char **argv);
extern void run_segmenter(int argc, char **argv);
extern void run_isegmenter(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);
extern void run_lsd(int argc, char **argv);
```

## average

```c
void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    network *sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];   
    load_weights(sum, weightfile);

    int i, j;
    int n = argc - 5;
    for(i = 0; i < n; ++i){
        weightfile = argv[i+5];   
        load_weights(net, weightfile);
        for(j = 0; j < net->n; ++j){
            layer l = net->layers[j];
            layer out = sum->layers[j];
            if(l.type == CONVOLUTIONAL){
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if(l.batch_normalize){
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if(l.type == CONNECTED){
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n+1;
    for(j = 0; j < net->n; ++j){
        layer l = sum->layers[j];
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1./n, l.biases, 1);
            scal_cpu(num, 1./n, l.weights, 1);
                if(l.batch_normalize){
                    scal_cpu(l.n, 1./n, l.scales, 1);
                    scal_cpu(l.n, 1./n, l.rolling_mean, 1);
                    scal_cpu(l.n, 1./n, l.rolling_variance, 1);
                }
        }
        if(l.type == CONNECTED){
            scal_cpu(l.outputs, 1./n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}
```

함수 이름: average&#x20;

입력:

* argc (int 타입): 프로그램 실행 시 전달되는 인자의 개수
* argv (char\* 타입): 프로그램 실행 시 전달되는 인자들의 배열
  * argv\[2] (char\* 타입): configuration 파일의 경로
  * argv\[3] (char\* 타입): 결과 파일의 경로
  * argv\[4] (char\* 타입): 가중치(weight) 파일의 경로
  * argv\[5]부터 (char\* 타입): 가중치(weight) 파일들의 경로들

동작:

* configuration 파일(cfgfile)을 파싱하여 network 구조체를 만든다.
* 결과를 저장할 network 구조체(sum)도 만든다. (cfgfile로부터 파싱한 것과 같은 구조)
* argv\[4]를 가중치 파일로 읽어들여 sum에 저장한다.
* argv\[5]부터 끝까지 반복하며 가중치 파일을 읽어들인 후, sum과 더한다.
* sum을 n+1로 나눠 평균값을 구한다. (n은 가중치 파일의 개수)
* 결과를 outfile에 저장한다.

설명:&#x20;

* 이 함수는 딥러닝 모델의 가중치(weight) 파일들을 입력으로 받아, 이들의 평균값을 구한 후, 하나의 가중치 파일로 저장하는 기능을 수행한다.&#x20;
* 이 함수는 Darknet 딥러닝 프레임워크의 일부분으로, C 언어로 작성되었다.&#x20;
* 이 함수는 CLI(Command-Line Interface) 환경에서 실행되며, argc와 argv를 통해 프로그램 실행 시 전달되는 인자들을 받는다.
* 이 함수는 입력으로 받은 가중치 파일들의 평균을 구하므로, 모델의 성능을 평가하는 데에 유용하게 사용될 수 있다.



## numops

```c
long numops(network *net)
{
    int i;
    long ops = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            ops += 2l * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w;
        } else if(l.type == CONNECTED){
            ops += 2l * l.inputs * l.outputs;
        } else if (l.type == RNN){
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        } else if (l.type == GRU){
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        } else if (l.type == LSTM){
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    return ops;
}
```

함수 이름: numops

입력:&#x20;

* network 구조체 포인터: 네트워크 구조를 저장하는 구조체에 대한 포인터입니다.

동작:&#x20;

* 네트워크에 있는 모든 레이어의 총 연산 횟수를 계산하여 반환합니다.

설명:

* network 구조체는 Darknet 라이브러리에서 사용하는 네트워크 구조를 저장합니다.
* 레이어의 타입에 따라 연산 횟수가 다르기 때문에, CONVOLUTIONAL, CONNECTED, RNN, GRU, LSTM 레이어 타입에 따라 계산을 수행합니다.
* CONVOLUTIONAL 레이어의 경우, 컨볼루션 연산과 덧셈 연산이 이루어지므로, 2 \* 필터 총 원소 수 \* 출력 크기 만큼의 연산 횟수가 더해집니다.
* CONNECTED 레이어의 경우, 행렬 곱셈과 덧셈 연산이 이루어지므로, 2 \* 입력 크기 \* 출력 크기 만큼의 연산 횟수가 더해집니다.
* RNN, GRU, LSTM 레이어의 경우, 입력 게이트, 출력 게이트, 삭제 게이트와 같은 각 게이트마다 연산 횟수를 계산합니다. 이들 레이어의 총 연산 횟수는 각각 3개, 6개, 8개 입니다.
* 최종적으로 모든 레이어의 연산 횟수를 더한 값을 반환합니다.



## speed

```c
void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network *net = parse_network_cfg(cfgfile);
    set_batch_network(net, 1);
    int i;
    double time=what_time_is_it_now();
    image im = make_image(net->w, net->h, net->c*net->batch);
    for(i = 0; i < tics; ++i){
        network_predict(net, im.data);
    }
    double t = what_time_is_it_now() - time;
    long ops = numops(net);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
    printf("FLOPS: %.2f Bn\n", (float)ops/1000000000.*tics/t);
    printf("Speed: %f sec/eval\n", t/tics);
    printf("Speed: %f Hz\n", tics/t);
}
```

함수 이름: speed 입력:

* cfgfile: 설정 파일 경로
* tics: 반복 횟수. 기본값은 1000.

동작:&#x20;

\-m주어진 설정 파일(cfgfile)을 파싱하여 네트워크를 생성하고, 입력 이미지를 생성한 후 반복 횟수(tics)만큼 네트워크에 입력을 주어 예측을 수행합니다.&#x20;

그리고 수행 시간과 부동 소수점 연산 횟수(Floating Point Operations)를 계산하여 출력합니다.

설명:&#x20;

* 이 함수는 입력 이미지를 생성한 후 반복 횟수만큼 네트워크를 실행하여 처리 속도와 연산량(FLOPS)을 출력합니다.&#x20;
* 먼저, 설정 파일(cfgfile)을 파싱하여 네트워크를 생성한 다음, 입력 이미지를 만듭니다.&#x20;
* 그리고, 반복 횟수(tics)만큼 네트워크에 입력을 주어 예측을 수행합니다. 이 때, 처리에 걸린 시간과 부동 소수점 연산 횟수를 계산합니다.&#x20;
* 마지막으로, 이 정보들을 출력합니다. 출력 정보에는 총 수행 시간, 부동 소수점 연산 횟수, FLOPS, 처리 속도(sec/eval), 처리 속도(Hz)가 포함됩니다.



## operations

```c
void operations(char *cfgfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    long ops = numops(net);
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
}
```

함수 이름: operations&#x20;

입력:&#x20;

* cfgfile (문자열 포인터): 네트워크 구성 파일 경로&#x20;

동작:&#x20;

* 주어진 네트워크 구성 파일을 파싱하여 해당 네트워크가 수행하는 총 부동 소수점 연산 횟수를 계산하고 출력합니다.&#x20;

설명:

* gpu\_index를 -1로 초기화합니다.
* 주어진 cfgfile을 파싱하여 네트워크를 생성합니다.
* numops 함수를 사용하여 해당 네트워크가 수행하는 총 부동 소수점 연산 횟수를 계산합니다.
* 계산된 부동 소수점 연산 횟수를 출력합니다. 출력할 때, 단위를 10억(Billion)으로 나누어 GFLOPs(Giga Floating Point Operations Per Second) 단위로 출력합니다.



## oneoff

```c
void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    int oldn = net->layers[net->n - 2].n;
    int c = net->layers[net->n - 2].c;
    scal_cpu(oldn*c, .1, net->layers[net->n - 2].weights, 1);
    scal_cpu(oldn, 0, net->layers[net->n - 2].biases, 1);
    net->layers[net->n - 2].n = 11921;
    net->layers[net->n - 2].biases += 5;
    net->layers[net->n - 2].weights += 5*c;
    if(weightfile){
        load_weights(net, weightfile);
    }
    net->layers[net->n - 2].biases -= 5;
    net->layers[net->n - 2].weights -= 5*c;
    net->layers[net->n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net->layers[net->n - 2];
    copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
    copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
    *net->seen = 0;
    save_weights(net, outfile);
}
```

함수 이름: oneoff&#x20;

입력:

* cfgfile (char\*): 네트워크 구성 파일 경로
* weightfile (char\*): 가중치 파일 경로 (옵션)
* outfile (char\*): 저장할 가중치 파일 경로

동작:&#x20;

* 주어진 네트워크 구성 파일(cfgfile)을 파싱하여 네트워크(net)를 생성하고, 이를 이용하여 가중치를 초기화한 후 일회성(one-off)으로 훈련시킨 결과를 저장한다.&#x20;
* 이 과정에서 가중치 파일(weightfile)을 제공하면 해당 파일에서 가중치를 로드하여 초기화하고, outfile에서 지정한 경로에 최종 가중치 파일을 저장한다.

설명:

* 주어진 네트워크 구성 파일(cfgfile)을 파싱하여 네트워크(net)를 생성한다.
* oldn은 net의 뒤에서 두 번째 레이어의 뉴런 수이며, c는 채널 수이다. 이 레이어의 가중치와 편향을 0.1과 0으로 초기화한다.
* net의 뒤에서 두 번째 레이어의 뉴런 수를 11921로 변경한다.
* net의 뒤에서 두 번째 레이어의 가중치와 편향에 대한 포인터를 각각 5와 5\*c만큼 옮긴다.
* weightfile이 주어진 경우 해당 파일에서 가중치를 로드하여 초기화한다.
* 가중치와 편향에 대한 포인터를 원래 위치로 되돌린다.
* net의 뒤에서 두 번째 레이어의 뉴런 수를 이전 값으로 복원한다.
* l은 net의 뒤에서 두 번째 레이어이며, 이 레이어의 1/3 지점과 2/3 지점에서의 편향과 가중치 값을 서로 복사한다.
* net->seen 값을 0으로 초기화한다.
* 최종 가중치 파일을 outfile에서 지정한 경로에 저장한다.



## oneoff2

```c
void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(net, weightfile, 0, net->n);
        load_weights_upto(net, weightfile, l, net->n);
    }
    *net->seen = 0;
    save_weights_upto(net, outfile, net->n);
}
```

함수 이름: oneoff2&#x20;

입력:&#x20;

* char \*cfgfile (구성 파일 경로)
* char \*weightfile (가중치 파일 경로)
* char \*outfile (저장할 가중치 파일 경로)
* int l (가중치를 로드할 레이어 인덱스)&#x20;

동작:&#x20;

* 주어진 구성 파일(cfgfile)을 사용하여 네트워크를 만들고, 주어진 가중치 파일(weightfile)에서 가중치를 로드합니다.&#x20;
* 그리고 로드한 가중치 중 l 번째 레이어까지의 가중치를 사용하여 네트워크를 업데이트합니다.&#x20;
* 마지막으로 변경된 가중치를 주어진 파일(outfile)에 저장합니다.&#x20;

설명:

* parse\_network\_cfg 함수는 주어진 구성 파일(cfgfile)에서 네트워크를 만듭니다.
* load\_weights\_upto 함수는 주어진 가중치 파일(weightfile)에서 가중치를 로드합니다. 마지막 인자로 지정한 레이어까지의 가중치를 로드합니다.
* save\_weights\_upto 함수는 변경된 가중치를 주어진 파일(outfile)에 저장합니다. 마지막 인자로 지정한 레이어까지의 가중치를 저장합니다.



## partial

```c
void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    save_weights_upto(net, outfile, max);
}
```

함수 이름: partial

입력:

* char \*cfgfile: 네트워크 구성 파일 경로
* char \*weightfile: 가중치 파일 경로
* char \*outfile: 저장될 가중치 파일 경로
* int max: 저장할 레이어의 최대 인덱스

동작:

* cfgfile과 weightfile을 사용하여 네트워크를 로드한다.
* max 레이어까지의 가중치만 저장하여 outfile에 저장한다.

설명:&#x20;

* 주어진 cfgfile과 weightfile을 사용하여 네트워크를 로드한 후, max 레이어까지의 가중치만 저장하여 outfile에 저장하는 함수이다.&#x20;
* 이때, gpu\_index는 -1로 설정되며, load\_network 함수를 사용하여 네트워크를 로드하고, save\_weights\_upto 함수를 사용하여 일부 레이어의 가중치만 저장한다.



## print\_weights

```c
void print_weights(char *cfgfile, char *weightfile, int n)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    layer l = net->layers[n];
    int i, j;
    //printf("[");
    for(i = 0; i < l.n; ++i){
        //printf("[");
        for(j = 0; j < l.size*l.size*l.c; ++j){
            //if(j > 0) printf(",");
            printf("%g ", l.weights[i*l.size*l.size*l.c + j]);
        }
        printf("\n");
        //printf("]%s\n", (i == l.n-1)?"":",");
    }
    //printf("]");
}
```

함수 이름: print\_weights

입력:&#x20;

* char \*cfgfile (네트워크 구성 파일 경로)
* char \*weightfile (가중치 파일 경로)
* int n (출력할 레이어 번호)

동작:&#x20;

* 주어진 네트워크 구성 파일과 가중치 파일을 로드하고, 지정된 레이어의 가중치를 출력하는 함수입니다.&#x20;
* 출력 형식은 해당 레이어의 크기와 채널 수에 따라 2차원 배열의 형태로 출력합니다.

설명:

* cfgfile: 네트워크 구성 파일 경로를 지정하는 문자열 포인터입니다.
* weightfile: 가중치 파일 경로를 지정하는 문자열 포인터입니다.
* n: 출력할 레이어의 번호를 지정하는 정수값입니다.
* net: 주어진 cfgfile과 weightfile을 이용해 로드한 네트워크를 가리키는 포인터입니다.
* l: 지정된 레이어를 가리키는 layer 구조체입니다.
* i, j: 반복문에서 사용되는 정수값으로, l의 크기에 따라 가중치 배열을 출력합니다.
* 출력 형식: 지정된 레이어의 가중치를 해당 레이어의 크기와 채널 수에 따라 2차원 배열 형태로 출력합니다.



## rescale\_net

```c
void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}
```

함수 이름: rescale\_net

입력:

* cfgfile: char 포인터 형식. YOLO 모델의 구성 파일 경로.
* weightfile: char 포인터 형식. YOLO 모델의 가중치 파일 경로.
* outfile: char 포인터 형식. 변경된 가중치를 저장할 파일 경로.

동작:&#x20;

* 주어진 YOLO 모델의 구성 파일과 가중치 파일을 불러와서, 가장 처음 발견된 컨볼루션 레이어의 가중치를 2배로 확장하고 -0.5를 뺀다.&#x20;
* 그리고 변경된 가중치를 저장한다.

설명:&#x20;

* 이 함수는 YOLO 모델의 구성 파일과 가중치 파일을 불러온 후, 가장 처음 발견된 컨볼루션 레이어의 가중치를 rescale하는 작업을 수행한다.&#x20;
* rescale\_weights() 함수를 사용하여 가중치를 2배로 확장하고 -0.5를 뺀다.&#x20;
* 변경된 가중치는 save\_weights() 함수를 사용하여 지정된 파일 경로에 저장된다.



## rgbgr\_net

```c
void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}
```

함수 이름: rgbgr\_net

입력:

* cfgfile (char\*): 모델 구성 파일 경로
* weightfile (char\*): 모델 가중치 파일 경로
* outfile (char\*): 변경된 가중치를 저장할 파일 경로

동작:&#x20;

* 입력으로 받은 모델 구성 파일과 가중치 파일을 로드하여, CONVOLUTIONAL 레이어에서 사용되는 가중치들을 RGB 이미지 채널에서 BGR 이미지 채널로 변경합니다.
* 변경된 가중치를 저장할 파일 경로를 입력받아 저장합니다.

설명:&#x20;

* RGB 이미지와 BGR 이미지는 색상 채널의 순서가 서로 다릅니다.&#x20;
* 따라서 RGB 이미지 채널로 학습된 모델을 BGR 이미지 채널로 변경하여 사용할 때는 가중치를 변경해주어야 합니다.&#x20;
* 이 함수는 입력으로 받은 모델에서 CONVOLUTIONAL 레이어에서 사용되는 가중치들을 RGB에서 BGR로 변경하고, 변경된 가중치를 지정된 파일 경로에 저장합니다.



## reset\_normalize\_net

```c
void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
    }
    save_weights(net, outfile);
}
```

함수 이름: reset\_normalize\_net

입력:

* cfgfile: char 포인터. 네트워크 구성 파일 경로.
* weightfile: char 포인터. 가중치 파일 경로.
* outfile: char 포인터. 출력 파일 경로.

동작:

* 입력된 cfgfile과 weightfile을 이용하여 네트워크를 로드한다.
* 모든 레이어를 반복하면서, CONVOLUTIONAL 레이어이고 batch\_normalize가 설정된 경우, denormalize\_convolutional\_layer() 함수를 호출하여 가중치와 스케일, 평균, 분산 값을 원래 값으로 복원한다.
* CONNECTED 레이어이고 batch\_normalize가 설정된 경우, denormalize\_connected\_layer() 함수를 호출하여 가중치와 스케일, 평균, 분산 값을 원래 값으로 복원한다.
* GRU 레이어이고 batch\_normalize가 설정된 경우, 입력 게이트, 잊어버리는 게이트, 상태 게이트의 가중치와 스케일, 평균, 분산 값을 복원한다.
* save\_weights() 함수를 호출하여 outfile 경로에 가중치를 저장한다.

설명:&#x20;

* 이 함수는 YOLOv3 딥러닝 프레임워크에서 사용되는 함수 중 하나로, 네트워크의 가중치를 denormalize하는 함수이다.&#x20;
* 각 레이어의 가중치와 스케일, 평균, 분산 값을 denormalize\_convolutional\_layer() 함수, denormalize\_connected\_layer() 함수, denormalize\_gru\_layer() 함수를 이용하여 원래 값으로 복원하고, 이를 저장하는 기능을 수행한다.&#x20;
* 이 함수는 네트워크를 fine-tuning할 때, batch\_normalize 레이어를 사용하지 않는 경우 등에 활용된다.



## normalize\_layer

```c
layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize=1;
    l.scales = calloc(n, sizeof(float));
    for(j = 0; j < n; ++j){
        l.scales[j] = 1;
    }
    l.rolling_mean = calloc(n, sizeof(float));
    l.rolling_variance = calloc(n, sizeof(float));
    return l;
}
```

함수 이름: normalize\_layer

입력:&#x20;

* layer l (레이어 구조체)
* int n (레이어의 출력 채널 개수)

동작:&#x20;

* 입력된 레이어 구조체에 대해 배치 정규화(batch normalization)를 수행하는 함수이다.&#x20;
* 배치 정규화를 위해 스케일(scales), 이동(rolling mean), 분산(rolling variance)을 저장할 메모리를 할당하고, 스케일을 1로 초기화한다.

설명:

* 배치 정규화는 인공 신경망에서 입력 데이터나 출력 데이터의 분포를 정규 분포로 만들어 학습의 안정성과 수렴 속도를 향상시키는 방법이다.
* 이 함수는 단일 레이어에 대한 배치 정규화를 수행한다.
* 스케일은 출력 채널의 개수(n)만큼의 메모리를 동적으로 할당하고 1로 초기화한다.
* 이동과 분산은 스케일과 마찬가지로 출력 채널의 개수(n)만큼의 메모리를 동적으로 할당하고 0으로 초기화한다. 이후 학습 과정에서 이동과 분산 값이 업데이트된다.
* l.batch\_normalize 변수를 1로 설정하여 해당 레이어가 배치 정규화를 사용하도록 설정한다.
* 입력으로 전달된 레이어 구조체 l을 반환한다.



## normalize\_net

```c
void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL && !l.batch_normalize){
            net->layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net->layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net->layers[i].batch_normalize=1;
        }
    }
    save_weights(net, outfile);
}
```

함수 이름: normalize\_net

입력:&#x20;

* cfgfile (char \*): 네트워크 구성 파일 경로&#x20;
* weightfile (char \*): 네트워크 가중치 파일 경로&#x20;
* outfile (char \*): 저장될 가중치 파일 경로

동작:&#x20;

* 네트워크를 로드하고, 각 층에서 배치 정규화가 적용되지 않은 경우 배치 정규화 층을 추가하여 네트워크를 정규화하고, 모든 가중치를 저장한다.

설명:&#x20;

* normalize\_layer() 함수는 입력 층을 정규화하는 함수이다. 이 함수는 l의 배치 정규화 플래그를 설정하고, 스케일을 모두 1로 초기화한 다음, rolling\_mean과 rolling\_variance 배열을 할당한다.&#x20;
* 이 함수는 변경된 l을 반환한다.&#x20;
* 이 normalize\_net() 함수는 네트워크를 로드하고 각 층을 검사하여 배치 정규화가 적용되지 않은 경우 normalize\_layer() 함수를 사용하여 층을 정규화한다.&#x20;
* GRU 층에 대해서는, 각각의 연산을 위해 입력 층과 상태 층을 정규화한다. 마지막으로, save\_weights() 함수를 사용하여 정규화된 네트워크의 가중치를 저장한다.



## statistics\_net

```c
void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        printf("\n");
    }
}
```

함수 이름: statistics\_net

입력:

* cfgfile: 네트워크 구성 파일 경로를 나타내는 문자열 포인터
* weightfile: 학습된 가중치 파일 경로를 나타내는 문자열 포인터

동작:

* 지정된 네트워크 구성 파일(cfgfile)과 가중치 파일(weightfile)을 로드하여 네트워크를 생성한다.
* 생성된 네트워크의 각 레이어를 반복하면서,
  * 현재 레이어가 CONNECTED 레이어이고 batch\_normalize 속성이 True인 경우, statistics\_connected\_layer() 함수를 호출하여 연결된 레이어의 통계 정보를 출력한다.
  * 현재 레이어가 GRU 레이어이고 batch\_normalize 속성이 True인 경우, 연결된 레이어의 각각의 input\_z\_layer, input\_r\_layer, input\_h\_layer, state\_z\_layer, state\_r\_layer, state\_h\_layer에 대해 statistics\_connected\_layer() 함수를 호출하여 통계 정보를 출력한다.
* 모든 레이어의 통계 정보 출력을 마친 후, 함수 실행 종료한다.

설명:

* 지정된 네트워크(cfgfile와 weightfile)를 로드하여 생성된 네트워크에서 각 레이어의 통계 정보(평균, 분산, 표준편차 등)를 계산하고 출력하는 함수이다.
* 출력되는 통계 정보는 연결된 레이어의 입력값과 가중치의 분포를 확인하는 데 유용하다.



## denormalize\_net

```c
void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if ((l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL) && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net->layers[i].batch_normalize=0;
        }
    }
    save_weights(net, outfile);
}
```

함수 이름: denormalize\_net

입력:

* char \*cfgfile: 모델 구성 파일 경로
* char \*weightfile: 학습된 가중치 파일 경로
* char \*outfile: 가중치를 저장할 파일 경로

동작:&#x20;

* 입력으로 주어진 모델 구성 파일과 학습된 가중치 파일을 이용하여 모델을 로드하고, 각 레이어의 batch normalization을 되돌린 후, 가중치를 저장하는 기능을 수행한다.

설명:

* 주어진 cfgfile과 weightfile을 이용하여 모델을 로드한다.
* 모든 레이어에 대해서, l.type이 CONVOLUTIONAL 또는 DECONVOLUTIONAL인 경우에는 batch normalization을 되돌린다. 이를 위해 denormalize\_convolutional\_layer 함수를 호출하고, l.batch\_normalize를 0으로 설정한다.
* l.type이 CONNECTED인 경우에는 denormalize\_connected\_layer 함수를 호출하고, l.batch\_normalize를 0으로 설정한다.
* l.type이 GRU인 경우에는 input\_z\_layer, input\_r\_layer, input\_h\_layer, state\_z\_layer, state\_r\_layer, state\_h\_layer 각각에 대해서 denormalize\_connected\_layer 함수를 호출하고, 각 레이어의 batch\_normalize를 0으로 설정한다.
* 처리가 끝난 모델 가중치를 outfile에 저장한다.



## mkimg

```c
void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix)
{
    network *net = load_network(cfgfile, weightfile, 0);
    image *ims = get_weights(net->layers[0]);
    int n = net->layers[0].n;
    int z;
    for(z = 0; z < num; ++z){
        image im = make_image(h, w, 3);
        fill_image(im, .5);
        int i;
        for(i = 0; i < 100; ++i){
            image r = copy_image(ims[rand()%n]);
            rotate_image_cw(r, rand()%4);
            random_distort_image(r, 1, 1.5, 1.5);
            int dx = rand()%(w-r.w);
            int dy = rand()%(h-r.h);
            ghost_image(r, im, dx, dy);
            free_image(r);
        }
        char buff[256];
        sprintf(buff, "%s/gen_%d", prefix, z);
        save_image(im, buff);
        free_image(im);
    }
}
```

함수 이름: mkimg&#x20;

입력:

* cfgfile: YOLO 모델 구성 파일 경로
* weightfile: 학습된 YOLO 모델 가중치 파일 경로
* h: 생성할 이미지 높이
* w: 생성할 이미지 너비
* num: 생성할 이미지 개수
* prefix: 생성된 이미지 파일명에 추가될 prefix 문자열

동작:&#x20;

* 주어진 YOLO 모델의 첫번째 레이어의 가중치로부터 이미지를 생성하는 함수입니다.&#x20;
* 생성할 이미지 개수(num)만큼 반복하면서 각 이미지를 랜덤한 배치와 왜곡을 적용하여 생성합니다.&#x20;
* 생성된 이미지는 prefix와 이미지 번호를 조합하여 파일명을 만들고 저장합니다.

설명:&#x20;

* 이 함수는 YOLO 모델의 가중치로부터 이미지를 생성하는 함수입니다. 먼저 입력받은 cfgfile과 weightfile을 이용하여 YOLO 모델을 로드합니다.&#x20;
* 그 다음 모델의 첫번째 레이어의 가중치를 가져와 ims에 저장합니다.&#x20;
* n은 ims의 원소 개수를 나타냅니다.&#x20;
* 그 후, num만큼 이미지를 생성하면서 각 이미지를 랜덤한 배치와 왜곡을 적용합니다.&#x20;
* 생성된 이미지는 prefix와 이미지 번호를 조합하여 파일명을 만들고 저장합니다.&#x20;
* 마지막으로 할당한 메모리를 해제합니다.

## visualize

```c
void visualize(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    visualize_network(net);
}
```

함수 이름: visualize&#x20;

입력:

* cfgfile: YOLO 모델 구성 파일 경로
* weightfile: 학습된 YOLO 모델의 가중치 파일 경로

동작:&#x20;

* 입력된 YOLO 모델을 불러와서 시각화합니다.&#x20;
* 시각화된 모델은 graph.png 파일로 저장됩니다.

설명:&#x20;

* 입력된 cfgfile과 weightfile 경로에서 YOLO 모델을 불러와 시각화합니다.&#x20;
* 시각화된 모델은 레이어, 필터, 차원 등 다양한 정보를 그래프로 나타내어 모델 구조를 쉽게 이해할 수 있도록 돕습니다.



## main

```c
int main(int argc, char **argv)
{
    //test_resize("data/bad.jpg");
    //test_box();
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "yolo")){
        run_yolo(argc, argv);
    } else if (0 == strcmp(argv[1], "super")){
        run_super(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "detect")){
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        char *filename = (argc > 4) ? argv[4]: 0;
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    } else if (0 == strcmp(argv[1], "cifar")){
        run_cifar(argc, argv);
    } else if (0 == strcmp(argv[1], "go")){
        run_go(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "isegmenter")){
        run_isegmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "segmenter")){
        run_segmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "art")){
        run_art(argc, argv);
    } else if (0 == strcmp(argv[1], "tag")){
        run_tag(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "reset")){
        reset_normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "denormalize")){
        denormalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "statistics")){
        statistics_net(argv[2], argv[3]);
    } else if (0 == strcmp(argv[1], "normalize")){
        normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "ops")){
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    } else if (0 == strcmp(argv[1], "oneoff")){
        oneoff(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "oneoff2")){
        oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "print")){
        print_weights(argv[2], argv[3], atoi(argv[4]));
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "mkimg")){
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else if (0 == strcmp(argv[1], "imtest")){
        test_resize(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}
```

함수 이름: main&#x20;

입력:&#x20;

* int argc, char \*\*argv (인자의 개수와 값)&#x20;

동작:&#x20;

* argv\[1]의 값에 따라 다른 함수를 호출하는 분기문&#x20;

설명:

* 입력으로 받은 인자의 개수와 값들을 나타내는 int argc, char \*\*argv를 받는다.
* argv\[1]에 해당하는 값으로 분기하여 다른 함수를 호출한다.&#x20;
* 아래는 argv\[1]에 따른 호출할 함수들이다.
  * "average": average(argc, argv) 함수 호출
  * "yolo": run\_yolo(argc, argv) 함수 호출
  * "super": run\_super(argc, argv) 함수 호출
  * "lsd": run\_lsd(argc, argv) 함수 호출
  * "detector": test\_detector 함수 호출
  * "detect": test\_detector 함수 호출
  * "cifar": run\_cifar(argc, argv) 함수 호출
  * "go": run\_go(argc, argv) 함수 호출
  * "rnn": run\_char\_rnn(argc, argv) 함수 호출
  * "coco": run\_coco(argc, argv) 함수 호출
  * "classify": predict\_classifier 함수 호출
  * "classifier": run\_classifier(argc, argv) 함수 호출
  * "regressor": run\_regressor(argc, argv) 함수 호출
  * "isegmenter": run\_isegmenter(argc, argv) 함수 호출
  * "segmenter": run\_segmenter(argc, argv) 함수 호출
  * "art": run\_art(argc, argv) 함수 호출
  * "tag": run\_tag(argc, argv) 함수 호출
  * "3d": composite\_3d 함수 호출
  * "test": test\_resize 함수 호출
  * "nightmare": run\_nightmare(argc, argv) 함수 호출
  * "rgbgr": rgbgr\_net 함수 호출
  * "reset": reset\_normalize\_net 함수 호출
  * "denormalize": denormalize\_net 함수 호출
  * "statistics": statistics\_net 함수 호출
  * "normalize": normalize\_net 함수 호출
  * "rescale": rescale\_net 함수 호출
  * "ops": operations 함수 호출
  * "speed": speed 함수 호출
  * "oneoff": oneoff 함수 호출
  * "oneoff2": oneoff2 함수 호출
  * "print": print\_weights 함수 호출
  * "partial": partial 함수 호출
  * "visualize": visualize 함수 호출
  * "mkimg": mkimg 함수 호출
  * "imtest": test\_resize 함수 호출
  * 입력이 위에서 언급한 경우가 아닐 경우, 에러 메시지 출력
  * 함수의 반환값은 0이다.

