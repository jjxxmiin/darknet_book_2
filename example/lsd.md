# lsd

```c
#include <math.h>
#include "darknet.h"
```

## slerp

```c
void slerp(float *start, float *end, float s, int n, float *out)
{
    float omega = acos(dot_cpu(n, start, 1, end, 1));
    float so = sin(omega);
    fill_cpu(n, 0, out, 1);
    axpy_cpu(n, sin((1-s)*omega)/so, start, 1, out, 1);
    axpy_cpu(n, sin(s*omega)/so, end, 1, out, 1);

    float mag = mag_array(out, n);
    scale_array(out, n, 1./mag);
}
```

함수 이름: slerp&#x20;

입력:

* start: float 포인터. 시작 벡터를 가리키는 포인터입니다.
* end: float 포인터. 끝 벡터를 가리키는 포인터입니다.
* s: float 값. 보간 매개변수입니다. \[0, 1] 범위의 값으로 설정됩니다.
* n: 정수. 벡터의 길이 또는 요소 수입니다.
* out: float 포인터. 결과 벡터를 저장할 포인터입니다.

동작:&#x20;

* 이 함수는 두 벡터 사이를 구면 보간(spherical interpolation)하여 새로운 벡터를 생성합니다.&#x20;
* 시작 벡터(start)와 끝 벡터(end) 사이를 매개변수 s를 사용하여 선형적으로 보간하고, 생성된 보간 벡터를 out에 저장합니다.

설명:&#x20;

* 이 함수는 주어진 시작 벡터(start)와 끝 벡터(end) 사이를 구면 보간하여 새로운 벡터를 생성합니다.
* 먼저, 두 벡터 사이의 각도(omega)를 계산하기 위해 dot\_cpu 함수를 사용하여 내적 값을 얻습니다. 그리고 이 각도를 사용하여 사인(sin) 값을 계산합니다.
* 그 다음, 결과 벡터(out)를 0으로 초기화합니다.
* axpy\_cpu 함수를 사용하여 보간된 벡터를 계산하여 결과 벡터(out)에 더합니다.&#x20;
* 여기서는 시작 벡터(start)와 끝 벡터(end)를 보간 매개변수(s)와 사인 값에 따라 선형적으로 보간하여 계산합니다.
* 마지막으로, 결과 벡터(out)의 크기를 계산하기 위해 mag\_array 함수를 사용합니다.&#x20;
* mag\_array 함수는 벡터의 크기 또는 배열의 크기를 반환합니다.&#x20;
* 그리고 scale\_array 함수를 사용하여 결과 벡터(out)를 이 크기로 나누어 정규화합니다.



## random\_unit\_vector\_image

```c
image random_unit_vector_image(int w, int h, int c)
{
    image im = make_image(w, h, c);
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        im.data[i] = rand_normal();
    }
    float mag = mag_array(im.data, im.w*im.h*im.c);
    scale_array(im.data, im.w*im.h*im.c, 1./mag);
    return im;
}
```

함수 이름: random\_unit\_vector\_image&#x20;

입력:

* w: 정수. 생성할 이미지의 너비입니다.
* h: 정수. 생성할 이미지의 높이입니다.
* c: 정수. 생성할 이미지의 채널 수입니다.

동작:&#x20;

* 이 함수는 주어진 너비(w), 높이(h), 채널(c)을 가지는 이미지를 생성합니다.&#x20;
* 생성된 이미지는 각 픽셀의 값이 무작위로 선택된 단위 벡터(unit vector)로 초기화됩니다.

설명:&#x20;

* 이 함수는 주어진 너비(w), 높이(h), 채널(c)을 가지는 이미지(im)를 생성합니다.&#x20;
* make\_image 함수를 사용하여 너비, 높이, 채널에 맞는 이미지 객체를 생성합니다.
* 그리고 이미지의 각 픽셀을 루프를 통해 순회하면서 rand\_normal 함수를 사용하여 무작위 값으로 초기화합니다.&#x20;
* 이는 각 픽셀의 값을 무작위로 선택한 값으로 설정하는 것을 의미합니다.
* 이어서 mag\_array 함수를 사용하여 이미지 데이터의 크기를 계산합니다.&#x20;
* mag\_array 함수는 이미지 데이터의 크기 또는 벡터의 크기를 반환합니다.&#x20;
* 그리고 scale\_array 함수를 사용하여 이미지 데이터를 이 크기로 나누어 단위 벡터(unit vector)로 정규화합니다.&#x20;
* scale\_array 함수는 배열을 주어진 스케일로 나누는 기능을 수행합니다.
* 마지막으로, 생성된 이미지(im)를 반환합니다.



## inter\_dcgan

```c
void inter_dcgan(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    int i, imlayer = 0;

    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = i;
            printf("%d\n", i);
            break;
        }
    }
    image start = random_unit_vector_image(net->w, net->h, net->c);
    image end = random_unit_vector_image(net->w, net->h, net->c);
        image im = make_image(net->w, net->h, net->c);
        image orig = copy_image(start);

    int c = 0;
    int count = 0;
    int max_count = 15;
    while(1){
        ++c;

        if(count == max_count){
            count = 0;
            free_image(start);
            start = end;
            end = random_unit_vector_image(net->w, net->h, net->c);
            if(c > 300){
                end = orig;
            }
            if(c>300 + max_count) return;
        }
        ++count;

        slerp(start.data, end.data, (float)count / max_count, im.w*im.h*im.c, im.data);

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);
        //yuv_to_rgb(out);
        normalize_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        //char buff[256];
        sprintf(buff, "out%05d", c);
        save_image(out, "out");
        save_image(out, buff);
        show_image(out, "out", 0);
    }
}
```

함수 이름: inter\_dcgan&#x20;

입력:

* cfgfile: 문자열 포인터. 네트워크의 구성 설정 파일 경로입니다.
* weightfile: 문자열 포인터. 네트워크의 가중치 파일 경로입니다.

동작:&#x20;

* 이 함수는 주어진 구성 설정 파일(cfgfile)과 가중치 파일(weightfile)을 사용하여 DCGAN(Deep Convolutional Generative Adversarial Network)을 보간(interpolation)합니다.&#x20;
* DCGAN은 이미지 생성을 위한 생성자(generator)와 이미지 식별을 위한 판별자(discriminator)를 포함하는 딥러닝 모델입니다.&#x20;
* 이 함수는 두 개의 무작위 단위 벡터로부터 시작과 끝 이미지를 생성하고, 이 두 이미지 사이를 보간하여 중간 이미지를 생성합니다.

설명:&#x20;

* 이 함수는 구성 설정 파일(cfgfile)과 가중치 파일(weightfile)을 사용하여 네트워크 객체(net)를 로드합니다.&#x20;
* 그리고 set\_batch\_network 함수를 사용하여 네트워크의 배치 크기를 1로 설정합니다. srand 함수를 사용하여 난수 시드를 설정합니다.

이 함수는 루프를 실행하면서 다음 작업을 반복합니다:

1. 네트워크의 레이어를 순회하면서 출력 채널 수(out\_c)가 3인 레이어를 찾습니다. 이 레이어를 이미지 레이어(imlayer)로 설정합니다.
2. random\_unit\_vector\_image 함수를 사용하여 무작위 단위 벡터를 가지는 시작 이미지(start)와 끝 이미지(end)를 생성합니다.
3. 네트워크의 가로(net->w), 세로(net->h), 채널(net->c)을 크기로 가지는 이미지(im)와 복사된 이미지(orig)를 생성합니다.
4. 카운터 변수(count)가 최대 카운트(max\_count)에 도달할 때마다 이미지를 업데이트합니다. 시작 이미지(start)를 해제하고, 시작 이미지(start)에 끝 이미지(end)를 할당합니다. 만약 c가 300을 초과하면 끝 이미지(end)를 원래 이미지(orig)로 설정합니다.
5. slerp 함수를 사용하여 시작 이미지(start)와 끝 이미지(end) 사이를 보간하여 중간 이미지(im)를 생성합니다.
6. 생성된 이미지(im)를 네트워크를 통해 예측합니다.
7. get\_network\_image\_layer 함수를 사용하여 예측 결과 중 원하는 레이어(imlayer)의 이미지(out)를 가져옵니다.
8. normalize\_image 함수를 사용하여 이미지(out)를 정규화합니다.
9. 생성된 이미지(out)를 출력하고 "out"이라는 이름으로 저장합니다. 또한 순차적인 번호를 가진 파일 이름(buff)으로 이미지를 저장합니다.



## test\_dcgan

```c
void test_dcgan(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    int imlayer = 0;

    imlayer = net->n-1;

    while(1){
        image im = make_image(net->w, net->h, net->c);
        int i;
        for(i = 0; i < im.w*im.h*im.c; ++i){
            im.data[i] = rand_normal();
        }
        //float mag = mag_array(im.data, im.w*im.h*im.c);
        //scale_array(im.data, im.w*im.h*im.c, 1./mag);

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);
        //yuv_to_rgb(out);
        normalize_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");
        show_image(out, "out", 0);

        free_image(im);
    }
}
```

함수 이름: test\_dcgan&#x20;

입력:

* cfgfile: 문자열 포인터. 네트워크의 구성 설정 파일 경로입니다.
* weightfile: 문자열 포인터. 네트워크의 가중치 파일 경로입니다.

동작:&#x20;

* 이 함수는 주어진 구성 설정 파일(cfgfile)과 가중치 파일(weightfile)을 사용하여 DCGAN(Deep Convolutional Generative Adversarial Network)을 테스트합니다.&#x20;
* DCGAN은 이미지 생성을 위한 생성자(generator)와 이미지 식별을 위한 판별자(discriminator)를 포함하는 딥러닝 모델입니다.&#x20;
* 이 함수는 생성자 네트워크를 사용하여 이미지를 생성하고, 생성된 이미지를 출력하고 저장합니다.

설명:&#x20;

* 이 함수는 구성 설정 파일(cfgfile)과 가중치 파일(weightfile)을 사용하여 네트워크 객체(net)를 로드합니다.&#x20;
* 그리고 set\_batch\_network 함수를 사용하여 네트워크의 배치 크기를 1로 설정합니다.&#x20;
* srand 함수를 사용하여 난수 시드를 설정합니다.
* 그 후, 무한 루프를 실행하면서 다음 작업을 반복합니다:

1. make\_image 함수를 사용하여 네트워크의 가로(net->w), 세로(net->h), 채널(net->c)을 크기로 가지는 이미지(im)를 생성합니다.
2. 이미지의 각 픽셀을 rand\_normal 함수를 사용하여 무작위로 초기화합니다.
3. network\_predict 함수를 사용하여 생성자 네트워크를 통해 이미지를 예측합니다.
4. get\_network\_image\_layer 함수를 사용하여 예측 결과 중 원하는 레이어(imlayer)의 이미지(out)를 가져옵니다.
5. normalize\_image 함수를 사용하여 이미지(out)를 정규화합니다.
6. 생성된 이미지(out)를 출력하고 "out"이라는 이름으로 저장합니다.
7. 사용한 이미지(im)를 메모리에서 해제합니다.



## set\_network\_alpha\_beta

<pre class="language-c"><code class="lang-c">void set_network_alpha_beta(network *net, float alpha, float beta)
{
    int i;
    for(i = 0; i &#x3C; net->n; ++i){
        if(net->layers[i].type == SHORTCUT){
            net->layers[i].alpha = alpha;
<strong>            net->layers[i].beta = beta;
</strong>        }
    }
}
</code></pre>

\
함수 이름: set\_network\_alpha\_beta&#x20;

입력:

* net: network 구조체 포인터. 네트워크 객체를 나타냅니다.
* alpha: 부동 소수점 값. alpha 매개변수로 사용할 값입니다.
* beta: 부동 소수점 값. beta 매개변수로 사용할 값입니다.

동작:&#x20;

* 이 함수는 주어진 네트워크 객체(net)의 각 레이어 중 타입이 SHORTCUT인 레이어의 alpha와 beta 값을 설정합니다.&#x20;
* SHORTCUT 레이어는 skip 연결을 나타내며, alpha와 beta는 이러한 skip 연결의 가중치를 조절하는 데 사용됩니다.

설명:&#x20;

* 이 함수는 네트워크의 레이어를 반복하면서 각 레이어의 타입을 확인합니다.&#x20;
* 만약 레이어의 타입이 SHORTCUT인 경우, 해당 레이어의 alpha와 beta 값을 주어진 alpha와 beta로 설정합니다.&#x20;
* 이렇게 함으로써 SHORTCUT 레이어의 가중치를 조절할 수 있습니다.



## train\_prog

```c
void train_prog(char *cfg, char *weight, char *acfg, char *aweight, int clear, int display, char *train_images, int maxbatch)
{
#ifdef GPU
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network *gnet = load_network(cfg, weight, clear);
    network *anet = load_network(acfg, aweight, clear);

    int i, j, k;
    layer imlayer = gnet->layers[gnet->n-1];

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", gnet->learning_rate, gnet->momentum, gnet->decay);
    int imgs = gnet->batch*gnet->subdivisions;
    i = *gnet->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(anet);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;
    args.threads=16;
    args.classes = 1;
    char *ls[2] = {"imagenet", "zzzzzzzz"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    gnet->train = 1;
    anet->train = 1;

    int x_size = gnet->inputs*gnet->batch;
    int y_size = gnet->truths*gnet->batch;
    float *imerror = cuda_make_array(0, y_size);

    float aloss_avg = -1;

    if (maxbatch == 0) maxbatch = gnet->max_batches;
    while (get_current_batch(gnet) < maxbatch) {
        {
            int cb = get_current_batch(gnet);
            float alpha = (float) cb / (maxbatch/2);
            if(alpha > 1) alpha = 1;
            float beta = 1 - alpha;
            printf("%f %f\n", alpha, beta);
            set_network_alpha_beta(gnet, alpha, beta);
            set_network_alpha_beta(anet, beta, alpha);
        }

        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gen = copy_data(train);
        for (j = 0; j < imgs; ++j) {
            train.y.vals[j][0] = 1;
            gen.y.vals[j][0] = 0;
        }
        time=clock();

        for (j = 0; j < gnet->subdivisions; ++j) {
            get_next_batch(train, gnet->batch, j*gnet->batch, gnet->truth, 0);
            int z;
            for(z = 0; z < x_size; ++z){
                gnet->input[z] = rand_normal();
            }
            /*
               for(z = 0; z < gnet->batch; ++z){
               float mag = mag_array(gnet->input + z*gnet->inputs, gnet->inputs);
               scale_array(gnet->input + z*gnet->inputs, gnet->inputs, 1./mag);
               }
             */
            *gnet->seen += gnet->batch;
            forward_network(gnet);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            fill_cpu(anet->truths*anet->batch, 1, anet->truth, 1);
            copy_cpu(anet->inputs*anet->batch, imlayer.output, 1, anet->input, 1);
            anet->delta_gpu = imerror;
            forward_network(anet);
            backward_network(anet);

            //float genaloss = *anet->cost / anet->batch;

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
            scal_gpu(imlayer.outputs*imlayer.batch, 0, gnet->layers[gnet->n-1].delta_gpu, 1);

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);

            backward_network(gnet);

            for(k = 0; k < gnet->batch; ++k){
                int index = j*gnet->batch + k;
                copy_cpu(gnet->outputs, gnet->output + k*gnet->outputs, 1, gen.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gen);
        float aloss = train_network(anet, merge);

#ifdef OPENCV
        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gen.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen", 1);
            show_image(im2, "train", 1);
            save_image(im, "gen");
            save_image(im2, "train");
        }
#endif

        update_network_gpu(gnet);

        free_data(merge);
        free_data(train);
        free_data(gen);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;

        printf("%d: adv: %f | adv_avg: %f, %f rate, %lf seconds, %d images\n", i, aloss, aloss_avg, get_current_rate(gnet), sec(clock()-time), i*imgs);
        if(i%10000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(gnet, buff);
#endif
}
```

함수 이름: train\_prog&#x20;

입력:

* cfg: 문자열 포인터. GAN 네트워크의 설정 파일 경로.
* weight: 문자열 포인터. 사전 학습된 GAN 네트워크의 가중치 파일 경로.
* acfg: 문자열 포인터. 판별자(Discriminator) 네트워크의 설정 파일 경로.
* aweight: 문자열 포인터. 사전 학습된 판별자(Discriminator) 네트워크의 가중치 파일 경로.
* clear: 정수. 학습 전 네트워크를 초기화할지 여부를 나타내는 플래그. 1로 설정하면 초기화됩니다.
* display: 정수. 학습 중 생성된 이미지를 표시할지 여부를 나타내는 플래그. 1로 설정하면 표시됩니다.
* train\_images: 문자열 포인터. 학습에 사용할 이미지 데이터의 경로.
* maxbatch: 정수. 최대 배치 수.

동작:&#x20;

* 이 함수는 Deep Convolutional GAN(DCGAN)을 학습하는 역할을 수행합니다.&#x20;
* 입력으로 주어진 설정 파일과 가중치 파일을 사용하여 GAN 및 판별자 네트워크를 로드합니다.&#x20;
* 학습 이미지 데이터를 로드하고, 네트워크를 초기화한 후 학습을 시작합니다. 학습은 주어진 최대 배치 수(maxbatch)에 도달할 때까지 반복됩니다.&#x20;
* 각 배치에서는 생성자(Generator)와 판별자(Discriminator) 네트워크를 번갈아가며 업데이트하고, 오차를 계산하여 학습합니다.&#x20;
* 학습 중에는 생성된 이미지를 표시하고, 일정 주기마다 가중치를 저장합니다.

설명:&#x20;

* 이 함수는 주로 GPU를 사용하는 경우를 가정하고 작성되었습니다.&#x20;
* GPU 지원을 사용할 수 없는 경우 해당 부분은 실행되지 않을 수 있습니다.&#x20;
* 또한, 학습 중 생성된 이미지를 표시하려면 OPENCV가 필요합니다.



## train\_dcgan

```c
void train_dcgan(char *cfg, char *weight, char *acfg, char *aweight, int clear, int display, char *train_images, int maxbatch)
{
#ifdef GPU
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network *gnet = load_network(cfg, weight, clear);
    network *anet = load_network(acfg, aweight, clear);
    //float orig_rate = anet->learning_rate;

    int i, j, k;
    layer imlayer = {0};
    for (i = 0; i < gnet->n; ++i) {
        if (gnet->layers[i].out_c == 3) {
            imlayer = gnet->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", gnet->learning_rate, gnet->momentum, gnet->decay);
    int imgs = gnet->batch*gnet->subdivisions;
    i = *gnet->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(anet);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;
    args.threads=16;
    args.classes = 1;
    char *ls[2] = {"imagenet", "zzzzzzzz"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    gnet->train = 1;
    anet->train = 1;

    int x_size = gnet->inputs*gnet->batch;
    int y_size = gnet->truths*gnet->batch;
    float *imerror = cuda_make_array(0, y_size);

    //int ay_size = anet->truths*anet->batch;

    float aloss_avg = -1;

    //data generated = copy_data(train);

    if (maxbatch == 0) maxbatch = gnet->max_batches;
    while (get_current_batch(gnet) < maxbatch) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        //translate_data_rows(train, -.5);
        //scale_data_rows(train, 2);

        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gen = copy_data(train);
        for (j = 0; j < imgs; ++j) {
            train.y.vals[j][0] = 1;
            gen.y.vals[j][0] = 0;
        }
        time=clock();

        for(j = 0; j < gnet->subdivisions; ++j){
            get_next_batch(train, gnet->batch, j*gnet->batch, gnet->truth, 0);
            int z;
            for(z = 0; z < x_size; ++z){
                gnet->input[z] = rand_normal();
            }
            for(z = 0; z < gnet->batch; ++z){
                float mag = mag_array(gnet->input + z*gnet->inputs, gnet->inputs);
                scale_array(gnet->input + z*gnet->inputs, gnet->inputs, 1./mag);
            }
            /*
               for(z = 0; z < 100; ++z){
               printf("%f, ", gnet->input[z]);
               }
               printf("\n");
               printf("input: %f %f\n", mean_array(gnet->input, x_size), variance_array(gnet->input, x_size));
             */

            //cuda_push_array(gnet->input_gpu, gnet->input, x_size);
            //cuda_push_array(gnet->truth_gpu, gnet->truth, y_size);
            *gnet->seen += gnet->batch;
            forward_network(gnet);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            fill_cpu(anet->truths*anet->batch, 1, anet->truth, 1);
            copy_cpu(anet->inputs*anet->batch, imlayer.output, 1, anet->input, 1);
            anet->delta_gpu = imerror;
            forward_network(anet);
            backward_network(anet);

            //float genaloss = *anet->cost / anet->batch;
            //printf("%f\n", genaloss);

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
            scal_gpu(imlayer.outputs*imlayer.batch, 0, gnet->layers[gnet->n-1].delta_gpu, 1);

            //printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
            //printf("features %f\n", cuda_mag_array(gnet->layers[gnet->n-1].delta_gpu, imlayer.outputs*imlayer.batch));

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);

            backward_network(gnet);

            /*
               for(k = 0; k < gnet->n; ++k){
               layer l = gnet->layers[k];
               cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
               printf("%d: %f %f\n", k, mean_array(l.output, l.outputs*l.batch), variance_array(l.output, l.outputs*l.batch));
               }
             */

            for(k = 0; k < gnet->batch; ++k){
                int index = j*gnet->batch + k;
                copy_cpu(gnet->outputs, gnet->output + k*gnet->outputs, 1, gen.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gen);
        //randomize_data(merge);
        float aloss = train_network(anet, merge);

        //translate_image(im, 1);
        //scale_image(im, .5);
        //translate_image(im2, 1);
        //scale_image(im2, .5);
#ifdef OPENCV
        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gen.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen", 1);
            show_image(im2, "train", 1);
            save_image(im, "gen");
            save_image(im2, "train");
        }
#endif

        /*
           if(aloss < .1){
           anet->learning_rate = 0;
           } else if (aloss > .3){
           anet->learning_rate = orig_rate;
           }
         */

        update_network_gpu(gnet);

        free_data(merge);
        free_data(train);
        free_data(gen);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;

        printf("%d: adv: %f | adv_avg: %f, %f rate, %lf seconds, %d images\n", i, aloss, aloss_avg, get_current_rate(gnet), sec(clock()-time), i*imgs);
        if(i%10000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(gnet, buff);
#endif
}
```

함수 이름: train\_dcgan&#x20;

입력:

* cfg: DCGAN의 구성 파일 경로 (문자열)
* weight: 사전 학습된 DCGAN 모델의 가중치 파일 경로 (문자열)
* acfg: Adversarial Network(경쟁 신경망)의 구성 파일 경로 (문자열)
* aweight: 사전 학습된 Adversarial Network 모델의 가중치 파일 경로 (문자열)
* clear: 가중치 초기화 여부 (정수, 0 또는 1)
* display: 학습 과정에서 이미지를 표시할지 여부 (정수, 0 또는 1)
* train\_images: 학습에 사용할 이미지 경로가 포함된 파일의 경로 (문자열)
* maxbatch: 최대 배치 수 (정수)

동작:&#x20;

* &#x20;DCGAN 및 Adversarial Network을 사용하여 모델을 학습하는 함수입니다. GPU를 사용할 때만 동작합니다.

설명:

* 주어진 경로로부터 DCGAN 및 Adversarial Network 모델을 로드합니다.
* 이미지 처리를 위한 레이어를 찾고 초기화 작업을 수행합니다.
* 학습 이미지를 로드하고 데이터를 처리하기 위한 인자를 설정합니다.
* 데이터를 비동기적으로 로드하고, DCGAN과 Adversarial Network를 학습합니다.
* 학습 진행 상황을 출력하고, 주기적으로 모델 가중치를 저장합니다.
* 학습이 완료되면 최종 모델 가중치를 저장합니다.
* 학습 과정에서 display 옵션이 활성화된 경우 이미지를 표시하고 저장합니다.



## train\_colorizer

```c
void train_colorizer(char *cfg, char *weight, char *acfg, char *aweight, int clear, int display)
{
#ifdef GPU
    //char *train_images = "/home/pjreddie/data/coco/train1.txt";
    //char *train_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    char *train_images = "/home/pjreddie/data/imagenet/imagenet1k.train.list";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network *net = load_network(cfg, weight, clear);
    network *anet = load_network(acfg, aweight, clear);

    int i, j, k;
    layer imlayer = {0};
    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = net->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(net);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[2] = {"imagenet"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    int x_size = net->inputs*net->batch;
    //int y_size = x_size;
    net->delta = 0;
    net->train = 1;
    float *pixs = calloc(x_size, sizeof(float));
    float *graypixs = calloc(x_size, sizeof(float));
    //float *y = calloc(y_size, sizeof(float));

    //int ay_size = anet->outputs*anet->batch;
    anet->delta = 0;
    anet->train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs*imlayer.batch);

    float aloss_avg = -1;
    float gloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(net) < net->max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gray = copy_data(train);
        for(j = 0; j < imgs; ++j){
            image gim = float_to_image(net->w, net->h, net->c, gray.X.vals[j]);
            grayscale_image_3c(gim);
            train.y.vals[j][0] = .95;
            gray.y.vals[j][0] = .05;
        }
        time=clock();
        float gloss = 0;

        for(j = 0; j < net->subdivisions; ++j){
            get_next_batch(train, net->batch, j*net->batch, pixs, 0);
            get_next_batch(gray, net->batch, j*net->batch, graypixs, 0);
            cuda_push_array(net->input_gpu, graypixs, net->inputs*net->batch);
            cuda_push_array(net->truth_gpu, pixs, net->truths*net->batch);
            /*
               image origi = float_to_image(net->w, net->h, 3, pixs);
               image grayi = float_to_image(net->w, net->h, 3, graypixs);
               show_image(grayi, "gray");
               show_image(origi, "orig");
               cvWaitKey(0);
             */
            *net->seen += net->batch;
            forward_network_gpu(net);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            copy_gpu(anet->inputs*anet->batch, imlayer.output_gpu, 1, anet->input_gpu, 1);
            fill_gpu(anet->inputs*anet->batch, .95, anet->truth_gpu, 1);
            anet->delta_gpu = imerror;
            forward_network_gpu(anet);
            backward_network_gpu(anet);

            scal_gpu(imlayer.outputs*imlayer.batch, 1./100., net->layers[net->n-1].delta_gpu, 1);

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);

            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs*imlayer.batch));

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, net->layers[net->n-1].delta_gpu, 1);

            backward_network_gpu(net);


            gloss += *net->cost /(net->subdivisions*net->batch);

            for(k = 0; k < net->batch; ++k){
                int index = j*net->batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, gray.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gray);
        //randomize_data(merge);
        float aloss = train_network(anet, merge);

        update_network_gpu(net);

#ifdef OPENCV
        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gray.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen", 1);
            show_image(im2, "train", 1);
        }
#endif
        free_data(merge);
        free_data(train);
        free_data(gray);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;
        gloss_avg = gloss_avg*.9 + gloss*.1;

        printf("%d: gen: %f, adv: %f | gen_avg: %f, adv_avg: %f, %f rate, %lf seconds, %d images\n", i, gloss, aloss, gloss_avg, aloss_avg, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
#endif
}
```

함수 이름: train\_colorizer

입력:

* char \*cfg: 컬러라이저(Colorizer) 모델의 설정 파일 경로
* char \*weight: 컬러라이저 모델의 가중치 파일 경로
* char \*acfg: 컨디셔널(Conditional) 모델의 설정 파일 경로
* char \*aweight: 컨디셔널 모델의 가중치 파일 경로
* int clear: 가중치 초기화 여부를 나타내는 플래그
* int display: 중간 과정을 화면에 표시할지 여부를 나타내는 플래그

동작:

* 주어진 설정과 가중치를 사용하여 컬러라이저(Colorizer) 모델과 컨디셔널(Conditional) 모델을 로드한다.
* 학습 데이터를 로드하고 전처리 작업을 수행한다.
* 컬러 이미지와 흑백 이미지를 생성하고, 컨디셔널 모델의 학습을 진행한다.
* 컬러라이저 모델을 업데이트하고, 중간 결과를 화면에 표시한다.
* 주기적으로 가중치를 백업하고 저장한다.

설명:

* 이 함수는 컬러라이저(Colorizer) 모델을 학습시키는 함수이다.
* 주어진 설정 파일과 가중치 파일을 사용하여 컬러라이저 모델과 컨디셔널 모델을 로드한다.
* 학습 데이터를 로드하기 위해 경로 및 데이터 관련 설정을 초기화한다.
* 데이터를 로드하는 스레드를 시작하고, 로드 완료까지 대기한다.
* 로드된 데이터를 기반으로 흑백 이미지를 생성하고, 컨디셔널 모델의 입력으로 사용한다.
* 컬러라이저 모델을 업데이트하고, 중간 결과를 화면에 표시한다(OpenCV 사용).
* 학습 중에는 주기적으로 가중치를 백업하고 저장한다.
* 학습이 완료되면 최종 가중치를 저장한다.
* 주요한 변수와 동작은 다음과 같다:
  * cfg: 컬러라이저(Colorizer) 모델의 설정 파일 경로
  * weight: 컬러라이저 모델의 가중치 파일 경로
  * acfg: 컨디셔널(Conditional) 모델의 설정 파일 경로
  * aweight: 컨디셔널 모델의 가중치 파일 경로
  * clear: 가중치 초기화 여부를 나타내는 플래그
  * display: 중간 과정을 화면에 표시할지 여부를 나타내는 플래그



## test\_lsd

```c
void test_lsd(char *cfg, char *weights, char *filename, int gray)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    int i, imlayer = 0;

    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = i;
            printf("%d\n", i);
            break;
        }
    }

    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        if(gray) grayscale_image_3c(crop);

        float *X = crop.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);
        //yuv_to_rgb(out);
        constrain_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");
        show_image(out, "out", 1);
        show_image(crop, "crop", 0);

        free_image(im);
        free_image(resized);
        free_image(crop);
        if (filename) break;
    }
}
```

함수 이름: test\_lsd

입력:

* char \*cfg: LSD 테스트에 사용되는 설정 파일 경로
* char \*weights: LSD 테스트에 사용되는 가중치 파일 경로
* char \*filename: 테스트할 이미지 파일 경로
* int gray: 그레이스케일 이미지로 변환할지 여부를 나타내는 플래그

동작:

* 주어진 설정과 가중치를 사용하여 LSD (Line Segment Detection) 모델을 로드한다.
* LSD 모델에 대해 테스트 이미지를 입력으로 전달하고 예측 결과를 얻는다.
* 예측 결과를 가공하여 출력하고, 결과 이미지를 저장하고 표시한다.

설명:

* 이 함수는 LSD (Line Segment Detection) 모델을 테스트하는 함수이다.
* 주어진 설정 파일과 가중치 파일을 사용하여 LSD 모델을 로드한다.
* 테스트할 이미지 파일의 경로를 입력으로 받는다. 그렇지 않은 경우 사용자로부터 경로를 입력받는다.
* 테스트 이미지를 로드하고, 크기를 조정하고, 필요에 따라 그레이스케일로 변환한다.
* 변환된 이미지를 LSD 모델에 입력으로 전달하여 예측 결과를 얻는다.
* 예측 결과를 출력하고, 결과 이미지를 저장하고 표시한다.
* 테스트 작업을 반복하여 여러 이미지에 대해 테스트할 수 있다.
* 주요한 변수와 동작은 다음과 같다:
  * cfg: LSD 테스트에 사용되는 설정 파일 경로
  * weights: LSD 테스트에 사용되는 가중치 파일 경로
  * filename: 테스트할 이미지 파일 경로
  * gray: 그레이스케일 이미지로 변환할지 여부를 나타내는 플래그
  * imlayer: 출력 이미지를 가져올 레이어 인덱스
* 입력된 이미지 파일의 경로를 확인하고, 존재하는 경우 해당 경로를 입력으로 사용한다.
* 존재하지 않는 경우 사용자로부터 이미지 파일 경로를 입력받는다.
* 이미지 파일을 로드하고, 크기를 조정하고, 필요에 따라 그레이스케일로 변환한다.
* 변환된 이미지를 LSD 모델에 입력으로 전달하여 예측 결과를 얻는다.
* 결과 이미지를 출력하고, 저장한다.
* 테스트 작업을 반복하여 다른 이미지에 대해 테스트할 수 있다.



## run\_lsd

```c
void run_lsd(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    int batches = find_int_arg(argc, argv, "-b", 0);
    char *file = find_char_arg(argc, argv, "-file", "/home/pjreddie/data/imagenet/imagenet1k.train.list");

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    char *acfg = argv[5];
    char *aweights = (argc > 6) ? argv[6] : 0;
    //if(0==strcmp(argv[2], "train")) train_lsd(cfg, weights, clear);
    //else if(0==strcmp(argv[2], "train2")) train_lsd2(cfg, weights, acfg, aweights, clear);
    //else if(0==strcmp(argv[2], "traincolor")) train_colorizer(cfg, weights, acfg, aweights, clear);
    //else if(0==strcmp(argv[2], "train3")) train_lsd3(argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], clear);
    if(0==strcmp(argv[2], "traingan")) train_dcgan(cfg, weights, acfg, aweights, clear, display, file, batches);
    else if(0==strcmp(argv[2], "trainprog")) train_prog(cfg, weights, acfg, aweights, clear, display, file, batches);
    else if(0==strcmp(argv[2], "traincolor")) train_colorizer(cfg, weights, acfg, aweights, clear, display);
    else if(0==strcmp(argv[2], "gan")) test_dcgan(cfg, weights);
    else if(0==strcmp(argv[2], "inter")) inter_dcgan(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_lsd(cfg, weights, filename, 0);
    else if(0==strcmp(argv[2], "color")) test_lsd(cfg, weights, filename, 1);
    /*
       else if(0==strcmp(argv[2], "valid")) validate_lsd(cfg, weights);
     */
}
```

함수 이름: run\_lsd

입력:

* int argc: 프로그램 실행 시 전달되는 명령행 인자의 개수
* char \*\*argv: 프로그램 실행 시 전달되는 명령행 인자 배열

동작:

* 주어진 명령행 인자를 기반으로 LSD (Line Segment Detection) 관련 작업을 실행하는 함수이다.
* 프로그램 실행 시 필요한 인자의 개수를 체크하고, 부족한 경우 사용법을 출력하고 함수를 종료한다.
* 다양한 옵션과 함께 LSD 관련 작업을 실행한다.
* 주어진 명령행 인자를 분석하여 해당하는 작업을 수행한다.

설명:

* 이 함수는 LSD (Line Segment Detection) 관련 작업을 실행하는 함수이다.
* 프로그램 실행 시 전달되는 명령행 인자를 기반으로 작업을 결정하고 해당 작업을 수행한다.
* 프로그램 실행 시 인자의 개수가 충분하지 않은 경우, 즉시 사용법을 출력하고 함수를 종료한다.
* 주요한 인자와 작업은 다음과 같다:
  * cfg: LSD 작업에 사용되는 설정 파일 경로
  * weights: LSD 작업에 사용되는 가중치 파일 경로 (옵션)
  * filename: 테스트 작업에 사용되는 파일 경로 (옵션)
  * acfg: 다른 네트워크의 설정 파일 경로 (옵션)
  * aweights: 다른 네트워크의 가중치 파일 경로 (옵션)
  * clear: 작업 실행 전에 이전의 데이터를 지울지 여부를 나타내는 플래그
  * display: 결과를 표시할지 여부를 나타내는 플래그
  * batches: 배치 크기를 지정하는 정수 값
  * file: 데이터 파일 경로
* 주어진 명령행 인자를 분석하여 해당하는 작업을 수행한다.
* 주석 처리된 코드는 주석을 해제하고 해당하는 함수를 호출하는 부분이다.
* 주로 `train_*`, `test_*` 등의 함수를 호출하여 LSD 관련 작업을 수행한다.

