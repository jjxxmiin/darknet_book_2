# nightmare \[n]

```c
#include "darknet.h"

#include <math.h>

// ./darknet nightmare cfg/extractor.recon.cfg ~/trained/yolo-coco.conv frame6.png -reconstruct -iters 500 -i 3 -lambda .1 -rate .01 -smooth 2
```

## abs\_mean

```c
float abs_mean(float *x, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i){
        sum += fabs(x[i]);
    }
    return sum/n;
}
```

함수 이름: abs\_mean

입력:

* x: float형 포인터. 계산할 대상 데이터의 배열.
* n: int형. 데이터의 개수.

동작:&#x20;

* 주어진 배열 x의 요소들의 절대값 평균을 계산한다.

설명:&#x20;

* 이 함수는 입력으로 주어진 배열 x의 각 요소들에 대해 절대값을 취한 후 그 평균값을 계산하여 반환한다.&#x20;
* 반환된 값은 입력된 데이터의 절대값 평균을 나타낸다.



## calculate\_loss

```c
void calculate_loss(float *output, float *delta, int n, float thresh)
{
    int i;
    float mean = mean_array(output, n);
    float var = variance_array(output, n);
    for(i = 0; i < n; ++i){
        if(delta[i] > mean + thresh*sqrt(var)) delta[i] = output[i];
        else delta[i] = 0;
    }
}
```

## optimize\_picture

```c
void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm)
{
    //scale_image(orig, 2);
    //translate_image(orig, -1);
    net->n = max_layer + 1;

    int dx = rand()%16 - 8;
    int dy = rand()%16 - 8;
    int flip = rand()%2;

    image crop = crop_image(orig, dx, dy, orig.w, orig.h);
    image im = resize_image(crop, (int)(orig.w * scale), (int)(orig.h * scale));
    if(flip) flip_image(im);

    resize_network(net, im.w, im.h);
    layer last = net->layers[net->n-1];
    //net->layers[net->n - 1].activation = LINEAR;

    image delta = make_image(im.w, im.h, im.c);

#ifdef GPU
    net->delta_gpu = cuda_make_array(delta.data, im.w*im.h*im.c);
    copy_cpu(net->inputs, im.data, 1, net->input, 1);

    forward_network_gpu(net);
    copy_gpu(last.outputs, last.output_gpu, 1, last.delta_gpu, 1);

    cuda_pull_array(last.delta_gpu, last.delta, last.outputs);
    calculate_loss(last.delta, last.delta, last.outputs, thresh);
    cuda_push_array(last.delta_gpu, last.delta, last.outputs);

    backward_network_gpu(net);

    cuda_pull_array(net->delta_gpu, delta.data, im.w*im.h*im.c);
    cuda_free(net->delta_gpu);
    net->delta_gpu = 0;
#else
    printf("\nnet: %d %d %d im: %d %d %d\n", net->w, net->h, net->inputs, im.w, im.h, im.c);
    copy_cpu(net->inputs, im.data, 1, net->input, 1);
    net->delta = delta.data;
    forward_network(net);
    copy_cpu(last.outputs, last.output, 1, last.delta, 1);
    calculate_loss(last.output, last.delta, last.outputs, thresh);
    backward_network(net);
#endif

    if(flip) flip_image(delta);
    //normalize_array(delta.data, delta.w*delta.h*delta.c);
    image resized = resize_image(delta, orig.w, orig.h);
    image out = crop_image(resized, -dx, -dy, orig.w, orig.h);

    /*
       image g = grayscale_image(out);
       free_image(out);
       out = g;
     */

    //rate = rate / abs_mean(out.data, out.w*out.h*out.c);
    image gray = make_image(out.w, out.h, out.c);
    fill_image(gray, .5);
    axpy_cpu(orig.w*orig.h*orig.c, -1, orig.data, 1, gray.data, 1);
    axpy_cpu(orig.w*orig.h*orig.c, .1, gray.data, 1, out.data, 1);

    if(norm) normalize_array(out.data, out.w*out.h*out.c);
    axpy_cpu(orig.w*orig.h*orig.c, rate, out.data, 1, orig.data, 1);

    /*
       normalize_array(orig.data, orig.w*orig.h*orig.c);
       scale_image(orig, sqrt(var));
       translate_image(orig, mean);
     */

    //translate_image(orig, 1);
    //scale_image(orig, .5);
    //normalize_image(orig);

    constrain_image(orig);

    free_image(crop);
    free_image(im);
    free_image(delta);
    free_image(resized);
    free_image(out);

}
```





## smooth

```c
void smooth(image recon, image update, float lambda, int num)
{
    int i, j, k;
    int ii, jj;
    for(k = 0; k < recon.c; ++k){
        for(j = 0; j < recon.h; ++j){
            for(i = 0; i < recon.w; ++i){
                int out_index = i + recon.w*(j + recon.h*k);
                for(jj = j-num; jj <= j + num && jj < recon.h; ++jj){
                    if (jj < 0) continue;
                    for(ii = i-num; ii <= i + num && ii < recon.w; ++ii){
                        if (ii < 0) continue;
                        int in_index = ii + recon.w*(jj + recon.h*k);
                        update.data[out_index] += lambda * (recon.data[in_index] - recon.data[out_index]);
                    }
                }
            }
        }
    }
}
```

함수 이름: smooth&#x20;

입력:

* recon: 이미지 구조체 포인터 (image\*)
* update: 이미지 구조체 포인터 (image\*)
* lambda: 부동 소수점 값 (float)
* num: 정수 (int)

동작:

* recon 이미지를 입력으로 받아, num 값에 따라 블러(흐림) 효과를 적용하고, 결과를 update 이미지에 저장한다.
* 각 채널 별로 블러링을 수행하며, num의 값이 클수록 블러링 정도가 높아진다.
* 블러링을 수행하는 과정에서, lambda 값에 따라 smoothing 효과를 적용한다.

설명:

* 입력 이미지인 recon의 각 픽셀에 대해서, 주변 num x num 픽셀의 값들의 평균을 계산하고, 해당 픽셀 값을 해당 평균 값으로 대체한다. 이를 통해 블러링 효과를 구현한다.
* 블러링을 수행하는 과정에서 smoothing 효과를 적용하기 위해, recon 이미지의 각 픽셀 값과, 블러링으로 구해진 해당 픽셀 주변 값의 평균 값을 lambda 비율로 조절하여 update 이미지에 저장한다.
* 최종적으로 smoothing 효과가 적용된 이미지는 update에 저장되며, 이후 학습을 위해 사용된다.



## reconstruct\_picture

```c
void reconstruct_picture(network *net, float *features, image recon, image update, float rate, float momentum, float lambda, int smooth_size, int iters)
{
    int iter = 0;
    for (iter = 0; iter < iters; ++iter) {
        image delta = make_image(recon.w, recon.h, recon.c);

#ifdef GPU
        layer l = get_network_output_layer(net);
        cuda_push_array(net->input_gpu, recon.data, recon.w*recon.h*recon.c);
        //cuda_push_array(net->truth_gpu, features, net->truths);
        net->delta_gpu = cuda_make_array(delta.data, delta.w*delta.h*delta.c);

        forward_network_gpu(net);
        cuda_push_array(l.delta_gpu, features, l.outputs);
        axpy_gpu(l.outputs, -1, l.output_gpu, 1, l.delta_gpu, 1);
        backward_network_gpu(net);

        cuda_pull_array(net->delta_gpu, delta.data, delta.w*delta.h*delta.c);

        cuda_free(net->delta_gpu);
#else
        net->input = recon.data;
        net->delta = delta.data;
        net->truth = features;

        forward_network(net);
        backward_network(net);
#endif

        //normalize_array(delta.data, delta.w*delta.h*delta.c);
        axpy_cpu(recon.w*recon.h*recon.c, 1, delta.data, 1, update.data, 1);
        //smooth(recon, update, lambda, smooth_size);

        axpy_cpu(recon.w*recon.h*recon.c, rate, update.data, 1, recon.data, 1);
        scal_cpu(recon.w*recon.h*recon.c, momentum, update.data, 1);

        float mag = mag_array(delta.data, recon.w*recon.h*recon.c);
        printf("mag: %f\n", mag);
        //scal_cpu(recon.w*recon.h*recon.c, 600/mag, recon.data, 1);

        constrain_image(recon);
        free_image(delta);
    }
}
```

함수 이름: reconstruct\_picture

입력:

* net: 신경망 구조체
* features: float 포인터, 입력 이미지에 대한 특징 벡터
* recon: 이미지 구조체, 재구성된 이미지를 저장할 포인터
* update: 이미지 구조체, 업데이트 된 값을 저장할 포인터
* rate: float, 경사 하강법에서 사용할 학습률
* momentum: float, 관성에 대한 가중치
* lambda: float, 재구성된 이미지를 부드럽게 만드는 정규화 가중치
* smooth\_size: int, 부드러운 효과를 위해 사용되는 커널 크기
* iters: int, 재구성을 위한 반복 횟수

동작:

* 재구성된 이미지를 업데이트하고, 신경망에서 backward propagation을 수행하여 역전파를 수행합니다.
* 업데이트된 이미지를 현재 이미지에 더하고, 부드러운 효과를 위해 smooth 함수를 사용합니다.
* 경사 하강법을 사용하여 업데이트된 이미지를 이전 이미지에 대한 학습률과 관성 가중치를 고려하여 업데이트합니다.
* 재구성된 이미지를 제약 조건에 따라 클리핑합니다.

설명:&#x20;

* reconstruct\_picture 함수는 신경망에서 특징 벡터를 기반으로 이미지를 재구성하는 함수입니다.&#x20;
* 이 함수는 역전파를 사용하여 입력 이미지의 특징 벡터와 출력 벡터 간의 오차를 줄이고, 재구성된 이미지를 업데이트합니다.&#x20;
* 이 함수는 경사 하강법을 사용하여 재구성된 이미지를 부드럽게 만들고, 업데이트된 이미지를 현재 이미지에 더하는 것으로 이미지를 재구성합니다.&#x20;
* 이 함수는 재구성된 이미지에 제약 조건을 적용하여 이미지를 클리핑합니다.



## run\_nightmare

```c
void run_nightmare(int argc, char **argv)
{
    srand(0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [cfg] [weights] [image] [layer] [options! (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[2];
    char *weights = argv[3];
    char *input = argv[4];
    int max_layer = atoi(argv[5]);

    int range = find_int_arg(argc, argv, "-range", 1);
    int norm = find_int_arg(argc, argv, "-norm", 1);
    int rounds = find_int_arg(argc, argv, "-rounds", 1);
    int iters = find_int_arg(argc, argv, "-iters", 10);
    int octaves = find_int_arg(argc, argv, "-octaves", 4);
    float zoom = find_float_arg(argc, argv, "-zoom", 1.);
    float rate = find_float_arg(argc, argv, "-rate", .04);
    float thresh = find_float_arg(argc, argv, "-thresh", 1.);
    float rotate = find_float_arg(argc, argv, "-rotate", 0);
    float momentum = find_float_arg(argc, argv, "-momentum", .9);
    float lambda = find_float_arg(argc, argv, "-lambda", .01);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    int reconstruct = find_arg(argc, argv, "-reconstruct");
    int smooth_size = find_int_arg(argc, argv, "-smooth", 1);

    network *net = load_network(cfg, weights, 0);
    char *cfgbase = basecfg(cfg);
    char *imbase = basecfg(input);

    set_batch_network(net, 1);
    image im = load_image_color(input, 0, 0);
    if(0){
        float scale = 1;
        if(im.w > 512 || im.h > 512){
            if(im.w > im.h) scale = 512.0/im.w;
            else scale = 512.0/im.h;
        }
        image resized = resize_image(im, scale*im.w, scale*im.h);
        free_image(im);
        im = resized;
    }
    //im = letterbox_image(im, net->w, net->h);

    float *features = 0;
    image update;
    if (reconstruct){
        net->n = max_layer;
        im = letterbox_image(im, net->w, net->h);
        //resize_network(&net, im.w, im.h);

        network_predict(net, im.data);
        if(net->layers[net->n-1].type == REGION){
            printf("region!\n");
            zero_objectness(net->layers[net->n-1]);
        }
        image out_im = copy_image(get_network_image(net));
        /*
           image crop = crop_image(out_im, zz, zz, out_im.w-2*zz, out_im.h-2*zz);
        //flip_image(crop);
        image f_im = resize_image(crop, out_im.w, out_im.h);
        free_image(crop);
         */
        printf("%d features\n", out_im.w*out_im.h*out_im.c);

        features = out_im.data;

        /*
        int i;
           for(i = 0; i < 14*14*512; ++i){
        //features[i] += rand_uniform(-.19, .19);
        }
        free_image(im);
        im = make_random_image(im.w, im.h, im.c);
         */
        update = make_image(im.w, im.h, im.c);
    }

    int e;
    int n;
    for(e = 0; e < rounds; ++e){
        fprintf(stderr, "Iteration: ");
        fflush(stderr);
        for(n = 0; n < iters; ++n){  
            fprintf(stderr, "%d, ", n);
            fflush(stderr);
            if(reconstruct){
                reconstruct_picture(net, features, im, update, rate, momentum, lambda, smooth_size, 1);
                //if ((n+1)%30 == 0) rate *= .5;
                show_image(im, "reconstruction", 10);
            }else{
                int layer = max_layer + rand()%range - range/2;
                int octave = rand()%octaves;
                optimize_picture(net, im, layer, 1/pow(1.33333333, octave), rate, thresh, norm);
            }
        }
        fprintf(stderr, "done\n");
        if(0){
            image g = grayscale_image(im);
            free_image(im);
            im = g;
        }
        char buff[256];
        if (prefix){
            sprintf(buff, "%s/%s_%s_%d_%06d",prefix, imbase, cfgbase, max_layer, e);
        }else{
            sprintf(buff, "%s_%s_%d_%06d",imbase, cfgbase, max_layer, e);
        }
        printf("%d %s\n", e, buff);
        save_image(im, buff);
        //show_image(im, buff, 0);

        if(rotate){
            image rot = rotate_image(im, rotate);
            free_image(im);
            im = rot;
        }
        image crop = crop_image(im, im.w * (1. - zoom)/2., im.h * (1.-zoom)/2., im.w*zoom, im.h*zoom);
        image resized = resize_image(crop, im.w, im.h);
        free_image(im);
        free_image(crop);
        im = resized;
    }
}
```

함수 이름: run\_nightmare

입력:

* argc: int 타입, 프로그램 실행 시 전달된 인자의 개수
* argv: char\*\* 타입, 프로그램 실행 시 전달된 인자들의 배열
* argv\[0]: 프로그램 실행 파일 이름
* argv\[1]: 실행 옵션
* argv\[2]: char\* 타입, 모델의 구성 파일(config file) 경로
* argv\[3]: char\* 타입, 모델의 가중치(weight) 파일 경로
* argv\[4]: char\* 타입, 입력 이미지 파일 경로
* argv\[5]: int 타입, 최대 레이어 수

동작:&#x20;

* 입력된 인자들을 파싱하고, reconstruct가 1이면 주어진 모델을 사용하여 입력 이미지를 재구성하고, 그렇지 않으면 주어진 모델을 사용하여 입력 이미지를 최적화합니다.&#x20;
* 이때, 다양한 옵션들이 설정될 수 있습니다. 최적화된 이미지는 파일로 저장됩니다.

설명:&#x20;

* 이 함수는 Darknet 프레임워크에서 사용되는 "nightmare" 기능을 구현한 함수입니다.&#x20;
* "nightmare"는 DeepDream 기술을 응용하여 주어진 이미지를 최적화하는 기술로, 이미지의 내부 feature를 시각화하는데 사용됩니다.&#x20;
* 이 함수는 C언어로 작성되어 있으며, 주어진 모델 파일과 입력 이미지를 사용하여 최적화된 이미지를 생성합니다.&#x20;
* 이 함수는 프로그램 실행 시 전달된 인자들을 파싱하여 사용하며, 옵션에 따라 최적화 과정에서 다양한 변형들이 적용됩니다.&#x20;
* 최종적으로 생성된 이미지는 파일로 저장됩니다.

