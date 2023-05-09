# rnn\_vid

```c
#include "darknet.h"

#ifdef OPENCV
image get_image_from_stream(CvCapture *cap);
image ipl_to_image(IplImage* src);

void reconstruct_picture(network net, float *features, image recon, image update, float rate, float momentum, float lambda, int smooth_size, int iters);

typedef struct {
    float *x;
    float *y;
} float_pair;
```

## get\_rnn\_vid\_data

```c
float_pair get_rnn_vid_data(network net, char **files, int n, int batch, int steps)
{
    int b;
    assert(net.batch == steps + 1);
    image out_im = get_network_image(net);
    int output_size = out_im.w*out_im.h*out_im.c;
    printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
    float *feats = calloc(net.batch*batch*output_size, sizeof(float));
    for(b = 0; b < batch; ++b){
        int input_size = net.w*net.h*net.c;
        float *input = calloc(input_size*net.batch, sizeof(float));
        char *filename = files[rand()%n];
        CvCapture *cap = cvCaptureFromFile(filename);
        int frames = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
        int index = rand() % (frames - steps - 2);
        if (frames < (steps + 4)){
            --b;
            free(input);
            continue;
        }

        printf("frames: %d, index: %d\n", frames, index);
        cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, index);

        int i;
        for(i = 0; i < net.batch; ++i){
            IplImage* src = cvQueryFrame(cap);
            image im = ipl_to_image(src);
            rgbgr_image(im);
            image re = resize_image(im, net.w, net.h);
            //show_image(re, "loaded");
            //cvWaitKey(10);
            memcpy(input + i*input_size, re.data, input_size*sizeof(float));
            free_image(im);
            free_image(re);
        }
        float *output = network_predict(net, input);

        free(input);

        for(i = 0; i < net.batch; ++i){
            memcpy(feats + (b + i*batch)*output_size, output + i*output_size, output_size*sizeof(float));
        }

        cvReleaseCapture(&cap);
    }

    //printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
    float_pair p = {0};
    p.x = feats;
    p.y = feats + output_size*batch; //+ out_im.w*out_im.h*out_im.c;

    return p;
}
```

함수 이름: get\_rnn\_vid\_data&#x20;

입력:

* network net: RNN 네트워크 모델
* char \*\*files: 비디오 파일 경로 배열
* int n: 비디오 파일 경로 배열의 크기
* int batch: 배치 크기
* int steps: RNN 입력 시퀀스의 길이

동작:&#x20;

* 주어진 비디오 파일에서 임의의 프레임을 선택하고, 선택된 프레임으로부터 RNN 모델에 입력으로 사용될 시퀀스를 생성하여 네트워크를 실행합니다.&#x20;
* 이 과정을 지정된 배치 크기만큼 반복하여 각각의 배치에 대한 출력을 추출하고, 이를 한 번에 하나의 큰 2D 배열로 합칩니다.&#x20;
* 합쳐진 출력은 float\_pair 구조체로 반환됩니다.&#x20;
* 반환된 float\_pair 구조체는 두 개의 포인터(float \*)를 갖는데, 첫 번째 포인터는 RNN 모델의 출력 중 첫 번째 배치에 대한 결과를 가리키고, 두 번째 포인터는 두 번째 배치에 대한 결과를 가리킵니다.

설명:&#x20;

* get\_rnn\_vid\_data 함수는 RNN 모델의 비디오 입력 데이터를 생성하는 함수입니다.&#x20;
* 함수는 비디오 파일 경로 배열에서 임의의 파일을 선택하고, 해당 파일에서 임의의 프레임을 선택하여 이를 이용하여 RNN 입력 시퀀스를 생성합니다.&#x20;
* RNN 모델은 이 시퀀스를 실행하여 출력을 생성하고, 이 출력은 지정된 배치 크기만큼 모으고, float\_pair 구조체로 반환됩니다.
* 이 함수는 입력으로 RNN 모델과 비디오 파일 경로 배열, 배치 크기, RNN 입력 시퀀스의 길이를 받습니다.&#x20;
* RNN 모델은 네트워크 구조와 가중치 등의 정보를 담고 있으며, 비디오 파일 경로 배열은 입력 데이터를 가져올 파일 경로들을 저장합니다.
* &#x20;배치 크기는 RNN 모델에 입력으로 주어지는 시퀀스의 길이입니다. 이는 RNN 모델의 내부 상태를 조절하는 데 사용됩니다.&#x20;
* RNN 입력 시퀀스의 길이는 이전 RNN 출력을 다음 입력으로 사용하는 시퀀스의 길이를 결정합니다.
* 함수의 출력은 float\_pair 구조체입니다. 이 구조체는 두 개의 포인터를 갖습니다.&#x20;
* 첫 번째 포인터는 RNN 모델의 출력 중 첫 번째 배치에 대한 결과를 가리키고, 두 번째 포인터는 두 번째 배치에 대한 결과를 가리킵니다.



## train\_vid\_rnn

```c
void train_vid_rnn(char *cfgfile, char *weightfile)
{
    char *train_videos = "data/vid/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;

    list *plist = get_paths(train_videos);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);
    clock_t time;
    int steps = net.time_steps;
    int batch = net.batch / net.time_steps;

    network extractor = parse_network_cfg("cfg/extractor.cfg");
    load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        float_pair p = get_rnn_vid_data(extractor, paths, N, batch, steps);

        copy_cpu(net.inputs*net.batch, p.x, 1, net.input, 1);
        copy_cpu(net.truths*net.batch, p.y, 1, net.truth, 1);
        float loss = train_network_datum(net) / (net.batch);


        free(p.x);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time));
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%10==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```

함수 이름: train\_vid\_rnn

입력:

* cfgfile: char 포인터. RNN 네트워크의 설정 파일 경로.
* weightfile: char 포인터. 미리 학습된 가중치 파일 경로.

동작:&#x20;

* train\_vid\_rnn 함수는 RNN 네트워크를 학습시키는 함수입니다.&#x20;
* 주어진 cfgfile과 weightfile을 사용하여 네트워크를 초기화한 후, 지정된 비디오 파일에서 무작위로 추출한 프레임들로부터 입력 데이터를 생성하여 RNN 네트워크를 학습시킵니다.&#x20;
* 학습 중에는 네트워크의 가중치 파일을 주기적으로 저장하고, 학습 중에 발생하는 손실을 출력합니다.

설명:

* plist: list 포인터. 학습할 비디오 파일들의 경로를 포함하는 list입니다.
* N: int 형. 학습할 비디오 파일들의 수입니다.
* paths: char 이중 포인터. 학습할 비디오 파일들의 경로입니다.
* time: clock\_t 형. 현재 시간을 나타내는 clock\_t 변수입니다.
* steps: int 형. RNN 네트워크의 타임 스텝 수입니다.
* batch: int 형. RNN 네트워크의 배치 크기를 타임 스텝 수로 나눈 값입니다.
* net: network 구조체. 학습할 RNN 네트워크입니다.
* extractor: network 구조체. 비디오에서 특성을 추출하는 YOLO 네트워크입니다.
* avg\_loss: float 형. 이전 손실과 현재 손실을 평균화한 값입니다.
* i: int 형 포인터. 현재 학습 배치의 인덱스입니다.
* imgs: int 형. 하나의 배치에 들어가는 이미지 수입니다. (batch \* subdivisions)
* train\_videos: char 포인터. 학습할 비디오 파일들의 목록이 포함된 파일의 경로입니다.
* backup\_directory: char 포인터. 가중치 파일 및 백업 파일을 저장할 디렉토리의 경로입니다.



## save\_reconstruction

```c
image save_reconstruction(network net, image *init, float *feat, char *name, int i)
{
    image recon;
    if (init) {
        recon = copy_image(*init);
    } else {
        recon = make_random_image(net.w, net.h, 3);
    }

    image update = make_image(net.w, net.h, 3);
    reconstruct_picture(net, feat, recon, update, .01, .9, .1, 2, 50);
    char buff[256];
    sprintf(buff, "%s%d", name, i);
    save_image(recon, buff);
    free_image(update);
    return recon;
}
```

함수 이름: save\_reconstruction

입력:

* network net: 신경망 모델
* image \*init: 초기 이미지 포인터 (선택적)
* float \*feat: 특징 벡터
* char \*name: 저장할 이미지 이름의 prefix
* int i: 이미지 이름에 붙일 번호

동작:

* 주어진 특징 벡터 feat를 사용하여 입력 이미지를 재구성한다.
* 초기 이미지 포인터 init가 제공되면, 해당 이미지로부터 재구성을 시작한다.
* 재구성된 이미지를 파일로 저장하고, 해당 이미지를 반환한다.

설명:

* 이 함수는 Variational Autoencoder (VAE)와 같은 신경망 모델을 사용하여 입력 이미지를 재구성하는 기능을 수행한다.
* 입력으로는 VAE 모델 (network net), 재구성을 시작할 초기 이미지 (image \*init), 입력 이미지를 특징 벡터로 변환한 값 (float \*feat), 저장할 이미지 이름의 prefix (char \*name), 이미지 이름에 붙일 번호 (int i)가 제공된다.
* 만약 초기 이미지 포인터 init가 제공되면, 해당 이미지로부터 재구성을 시작하고, 그렇지 않으면 무작위 이미지를 생성하여 재구성을 시작한다.
* 재구성된 이미지는 update 이미지를 사용하여 iter\_num 회수만큼 업데이트하며, 최종적으로 재구성된 이미지를 파일로 저장하고 해당 이미지를 반환한다.



## generate\_vid\_rnn

```c
void generate_vid_rnn(char *cfgfile, char *weightfile)
{
    network extractor = parse_network_cfg("cfg/extractor.recon.cfg");
    load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&extractor, 1);
    set_batch_network(&net, 1);

    int i;
    CvCapture *cap = cvCaptureFromFile("/extra/vid/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00007030.mp4");
    float *feat;
    float *next;
    image last;
    for(i = 0; i < 25; ++i){
        image im = get_image_from_stream(cap);
        image re = resize_image(im, extractor.w, extractor.h);
        feat = network_predict(extractor, re.data);
        if(i > 0){
            printf("%f %f\n", mean_array(feat, 14*14*512), variance_array(feat, 14*14*512));
            printf("%f %f\n", mean_array(next, 14*14*512), variance_array(next, 14*14*512));
            printf("%f\n", mse_array(feat, 14*14*512));
            axpy_cpu(14*14*512, -1, feat, 1, next, 1);
            printf("%f\n", mse_array(next, 14*14*512));
        }
        next = network_predict(net, feat);

        free_image(im);

        free_image(save_reconstruction(extractor, 0, feat, "feat", i));
        free_image(save_reconstruction(extractor, 0, next, "next", i));
        if (i==24) last = copy_image(re);
        free_image(re);
    }
    for(i = 0; i < 30; ++i){
        next = network_predict(net, next);
        image new = save_reconstruction(extractor, &last, next, "new", i);
        free_image(last);
        last = new;
    }
}
```

함수 이름: generate\_vid\_rnn&#x20;

입력:

* cfgfile: YOLO 모델 구성 파일 경로 (문자열)
* weightfile: YOLO 모델 가중치 파일 경로 (문자열)

동작:&#x20;

* 이 함수는 YOLO 모델과 LSTM(Long Short-Term Memory) 모델을 사용하여 비디오 데이터를 생성하는 역할을 합니다.
* extractor: 미리 학습된 YOLO 모델을 사용하여 입력 이미지에서 특징을 추출합니다.
* net: LSTM 모델을 사용하여 특징 시퀀스를 생성합니다.
* cap: 입력 비디오 파일 경로를 가리키는 CvCapture 객체를 만듭니다.
* for 루프에서 입력 비디오에서 프레임을 읽어온 다음, extractor 모델을 사용하여 해당 이미지에서 특징을 추출하고 LSTM 모델을 사용하여 다음 특징 시퀀스를 예측합니다.
* 이후 LSTM 모델의 예측 결과를 사용하여 새로운 이미지를 생성합니다.
* 마지막으로 생성된 이미지를 반환합니다.

설명:&#x20;

* 이 함수는 YOLO와 LSTM 모델을 사용하여 비디오 데이터를 생성하는 것으로, 입력으로 YOLO 모델 구성 파일 경로와 가중치 파일 경로를 받습니다.
* 함수의 내부에서는 미리 학습된 YOLO 모델을 사용하여 입력 이미지에서 특징을 추출하고 LSTM 모델을 사용하여 이러한 특징 시퀀스를 생성합니다.
* 입력 비디오에서 프레임을 가져와 각 프레임에서 특징을 추출하고 LSTM 모델을 사용하여 다음 특징 시퀀스를 예측합니다. LSTM 모델이 예측한 다음 특징 시퀀스를 사용하여 새로운 이미지를 생성합니다.
* 마지막으로 생성된 이미지를 반환합니다.



## run\_vid\_rnn

```c
void run_vid_rnn(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    //char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_vid_rnn(cfg, weights);
    else if(0==strcmp(argv[2], "generate")) generate_vid_rnn(cfg, weights);
}
#else
void run_vid_rnn(int argc, char **argv){}
#endif
```

함수 이름: run\_vid\_rnn

입력:

* argc: int 타입. main 함수에서 전달되는 실행 인자(argument)의 개수를 나타냄.
* argv: char\*\* 타입. main 함수에서 전달되는 실행 인자(argument)들의 배열을 나타냄.

동작:

* 입력으로 받은 실행 인자(argument)에 따라 train\_vid\_rnn 함수 또는 generate\_vid\_rnn 함수를 호출함.
* 실행 인자가 충분하지 않은 경우, 사용법(usage)을 출력함.

설명:

* 이 함수는 비디오 데이터를 처리하는 recurrent neural network (RNN) 모델을 학습(train)하거나 생성(generate)하는 함수를 호출하는 역할을 함.
* 실행 인자로는 'train', 'test', 'valid' 중 하나를 지정하여 학습, 테스트, 검증 중 어떤 단계를 수행할지 결정하고, 'cfg'와 'weights'를 지정하여 모델의 설정 파일과 가중치 파일을 전달함.
* 'weights'는 선택적(optional)으로 지정할 수 있으며, 지정하지 않으면 NULL 값을 갖게 됨.
* 실행 인자가 충분하지 않은 경우, 사용법(usage)을 출력하고 함수를 종료함.

