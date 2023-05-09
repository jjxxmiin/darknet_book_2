# tag

```c
#include "darknet.h"
```

## train\_tag

```c
void train_tag(char *cfgfile, char *weightfile, int clear)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "/home/pjreddie/backup/";
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, clear);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = 1024;
    list *plist = get_paths("/home/pjreddie/tag/train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.min = net->w;
    args.max = net->max_crop;
    args.size = net->w;

    args.paths = paths;
    args.classes = net->outputs;
    args.n = imgs;
    args.m = N;
    args.d = &buffer;
    args.type = TAG_DATA;

    args.angle = net->angle;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;

    fprintf(stderr, "%d classes\n", net->outputs);

    load_thread = load_data_in_thread(args);
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        load_thread = load_data_in_thread(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    pthread_join(load_thread, 0);
    free_data(buffer);
    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
```

함수 이름: train\_tag&#x20;

입력:

* cfgfile: 문자열 포인터, 모델의 구성 파일 경로
* weightfile: 문자열 포인터, 모델 가중치 파일 경로
* clear: 정수, 0 또는 1 값을 가짐. 1이면 이전 학습 상태를 초기화하고 새로운 학습을 시작함.

동작:

* 모델을 cfgfile과 weightfile에서 로드함.
* 학습 이미지 경로를 가져와서 배치 단위로 모델 학습을 진행함.
* 각 배치마다 로스를 계산하고, 가중치를 업데이트함.
* 학습 중간에 모델 가중치를 백업하고, 학습 종료 후 최종 가중치를 저장함.

설명:&#x20;

* 이 함수는 Darknet 라이브러리에서 사용되는 함수로, YOLO와 같은 객체 탐지 알고리즘을 학습할 때 사용됩니다.&#x20;
* 이 함수는 모델 구성 파일(cfgfile)과 모델 가중치 파일(weightfile)을 로드하고, 학습 이미지를 가져와서 모델 학습을 진행합니다.&#x20;
* 이 함수는 각 배치에서 로스를 계산하고, 가중치를 업데이트합니다.
* &#x20;학습 중간에 모델 가중치를 백업하고, 학습 종료 후 최종 가중치를 저장합니다.&#x20;
* 이 함수는 학습을 위해 스레드를 사용하며, 이미지를 비동기적으로 로드합니다.



## test\_tag

```c
void test_tag(char *cfgfile, char *weightfile, char *filename)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    int i = 0;
    char **names = get_labels("data/tags.txt");
    clock_t time;
    int indexes[10];
    char buff[256];
    char *input = buff;
    int size = net->w;
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
        image r = resize_min(im, size);
        resize_network(net, r.w, r.h);
        printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        top_predictions(net, 10, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < 10; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: test\_tag

입력:

* cfgfile (문자열): 네트워크 구성 파일 경로
* weightfile (문자열): 학습된 가중치 파일 경로
* filename (문자열, 선택적): 이미지 파일 경로. 없으면 사용자로부터 입력 받음.

동작:&#x20;

* 주어진 이미지 파일 또는 사용자 입력으로부터 이미지를 로드하고, 네트워크로 예측을 수행하여 예측 결과를 출력함.&#x20;
* 출력된 결과는 예측된 태그와 해당 태그에 대한 확률값으로 구성됨.

설명:&#x20;

* 이 함수는 주어진 네트워크 구성 파일과 학습된 가중치 파일을 사용하여 이미지에 대한 태그 예측을 수행하는 기능을 제공합니다.&#x20;
* 함수는 입력으로 이미지 파일 경로를 받거나, 사용자로부터 경로를 입력 받습니다.&#x20;
* 이미지를 로드하고, 크기를 맞춘 후, 네트워크로 예측을 수행하고, 결과를 출력합니다.&#x20;
* 출력된 결과는 예측된 태그와 해당 태그에 대한 확률값으로 구성됩니다.&#x20;
* 이 함수는 테스트 및 디버깅 용도로 사용됩니다.



## run\_tag

```c
void run_tag(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = find_arg(argc, argv, "-clear");
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")) train_tag(cfg, weights, clear);
    else if(0==strcmp(argv[2], "test")) test_tag(cfg, weights, filename);
}
```

함수 이름: run\_tag

입력:

* argc: int 타입. 명령행에서 전달된 인수의 총 개수
* argv: char \*\* 타입. 명령행에서 전달된 인수를 가리키는 포인터 배열

동작:

* 입력된 인수에 따라 train\_tag 또는 test\_tag 함수를 호출함

설명:

* 만약 argc가 4보다 작으면 사용법을 출력하고 리턴함
* argv\[2]가 "train"이면 train\_tag 함수를 cfg와 weights를 이용하여 호출함
* argv\[2]가 "test"이면 test\_tag 함수를 cfg, weights, filename을 이용하여 호출함
* weights와 filename은 옵션으로 전달할 수 있음 (argc > 4, argc > 5)

