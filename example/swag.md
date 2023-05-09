# swag

```c
#include "darknet.h"
#include <sys/time.h>
```

## train\_swag

```c
void train_swag(char *cfgfile, char *weightfile)
{
    char *train_images = "data/voc.0712.trainval";
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
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || i == 600){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```

함수 이름: train\_swag

입력:

* cfgfile: YOLO 모델의 설정 파일 경로
* weightfile: 미리 학습된 YOLO 모델의 가중치 파일 경로

동작:

* YOLO 모델을 학습시키는 함수입니다.
* VOC 데이터셋을 사용하며, 데이터셋 경로는 "data/voc.0712.trainval"로 고정되어 있습니다.
* 학습 중 가중치 파일을 주기적으로 저장합니다.
* 함수가 종료될 때 최종 학습된 가중치 파일을 저장합니다.

설명:

* YOLO 모델을 학습시키기 위해 필요한 경로와 설정 값을 입력으로 받습니다.
* 모델 설정 파일(cfgfile)을 파싱하여 YOLO 모델(network)을 생성합니다.
* 만약 미리 학습된 가중치 파일(weightfile)이 주어졌다면 해당 가중치를 모델에 로드합니다.
* VOC 데이터셋 경로를 상수 값으로 고정합니다.
* 모델의 배치 크기와 서브디비전 수를 기반으로 이미지 수(imgs)를 계산합니다.
* 현재 학습된 배치 수(i)를 계산합니다.
* YOLO 모델의 마지막 레이어(l)에서 사이드(side)와 클래스 수(classes), 랜덤 변형 정도(jitter) 등을 가져옵니다.
* VOC 데이터셋에서 이미지 경로 리스트(plist)를 가져온 뒤, 리스트를 문자열 배열(paths)로 변환합니다.
* YOLO 모델의 입력 크기, 이미지 경로, 클래스 수 등을 설정하여 데이터를 로드하는 데 필요한 매개 변수(args)를 설정합니다.
* 데이터 로드를 병렬로 처리하기 위해 스레드를 사용합니다.
* 주어진 최대 배치 수(net.max\_batches)에 도달할 때까지 모델을 학습시킵니다.
* 데이터를 로드하고, 모델을 학습시키며, 학습된 가중치를 주기적으로 저장합니다.
* 학습 종료 시 최종 학습된 가중치를 파일로 저장합니다.



## run\_swag

```c
void run_swag(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_swag(cfg, weights);
}
```

함수 이름: run\_swag&#x20;

입력:

* argc: int, 메인 함수로부터 전달받은 인자의 개수
* argv: char \*\*, 메인 함수로부터 전달받은 인자들의 배열

동작:

* 전달받은 인자들을 이용하여 swag 모델을 학습하는 train\_swag 함수를 호출한다.
* 인자의 개수가 4보다 작으면 사용 방법을 출력하고 함수를 종료한다.

설명:

* swag 모델을 학습하는 함수를 호출하는 함수이다.
* 인자로는 argv\[2]에 "train"을 전달해야 한다.
* argv\[3]에는 모델 설정 파일(.cfg)의 경로를 전달해야 한다.
* argv\[4]에는 모델의 가중치 파일(.weights)의 경로를 전달할 수 있다. 가중치 파일을 전달하지 않으면 학습을 처음부터 시작한다.

