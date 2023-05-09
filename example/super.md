# super

```c
#include "darknet.h"
```

## train\_super

```c
void train_super(char *cfgfile, char *weightfile, int clear)
{
    char *train_images = "/data/imagenet/imagenet1k.train.list";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *net = load_network(cfgfile, weightfile, clear);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    int i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.scale = 4;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = SUPER_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
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
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```

함수 이름: train\_super

입력:

* cfgfile: 학습을 위한 네트워크 구성 파일의 경로
* weightfile: 미리 학습된 가중치 파일의 경로
* clear: 이전 가중치 초기화 여부

동작:&#x20;

* 주어진 cfgfile과 weightfile을 사용하여 네트워크를 로드하고, imagenet1k.train.list에서 이미지 경로를 가져온 뒤 이를 사용하여 네트워크를 학습시킨다.&#x20;
* 각 배치마다 가중치를 업데이트하고, 일정 주기마다 가중치를 저장한다.

설명:&#x20;

* 이 함수는 지도 학습을 위해 네트워크를 학습시키는 함수이다.&#x20;
* 이 함수는 학습 데이터로부터 배치를 생성하고, 각 배치를 사용하여 네트워크를 학습시킨다.&#x20;
* 학습 중에는 현재 학습된 배치의 수와 손실 함수의 값을 출력하며, 지정된 주기마다 가중치를 저장한다.&#x20;
* 최종적으로 학습된 가중치는 backup\_directory/base\_final.weights 파일에 저장된다.



## test\_super

```c
void test_super(char *cfgfile, char *weightfile, char *filename)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
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
        resize_network(net, im.w, im.h);
        printf("%d %d\n", im.w, im.h);

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image(net);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");
        show_image(out, "out", 0);

        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: test\_super 입력:

* cfgfile: YOLO 모델의 configuration 파일 경로
* weightfile: 학습된 YOLO 모델의 가중치 파일 경로
* filename (선택적): 테스트할 이미지 파일 경로

동작:

* YOLO 모델을 불러와서 테스트를 수행하는 함수
* 이미지 파일을 입력으로 받아 YOLO 모델을 통해 객체 검출을 수행하고, 결과를 출력함
* 이미지 파일이 주어지지 않으면, 표준 입력으로부터 이미지 파일 경로를 입력받음
* 테스트 결과는 출력 이미지 파일로 저장됨

설명:

* load\_network 함수를 이용하여 cfg 파일과 weight 파일을 이용해 YOLO 모델을 로드함
* set\_batch\_network 함수를 이용하여 배치 크기를 1로 설정함
* load\_image\_color 함수를 이용하여 입력 이미지를 로드함
* resize\_network 함수를 이용하여 모델의 입력 크기를 이미지의 크기로 조정함
* network\_predict 함수를 이용하여 YOLO 모델을 통해 객체 검출을 수행함
* get\_network\_image 함수를 이용하여 검출된 객체를 이미지로 변환함
* save\_image 함수를 이용하여 검출된 객체를 출력 이미지 파일로 저장함
* show\_image 함수를 이용하여 검출된 객체를 화면에 출력함



## run\_super

```c
void run_super(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    int clear = find_arg(argc, argv, "-clear");
    if(0==strcmp(argv[2], "train")) train_super(cfg, weights, clear);
    else if(0==strcmp(argv[2], "test")) test_super(cfg, weights, filename);
    /*
    else if(0==strcmp(argv[2], "valid")) validate_super(cfg, weights);
    */
}
```

함수 이름: run\_super

입력:&#x20;

* argc와 argv, 두 개의 매개변수

동작:&#x20;

* 주어진 인자(argv)를 기반으로 YOLOv3-super 모델을 학습, 테스트 또는 검증하며, 이에 필요한 매개변수(cfg, weights, filename)를 설정한다.

설명:&#x20;

이 함수는 argv에서 인자를 읽어와서 YOLOv3-super 모델을 학습, 테스트 또는 검증하는 역할을 한다. 이 함수는 다음과 같은 동작을 수행한다.

* argv에 전달된 인자의 개수를 검사하고 인자가 충분하지 않으면 사용법을 출력하고 함수를 종료한다.
* cfg, weights 및 filename 매개변수를 설정한다.
* clear 인자를 검사하여 이전에 생성된 모든 캐시와 결과 파일을 지우는지 여부를 결정한다.
* 만약 "train" 인자가 주어지면 train\_super 함수를 호출하여 모델을 학습시킨다.
* 만약 "test" 인자가 주어지면 test\_super 함수를 호출하여 모델을 테스트한다.
* "valid" 인자는 주석 처리되어 있으므로 현재는 사용되지 않는다.
