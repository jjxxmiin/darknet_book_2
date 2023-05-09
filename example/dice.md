# dice

```c
#include "darknet.h"

char *dice_labels[] = {"face1","face2","face3","face4","face5","face6"};
```

## train\_dice

```c
void train_dice(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "/home/pjreddie/backup/";
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    int i = *net.seen/imgs;
    char **labels = dice_labels;
    list *plist = get_paths("data/dice/dice.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_old(paths, imgs, plist->size, labels, 6, net.w, net.h);
        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %ld images\n", i, loss, avg_loss, sec(clock()-time), *net.seen);
        free_data(train);
        if((i % 100) == 0) net.learning_rate *= .1;
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, i);
            save_weights(net, buff);
        }
    }
}
```

함수 이름: train\_dice&#x20;

입력:

* cfgfile (char\*): YOLO 모델 구성 파일 경로
* weightfile (char\*): 사전 학습된 모델 가중치 파일 경로

동작:&#x20;

* YOLO 모델을 학습시키는 함수입니다.&#x20;
* 주어진 cfgfile과 weightfile을 기반으로 YOLO 모델을 구성하고, 주어진 이미지와 라벨 데이터를 사용하여 모델을 학습합니다.&#x20;
* 학습 중에는 매 100번째 반복마다 학습률을 감소시키고, 매 100번째 반복마다 현재 모델 가중치를 지정된 경로에 저장합니다.

설명:&#x20;

* 이 함수는 YOLO 모델을 학습시키기 위해 사용됩니다. 함수는 cfgfile을 파싱하여 모델을 구성하고, weightfile이 주어지면 모델 가중치를 불러옵니다.&#x20;
* 학습 데이터는 "data/dice/dice.train.list"에서 읽어옵니다.&#x20;
* 매 반복마다 일부 매개변수를 출력하고, 데이터를 로드하고, 네트워크를 학습시키고, 평균 손실 값을 계산합니다.&#x20;
* 그런 다음 모델 가중치를 주기적으로 백업하고, 반복 횟수가 증가할 때마다 학습률을 조정합니다.

함수에서 사용되는 다른 함수들은 다음과 같습니다.

* basecfg(char \*cfgfile): cfgfile 경로에서 구성 파일 이름의 베이스 이름을 반환합니다.
* parse\_network\_cfg(char \*filename): 구성 파일을 파싱하여 network 구조체를 반환합니다.
* load\_weights(network \*net, char \*filename): 지정된 가중치 파일에서 모델 가중치를 로드합니다.
* get\_paths(char \*filename): 지정된 파일에서 이미지 파일 경로를 가져와 list 구조체에 저장합니다.
* list\_to\_array(list \*l): list 구조체에 저장된 문자열 배열을 반환합니다.
* load\_data\_old(char \*\*paths, int n, int m, char \*\*labels, int k, int w, int h): paths 배열에 지정된 이미지 파일을 로드하고, labels 배열에 지정된 라벨 파일을 로드하여 data 구조체에 저장합니다.
* train\_network(network net, data d): 주어진 모델과 데이터로 모델을 학습시키고, 손실 값을 반환합니다.
* free\_data(data d): data 구조체에서 할당된 메모리를 해제합니다.
* save\_weights(network net, char \*filename): 지정된 파일 이름으로 모델 가중치를 저장합니다.



## validate\_dice

```c
void validate_dice(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    char **labels = dice_labels;
    list *plist = get_paths("data/dice/dice.val.list");

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    data val = load_data_old(paths, m, 0, labels, 6, net.w, net.h);
    float *acc = network_accuracies(net, val, 2);
    printf("Validation Accuracy: %f, %d images\n", acc[0], m);
    free_data(val);
}
```

함수 이름: validate\_dice

입력:

* char \*filename: 구성 파일 경로
* char \*weightfile: 가중치 파일 경로

동작:&#x20;

* 주어진 구성 파일과 가중치 파일을 사용하여 주사위 이미지 분류 네트워크를 검증하는 함수입니다.&#x20;
* 네트워크 정확도와 검증 이미지 수를 출력합니다.

설명:

1. 주어진 구성 파일로부터 네트워크를 구성합니다.
2. 만약 가중치 파일이 주어졌다면 해당 가중치를 로드합니다.
3. 난수 발생기를 초기화합니다.
4. dice\_labels 배열에서 라벨 정보를 가져옵니다.
5. 검증 이미지 경로를 포함하는 리스트를 생성합니다.
6. 리스트를 배열로 변환합니다.
7. 검증 이미지 데이터를 로드합니다.
8. 로드된 데이터를 사용하여 네트워크의 정확도를 계산합니다.
9. 검증 정확도와 검증 이미지 수를 출력합니다.
10. 메모리를 해제합니다.



## test\_dice

```c
void test_dice(char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    int i = 0;
    char **names = dice_labels;
    char buff[256];
    char *input = buff;
    int indexes[6];
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
        image im = load_image_color(input, net.w, net.h);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        top_predictions(net, 6, indexes);
        for(i = 0; i < 6; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: test\_dice&#x20;

입력:

* char \*cfgfile: 모델 구성 파일 경로
* char \*weightfile: 학습된 가중치 파일 경로
* char \*filename (선택적): 테스트할 이미지 파일 경로

동작:

* 주어진 cfgfile과 weightfile로부터 네트워크를 로드하여 테스트 이미지에 대한 예측을 수행하는 함수
* 입력으로 filename이 주어지면 해당 이미지에 대한 예측을 출력하고 종료
* 입력으로 filename이 주어지지 않으면 사용자로부터 이미지 파일 경로를 입력받아 예측을 수행하고, 다음 이미지를 계속해서 입력받음
* 각 클래스에 대한 확률 예측과 함께 예측이 가장 높은 클래스를 출력함

설명:

* test\_dice 함수는 주어진 cfgfile과 weightfile로부터 네트워크를 로드하여 테스트 이미지에 대한 예측을 수행하는 함수입니다.
* 입력으로 filename이 주어지면 해당 이미지에 대한 예측을 출력하고 종료합니다. 입력으로 filename이 주어지지 않으면 사용자로부터 이미지 파일 경로를 입력받아 예측을 수행하고, 다음 이미지를 계속해서 입력받습니다.
* 각 클래스에 대한 확률 예측과 함께 예측이 가장 높은 클래스를 출력합니다. 이를 위해 top\_predictions 함수를 사용하여 예측이 가장 높은 클래스의 인덱스를 계산하고, 해당 클래스의 이름과 예측 확률을 출력합니다.



## run\_dice

```c
void run_dice(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_dice(cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_dice(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_dice(cfg, weights);
}
```

함수 이름: run\_dice

입력:

* int argc: 명령행 인자의 수
* char \*\*argv: 명령행 인자의 배열

동작:

* 명령행에서 입력된 인자에 따라 train\_dice, test\_dice, validate\_dice 함수 중 하나를 실행함
* "train"이 입력된 경우 train\_dice(cfg, weights)를 호출함
* "test"가 입력된 경우 test\_dice(cfg, weights, filename)을 호출함
* "valid"가 입력된 경우 validate\_dice(cfg, weights)를 호출함
* 인자가 부족한 경우 사용 방법(usage)을 출력함

설명:

* cfg: 설정 파일 경로
* weights (optional): 가중치 파일 경로 (선택적)
* filename (optional): 이미지 파일 경로 (선택적)

