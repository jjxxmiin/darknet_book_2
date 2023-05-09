# compare

## train\_compare

```c
void train_compare(char *cfgfile, char *weightfile)
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
    list *plist = get_paths("data/compare.train.list");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    printf("%d\n", N);
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = 20;
    args.n = imgs;
    args.m = N;
    args.d = &buffer;
    args.type = COMPARE_DATA;

    load_thread = load_data_in_thread(args);
    int epoch = *net.seen/N;
    int i = 0;
    while(1){
        ++i;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        load_thread = load_data_in_thread(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%.3f: %f, %f avg, %lf seconds, %ld images\n", (float)*net.seen/N, loss, avg_loss, sec(clock()-time), *net.seen);
        free_data(train);
        if(i%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%d_minor_%d.weights",backup_directory,base, epoch, i);
            save_weights(net, buff);
        }
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            i = 0;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
            if(epoch%22 == 0) net.learning_rate *= .1;
        }
    }
    pthread_join(load_thread, 0);
    free_data(buffer);
    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
```

함수 이름: train\_compare

입력:

* char \*cfgfile: 학습을 수행할 YOLO 모델의 구성 파일 경로
* char \*weightfile: 학습을 수행할 YOLO 모델의 가중치 파일 경로

동작:

* YOLO 모델을 이용하여 compare\_data 형식의 데이터를 학습한다.
* 지정된 횟수(epoch)만큼 학습을 반복하며, 학습 중에는 주기적으로 가중치 파일을 저장한다.
* 학습 종료 후, 학습된 YOLO 모델의 가중치를 저장한다.

설명:

* srand(time(0))은 시간에 따라 랜덤 시드를 설정한다.
* basecfg(cfgfile)은 cfgfile에서 파일 이름만 가져온다.
* network net = parse\_network\_cfg(cfgfile)은 cfgfile에서 모델 구성 정보를 파싱하여 네트워크 모델을 생성한다.
* load\_weights(\&net, weightfile)은 weightfile에서 저장된 가중치 정보를 로드하여 모델에 적용한다.
* imgs는 한 번에 읽을 이미지 파일의 개수이다.
* get\_paths("data/compare.train.list")는 학습에 사용할 이미지 파일 경로 리스트를 읽어온다.
* load\_args 구조체를 초기화하고, load\_data\_in\_thread(args)를 호출하여 이미지와 라벨 데이터를 비동기적으로 읽어온다.
* train\_network(net, train)은 모델(net)과 학습 데이터(train)를 이용하여 학습을 수행하고, 학습 손실값을 반환한다.
* save\_weights(net, buff)는 모델(net)의 가중치 정보를 buff 경로에 저장한다.
* net.learning\_rate \*= .1은 학습률(learning\_rate)을 10% 감소시킨다.
* free\_data(train)은 학습 데이터(train)를 해제한다.
* free\_network(net)은 모델(net)을 해제한다.
* free\_ptrs((void\*\*)paths, plist->size)은 paths 배열과 plist 리스트를 해제한다.
* free\_list(plist)는 plist 리스트를 해제한다.
* free(base)는 base 메모리를 해제한다.



## validate\_compare

```c
void validate_compare(char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *plist = get_paths("data/compare.val.list");
    //list *plist = get_paths("data/compare.val.old");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size/2;
    free_list(plist);

    clock_t time;
    int correct = 0;
    int total = 0;
    int splits = 10;
    int num = (i+1)*N/splits - i*N/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = 20;
    args.n = num;
    args.m = 0;
    args.d = &buffer;
    args.type = COMPARE_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*N/splits - i*N/splits;
        char **part = paths+(i*N/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);
        int j,k;
        for(j = 0; j < val.y.rows; ++j){
            for(k = 0; k < 20; ++k){
                if(val.y.vals[j][k*2] != val.y.vals[j][k*2+1]){
                    ++total;
                    if((val.y.vals[j][k*2] < val.y.vals[j][k*2+1]) == (pred.vals[j][k*2] < pred.vals[j][k*2+1])){
                        ++correct;
                    }
                }
            }
        }
        free_matrix(pred);
        printf("%d: Acc: %f, %lf seconds, %d images\n", i, (float)correct/total, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}
```

함수 이름: validate\_compare

입력:

* char\* filename : 네트워크 설정 파일 경로
* char\* weightfile : 미리 학습된 가중치 파일 경로

동작:

* 네트워크 설정 파일과 미리 학습된 가중치 파일을 이용하여 네트워크를 불러온다.
* 검증 데이터셋 경로를 읽어온다.
* 검증 데이터셋을 10개로 분할하여 각각에 대해 다음을 수행한다.
  * 검증 데이터를 읽어온다.
  * 네트워크를 이용하여 예측을 수행한다.
  * 예측값과 실제값을 비교하여 정확도를 계산한다.
* 분할된 각 검증 데이터셋에 대한 정확도와 소요 시간을 출력한다.

설명:&#x20;

* 이 함수는 이미 학습된 네트워크를 검증하는 역할을 한다. 검증 데이터셋은 10개로 분할하여 각각에 대해 검증을 수행하고, 최종적으로 분할된 전체 검증 데이터셋에 대한 정확도와 소요 시간을 출력한다.



## sortable\_bbox

```c
typedef struct {
    network net;
    char *filename;
    int class;
    int classes;
    float elo;
    float *elos;
} sortable_bbox;
```

* `network net`: YOLO 네트워크에 대한 정보가 담긴 구조체로, YOLO 객체 검출 모델에 대한 정보를 저장합니다.
* `char *filename`: 객체 검출이 수행된 이미지 파일의 경로를 저장합니다.
* `int class`: 객체가 속한 클래스의 인덱스를 저장합니다. 예를 들어, 클래스가 80개인 COCO 데이터셋에서는 0부터 79까지의 정수값으로 클래스를 표현합니다.
* `int classes`: 객체 검출 모델이 인식할 수 있는 클래스의 수를 저장합니다.
* `float elo`: 현재 객체 검출 결과에 대한 Elo rating 값을 저장합니다. Elo rating은 체스나 게임 등에서 플레이어의 능력치를 표현하기 위한 수치입니다.
* `float *elos`: 각 클래스에 대한 Elo rating 값을 저장합니다. classes와 같은 크기를 가지는 배열입니다.

이 구조체는 객체 검출 결과를 저장하고, 이를 정렬하기 위한 용도로 사용됩니다. 각 객체는 클래스 인덱스와 Elo rating 값으로 구성된 쌍으로 나타내어지며, Elo rating 값이 높은 객체일수록 상위에 위치하게 됩니다.



## elo\_comparator

```c
int total_compares = 0;
int current_class = 0;

int elo_comparator(const void*a, const void *b)
{
    sortable_bbox box1 = *(sortable_bbox*)a;
    sortable_bbox box2 = *(sortable_bbox*)b;
    if(box1.elos[current_class] == box2.elos[current_class]) return 0;
    if(box1.elos[current_class] >  box2.elos[current_class]) return -1;
    return 1;
}
```

함수 이름: elo\_comparator

입력:&#x20;

* void 포인터 형태로 정렬할 sortable\_bbox 구조체의 주소값인 a와 b

동작:&#x20;

* elo\_comparator 함수는 정렬 알고리즘에서 사용되는 비교 함수이다.
* 두 개의 sortable\_bbox 구조체를 비교하여 정렬 순서를 결정한다.&#x20;
* 현재 클래스(current\_class)에서의 elos 값을 비교하여 내림차순으로 정렬한다.

설명:

* total\_compares: 전체 비교 수를 나타내는 변수
* current\_class: 현재 클래스의 인덱스를 나타내는 변수
* elo\_comparator 함수는 qsort() 함수에서 사용될 비교 함수이다. qsort()는 일반적으로 C/C++에서 사용되는 정렬 함수로, 오름차순 또는 내림차순으로 배열을 정렬할 수 있다.
* sortable\_bbox 구조체는 bbox.c 파일에서 사용되며, 네트워크, 파일 이름, 클래스, 클래스 수, elo 값 등을 저장한다.
* elo\_comparator 함수는 두 개의 sortable\_bbox 구조체를 받아 각 구조체의 current\_class 인덱스에 해당하는 elos 값을 비교한다. elos 값이 높은 것부터 내림차순으로 정렬하며, 만약 elos 값이 같은 경우에는 순서를 바꾸지 않는다.
* 이 함수는 compare\_weights() 함수에서 호출되며, 정렬된 결과는 sorted\_boxes에 저장된다.



## bbox\_comparator

```c
int bbox_comparator(const void *a, const void *b)
{
    ++total_compares;
    sortable_bbox box1 = *(sortable_bbox*)a;
    sortable_bbox box2 = *(sortable_bbox*)b;
    network net = box1.net;
    int class   = box1.class;

    image im1 = load_image_color(box1.filename, net.w, net.h);
    image im2 = load_image_color(box2.filename, net.w, net.h);
    float *X  = calloc(net.w*net.h*net.c, sizeof(float));
    memcpy(X,                   im1.data, im1.w*im1.h*im1.c*sizeof(float));
    memcpy(X+im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c*sizeof(float));
    float *predictions = network_predict(net, X);

    free_image(im1);
    free_image(im2);
    free(X);
    if (predictions[class*2] > predictions[class*2+1]){
        return 1;
    }
    return -1;
}
```

함수 이름: bbox\_comparator

입력:

* const void 포인터 a: 비교하고자 하는 첫 번째 sortable\_bbox 구조체의 포인터
* const void 포인터 b: 비교하고자 하는 두 번째 sortable\_bbox 구조체의 포인터

동작:

* total\_compares 값을 1 증가시킴
* a와 b를 sortable\_bbox 구조체로 변환하여 box1, box2에 저장
* box1에 저장된 network와 class 값을 사용하여 이미지를 로드하고, X 배열에 이미지 데이터를 복사
* box2에 저장된 network와 class 값을 사용하여 이미지를 로드하고, X 배열에 이미지 데이터를 이어붙임
* network\_predict 함수를 사용하여 X 배열의 예측 값을 계산하여 predictions에 저장
* 사용한 이미지와 X 배열의 메모리를 해제함
* class에 해당하는 예측 값 비교 결과를 반환함

설명:&#x20;

* 이 함수는 두 개의 sortable\_bbox 구조체를 비교하여 정렬하기 위해 qsort 함수에서 사용됩니다.&#x20;
* 두 개의 구조체에서 network와 class 값은 같으므로, 이를 사용하여 두 개의 이미지를 로드하고, 예측 값을 계산하여 비교합니다.&#x20;
* 반환 값은 예측 값이 더 큰 경우 1, 그렇지 않은 경우 -1입니다. 이 함수가 호출될 때마다 total\_compares 값을 1 씩 증가시켜, 이 함수가 총 몇 번 호출되었는지를 추적합니다.



## bbox\_update

```c
void bbox_update(sortable_bbox *a, sortable_bbox *b, int class, int result)
{
    int k = 32;
    float EA = 1./(1+pow(10, (b->elos[class] - a->elos[class])/400.));
    float EB = 1./(1+pow(10, (a->elos[class] - b->elos[class])/400.));
    float SA = result ? 1 : 0;
    float SB = result ? 0 : 1;
    a->elos[class] += k*(SA - EA);
    b->elos[class] += k*(SB - EB);
}
```

함수 이름: bbox\_update&#x20;

입력:&#x20;

* sortable\_bbox 타입의 두 개의 포인터 a와 b, int 타입의 class와 result&#x20;

동작:&#x20;

* Elo 레이팅 시스템을 사용하여 a와 b의 레이팅을 갱신한다. a와 b는 각각 class와 관련된 Elo 레이팅을 가지고 있으며, result는 a와 b 중 어느 쪽이 승리했는지를 나타낸다. 각각의 레이팅은 SA와 SB로 표현된다.&#x20;
* 이전 레이팅에 대한 예상 승률 EA와 EB는 현재 레이팅과 이전 레이팅 사이의 차이를 통해 계산된다.&#x20;
* k는 승리나 패배에 대한 가중치를 조절하는 상수이다.&#x20;

설명:&#x20;

* Elo 레이팅 시스템은 체스 선수들의 레이팅을 갱신하기 위해 고안된 시스템으로, 이제는 다양한 분야에서 사용되고 있다.&#x20;
* 이 함수는 이 시스템을 사용하여 a와 b의 레이팅을 갱신한다.&#x20;
* 승리한 쪽의 레이팅은 상대방에 비해 더 큰 증가를 하고, 패배한 쪽은 상대방에 비해 더 큰 감소를 한다.&#x20;
* 이전 레이팅과 예상 승률을 비교하여 새로운 레이팅을 계산한다.



## bbox\_fight

```c
void bbox_fight(network net, sortable_bbox *a, sortable_bbox *b, int classes, int class)
{
    image im1 = load_image_color(a->filename, net.w, net.h);
    image im2 = load_image_color(b->filename, net.w, net.h);
    float *X  = calloc(net.w*net.h*net.c, sizeof(float));
    memcpy(X,                   im1.data, im1.w*im1.h*im1.c*sizeof(float));
    memcpy(X+im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c*sizeof(float));
    float *predictions = network_predict(net, X);
    ++total_compares;

    int i;
    for(i = 0; i < classes; ++i){
        if(class < 0 || class == i){
            int result = predictions[i*2] > predictions[i*2+1];
            bbox_update(a, b, i, result);
        }
    }

    free_image(im1);
    free_image(im2);
    free(X);
}
```

함수 이름: bbox\_fight&#x20;

입력:

* network net: YOLO 모델
* sortable\_bbox \*a: 비교 대상 A
* sortable\_bbox \*b: 비교 대상 B
* int classes: 클래스 수
* int class: 비교할 클래스 인덱스. 음수이면 모든 클래스 비교

동작:

* A와 B를 비교하여 예측값 계산
* 각 클래스별로, 해당 클래스가 선택된 경우 또는 음수이면 모든 클래스에 대해, A와 B의 elo를 갱신

설명:

* 두 이미지 a->filename과 b->filename를 YOLO 모델로 예측한 결과(predictions)를 받음
* 클래스별로 predictions에서 이긴 이미지를 결정하여 bbox\_update 함수를 호출하여 elo 값 갱신
* 갱신된 값은 sortable\_bbox 구조체 내 elos 배열에 저장



## SortMaster3000

```c
void SortMaster3000(char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    set_batch_network(&net, 1);

    list *plist = get_paths("data/compare.sort.list");
    //list *plist = get_paths("data/compare.val.old");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    free_list(plist);
    sortable_bbox *boxes = calloc(N, sizeof(sortable_bbox));
    printf("Sorting %d boxes...\n", N);
    for(i = 0; i < N; ++i){
        boxes[i].filename = paths[i];
        boxes[i].net = net;
        boxes[i].class = 7;
        boxes[i].elo = 1500;
    }
    clock_t time=clock();
    qsort(boxes, N, sizeof(sortable_bbox), bbox_comparator);
    for(i = 0; i < N; ++i){
        printf("%s\n", boxes[i].filename);
    }
    printf("Sorted in %d compares, %f secs\n", total_compares, sec(clock()-time));
}
```

함수 이름: SortMaster3000

입력:

* filename: 정렬할 네트워크 구성 파일 경로
* weightfile: 가중치 파일 경로

동작:

* filename과 weightfile로부터 네트워크를 파싱하고, 가중치를 불러온다.
* 네트워크를 1개의 배치로 설정하고, 시드를 초기화한다.
* "data/compare.sort.list"에서 파일 경로 리스트를 가져온다.
* 가져온 리스트의 크기만큼 sortable\_bbox 구조체를 생성하고, 파일 경로, 네트워크, 클래스, elo 등의 정보를 저장한다.
* qsort를 사용하여 boxes를 정렬한다.
* 정렬된 boxes의 파일 경로를 출력한다.
* 총 비교 횟수와 걸린 시간을 출력한다.

설명:&#x20;

* 주어진 파일 경로에 있는 네트워크 구성 파일과 가중치 파일을 사용하여 네트워크를 파싱하고, "data/compare.sort.list"에서 파일 경로 리스트를 가져온 후, 해당 경로의 파일들을 sortable\_bbox 구조체에 저장합니다.&#x20;
* 그리고 qsort를 사용하여 이를 정렬한 후, 파일 경로를 출력하며, 총 비교 횟수와 걸린 시간을 출력합니다.



## BattleRoyaleWithCheese

```c
void BattleRoyaleWithCheese(char *filename, char *weightfile)
{
    int classes = 20;
    int i,j;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    set_batch_network(&net, 1);

    list *plist = get_paths("data/compare.sort.list");
    //list *plist = get_paths("data/compare.small.list");
    //list *plist = get_paths("data/compare.cat.list");
    //list *plist = get_paths("data/compare.val.old");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    int total = N;
    free_list(plist);
    sortable_bbox *boxes = calloc(N, sizeof(sortable_bbox));
    printf("Battling %d boxes...\n", N);
    for(i = 0; i < N; ++i){
        boxes[i].filename = paths[i];
        boxes[i].net = net;
        boxes[i].classes = classes;
        boxes[i].elos = calloc(classes, sizeof(float));;
        for(j = 0; j < classes; ++j){
            boxes[i].elos[j] = 1500;
        }
    }
    int round;
    clock_t time=clock();
    for(round = 1; round <= 4; ++round){
        clock_t round_time=clock();
        printf("Round: %d\n", round);
        shuffle(boxes, N, sizeof(sortable_bbox));
        for(i = 0; i < N/2; ++i){
            bbox_fight(net, boxes+i*2, boxes+i*2+1, classes, -1);
        }
        printf("Round: %f secs, %d remaining\n", sec(clock()-round_time), N);
    }

    int class;

    for (class = 0; class < classes; ++class){

        N = total;
        current_class = class;
        qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
        N /= 2;

        for(round = 1; round <= 100; ++round){
            clock_t round_time=clock();
            printf("Round: %d\n", round);

            sorta_shuffle(boxes, N, sizeof(sortable_bbox), 10);
            for(i = 0; i < N/2; ++i){
                bbox_fight(net, boxes+i*2, boxes+i*2+1, classes, class);
            }
            qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
            if(round <= 20) N = (N*9/10)/2*2;

            printf("Round: %f secs, %d remaining\n", sec(clock()-round_time), N);
        }
        char buff[256];
        sprintf(buff, "results/battle_%d.log", class);
        FILE *outfp = fopen(buff, "w");
        for(i = 0; i < N; ++i){
            fprintf(outfp, "%s %f\n", boxes[i].filename, boxes[i].elos[class]);
        }
        fclose(outfp);
    }
    printf("Tournament in %d compares, %f secs\n", total_compares, sec(clock()-time));
}
```

함수 이름: BattleRoyaleWithCheese&#x20;

입력:&#x20;

* char\* filename: 구성 파일 경로
* char\* weightfile: 사전 학습된 모델 가중치 파일 경로&#x20;

동작:&#x20;

* 지정된 구성 파일을 파싱하여 네트워크를 만들고, 사전 학습된 모델 가중치를 로드한 후, 지정된 목록 파일에서 비교할 이미지 경로 목록을 가져옵니다.&#x20;
* 이미지 경로 목록에서 각 이미지에 대한 Elo Rating을 계산합니다. Elo Rating은 두 객체 간의 상대적인 승률을 나타내는 지수입니다.&#x20;
* Elo Rating을 계산하면 높은 순서대로 이미지가 정렬됩니다.&#x20;
* 다음으로 Elo Rating이 가장 높은 이미지와 두 번째로 높은 이미지를 뽑아서 싸웁니다.&#x20;
* Elo Rating이 가장 높은 이미지는 Elo Rating이 두 번째로 높은 이미지보다 이길 가능성이 높습니다.&#x20;
* 이 과정을 여러 번 반복합니다. 마지막으로 Elo Rating이 가장 높은 이미지를 출력합니다.&#x20;

설명:&#x20;

* Elo Rating이라는 지수를 사용하여 이미지를 정렬하고, 각 이미지가 얼마나 잘 분류되는지 측정합니다.&#x20;
* 이 프로그램은 이미지 분류 문제를 해결하는 데 매우 효과적입니다.



## run\_compare

```c
void run_compare(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    //char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_compare(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_compare(cfg, weights);
    else if(0==strcmp(argv[2], "sort")) SortMaster3000(cfg, weights);
    else if(0==strcmp(argv[2], "battle")) BattleRoyaleWithCheese(cfg, weights);
    /*
       else if(0==strcmp(argv[2], "train")) train_coco(cfg, weights);
       else if(0==strcmp(argv[2], "extract")) extract_boxes(cfg, weights);
       else if(0==strcmp(argv[2], "valid")) validate_recall(cfg, weights);
     */
}
```

함수 이름: run\_compare

입력:

* argc: 실행 인자의 개수
* argv: 실행 인자들을 저장하는 배열

동작:

* 실행 인자 개수가 4개 미만이면, 사용법을 출력하고 함수 종료
* 실행 인자 중에서 cfg와 weights를 설정
* 실행 인자 중에서 두 번째로 들어온 문자열 값에 따라 train, valid, sort, battle 중 하나의 함수를 실행
* 주석 처리된 코드는 실행하지 않음

설명:

* 이 함수는 Darknet의 compare 학습을 위해 사용됨
* 실행 인자로 train, valid, sort, battle 중 하나와 cfg 파일 경로, weights 파일 경로를 받음
* 각각의 인자에 따라 해당하는 함수를 실행하며, weights 파일이 없으면 0으로 설정됨



