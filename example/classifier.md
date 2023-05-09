# classifier

```c
#include "darknet.h"

#include <sys/time.h>
#include <assert.h>
```

## get\_regression\_values

```c
float *get_regression_values(char **labels, int n)
{
    float *v = calloc(n, sizeof(float));
    int i;
    for(i = 0; i < n; ++i){
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p+1);
    }
    return v;
}
```

함수 이름: get\_regression\_values

입력:&#x20;

* char \*\*labels (문자열 배열 포인터), int n (라벨 개수)

동작:&#x20;

* 라벨 배열에서 숫자 값만 추출하여 float 형식으로 반환하는 함수입니다.

설명:&#x20;

* 라벨 배열에서 숫자 값을 추출하는 과정에서 문자열 처리를 수행합니다.&#x20;
* 라벨의 값은 숫자와 공백으로 이루어져 있으며, 이 함수는 문자열에서 공백을 찾아서 해당 위치부터 숫자 값으로 파싱합니다.&#x20;
* 파싱된 값은 float 형식으로 배열에 저장되고, 최종적으로 배열 포인터가 반환됩니다.



## train\_classifier

```c
void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
            printf("Resizing\n");
            int dim = (rand() % 11 + 4) * 32;
            //if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
            printf("%d %d\n", args.min, args.max);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
```

함수 이름: train\_classifier&#x20;

입력:

* datacfg: char 포인터, 데이터 설정 파일 경로
* cfgfile: char 포인터, 모델 설정 파일 경로
* weightfile: char 포인터, 모델 가중치 파일 경로
* gpus: int 포인터, 사용할 GPU 번호 배열
* ngpus: int, 사용할 GPU 개수
* clear: int, 모델을 clear할지 여부 (0이면 clear 안 함, 1이면 clear)

동작:&#x20;

* 주어진 데이터와 모델 설정으로 분류기를 학습시키는 함수입니다. 학습된 모델 가중치는 지정된 경로에 저장됩니다.

설명:

* 함수는 void를 반환합니다.
* 함수 내부에서는 여러 개의 지역 변수들과 포인터들을 선언하고 초기화합니다.
* 함수의 실행 중간에는 모델의 input 이미지를 resize하는 작업이 이루어집니다.
* 학습 과정에서는 지정된 데이터셋을 사용하여 모델을 학습시키며, 매 iteration마다 loss 값을 계산합니다.
* 매 1000번 iteration마다 학습된 모델 가중치를 저장합니다.
* 모든 iteration이 완료되면 학습된 모델 가중치를 저장합니다.
* 함수 실행 중간에는 많은 print 문이 존재하여, 학습 과정에서 일어나는 여러 일들을 추적하기 쉽게 합니다.



## validate\_classifier\_crop

```c
void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network *net = load_network(filename, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}
```

함수 이름: validate\_classifier\_crop&#x20;

입력:

* datacfg: 데이터 구성 파일 경로를 나타내는 문자열 포인터
* filename: 학습된 네트워크 모델 파일 경로를 나타내는 문자열 포인터
* weightfile: 학습된 네트워크 모델 가중치 파일 경로를 나타내는 문자열 포인터

동작:&#x20;

* 주어진 학습된 네트워크 모델을 사용하여 이미지 분류기를 검증하는 함수입니다. 함수는 주어진 datacfg 파일을 읽어 데이터 구성 정보를 가져옵니다.&#x20;
* 라벨 리스트 파일 경로, 검증 데이터 리스트 파일 경로, 클래스 수, topk 등을 설정합니다. 검증 데이터를 여러 개의 미니 배치로 나누어서 검증을 수행하며, 각 미니 배치별로 정확도를 측정하고 평균 정확도와 평균 topk를 계산합니다.

설명:&#x20;

* 이 함수는 YOLO (You Only Look Once) 딥러닝 알고리즘의 구현체인 Darknet에서 사용되는 함수입니다. Darknet은 이미지 분류, 객체 검출 등의 작업을 수행하는 딥러닝 프레임워크입니다.
* 이 함수는 검증 데이터를 이용하여 학습된 네트워크 모델의 정확도를 측정합니다. datacfg는 데이터 구성 파일의 경로를, filename은 학습된 네트워크 모델 파일의 경로를, weightfile은 학습된 네트워크 모델 가중치 파일의 경로를 나타냅니다. 이 함수에서는 해당 경로에 있는 모델과 가중치를 불러와서 사용합니다.
* 함수는 검증에 필요한 데이터 구성 정보를 읽어들입니다. 이 정보는 options라는 리스트에 저장되며, 여기에는 라벨 리스트 파일 경로, 검증 데이터 리스트 파일 경로, 클래스 수, topk 등이 저장됩니다.
* 검증 데이터는 여러 개의 미니 배치로 나누어서 검증을 수행합니다. splits는 미니 배치의 수를 나타내며, plist 리스트에서 검증 데이터의 경로를 가져와서 paths 배열에 저장합니다. 각 미니 배치별로 정확도를 측정하고 평균 정확도와 평균 topk를 계산합니다.
* 검증 데이터를 불러들일 때는 load\_data\_in\_thread 함수를 사용합니다. 이 함수는 pthread 라이브러리를 이용해서 멀티스레딩으로 데이터를 불러옵니다. 검증 데이터를 불러들인 후에는 free\_data 함수를 사용해서 메모리를 해제합니다.



## validate\_classifier\_10

```c
void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int w = net->w;
        int h = net->h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = calloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}
```

함수 이름: validate\_classifier\_10

입력:

* datacfg (문자열): 데이터 구성 파일 경로
* filename (문자열): 네트워크 파일 경로
* weightfile (문자열): 가중치 파일 경로

동작:&#x20;

* 주어진 네트워크와 가중치를 사용하여 데이터셋의 정확도를 검증하는 함수입니다.&#x20;
* 검증에 사용할 데이터셋은 datacfg 파일에서 지정하며, 라벨 목록, 클래스 수, top-k 값을 설정할 수 있습니다.&#x20;
* 입력 이미지를 여러 방향으로 자르고 뒤집어서 예측을 수행하고, top-k 정확도와 top-1 정확도를 계산하여 출력합니다.

설명:

* char \*\*labels: 라벨 목록을 저장하는 문자열 배열
* list \*plist: 이미지 파일 경로 목록을 저장하는 링크드 리스트
* char \*\*paths: 이미지 파일 경로를 저장하는 문자열 배열
* int m: 검증에 사용할 이미지 파일의 개수
* int classes: 클래스 수
* int topk: top-k 값
* int \*indexes: top-k 예측 결과의 인덱스를 저장하는 정수형 배열
* image im: 입력 이미지를 저장하는 구조체
* image images\[10]: 입력 이미지를 자른 이미지들을 저장하는 구조체 배열
* float \*pred: 예측 결과를 저장하는 실수형 배열
* float \*p: 네트워크 예측 결과를 저장하는 실수형 배열

함수는 각 이미지에 대해 다음을 수행합니다:

1. 이미지 파일에서 입력 이미지를 로드하고, 여러 방향으로 자른 이미지를 생성합니다.
2. 생성된 이미지들에 대해 예측을 수행하고, 예측 결과를 누적합니다.
3. 예측 결과에서 top-k 예측 결과를 계산하고, top-k 정확도와 top-1 정확도를 계산합니다.
4. 메모리를 해제합니다.

함수는 검증 결과를 출력합니다.



## validate\_classifier\_full

```c
void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    int size = net->w;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(net, resized.w, resized.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, resized.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}
```

함수 이름: validate\_classifier\_full&#x20;

입력:

* datacfg: char 포인터, 데이터 구성 파일 경로
* filename: char 포인터, 네트워크 구조 파일 경로
* weightfile: char 포인터, 네트워크 가중치 파일 경로

동작:

* 지정된 경로에서 네트워크를 로드하고, 이를 이용하여 입력 이미지들을 분류함
* 분류 결과의 정확도를 출력함

설명:

* 지정된 경로에서 네트워크 구조와 가중치를 로드하여 네트워크를 생성함
* 배치 크기를 1로 설정함
* 시드 값을 현재 시간으로 설정하여 난수 생성기를 초기화함
* 데이터 구성 파일에서 클래스 수, 라벨 파일 경로, 검증 데이터 파일 경로, top-k 값을 읽어들임
* 라벨 파일에서 클래스 이름을 가져옴
* 검증 데이터 파일에서 이미지 경로 리스트를 가져옴
* 이미지 경로 리스트를 배열로 변환하고, 배열의 크기를 변수 m에 저장함
* top-k 값만큼의 인덱스를 저장할 배열을 할당함
* 각 이미지에 대해 다음을 수행함:
  * 이미지 파일을 로드하고, 지정된 크기로 크기를 조절함
  * 조절된 이미지를 이용하여 네트워크를 통과시키고, 결과를 예측함
  * 예측 결과를 이용하여 분류 정확도를 계산함
  * 계산된 정확도를 출력함



## validate\_classifier\_single

```c
void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        //grayscale_image_3c(crop);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}
```

함수 이름: validate\_classifier\_single

입력:

* char \*datacfg: 데이터 설정 파일 경로
* char \*filename: 네트워크 구조 설정 파일 경로
* char \*weightfile: 네트워크 가중치 파일 경로

동작:

* 지정된 경로에서 이미지를 로드하여 네트워크를 통해 분류하고 정확도를 평가합니다.
* 분류 결과와 각 이미지의 정확도(top 1, top k)를 출력합니다.

설명:

* 지정된 데이터 설정 파일(datacfg)을 읽어들여 옵션들을 가져옵니다.
* 네트워크 구조 설정 파일(filename)과 가중치 파일(weightfile)을 사용하여 네트워크를 로드합니다.
* 한 번에 하나의 이미지(batch\_size=1)를 처리하도록 배치 크기를 설정합니다.
* 지정된 경로에서 검증 데이터셋 리스트를 읽어들여 각 이미지 경로를 가져옵니다.
* 각 이미지를 로드하여 네트워크를 통해 분류하고 정확도를 평가합니다.
* 분류 결과와 각 이미지의 정확도(top 1, top k)를 출력합니다.
* 각 이미지에서 추론된 결과(pred)와 실제 레이블(class)을 함께 출력합니다.



## validate\_classifier\_multi

```c
void validate_classifier_multi(char *datacfg, char *cfg, char *weights)
{
    int i, j;
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    //int scales[] = {224, 288, 320, 352, 384};
    int scales[] = {224, 256, 288, 320};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_max(im, scales[j]);
            resize_network(net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}
```

함수 이름: validate\_classifier\_multi

입력:

* datacfg: char 포인터, 데이터 설정 파일 경로
* cfg: char 포인터, 모델 설정 파일 경로
* weights: char 포인터, 모델 가중치 파일 경로

동작:&#x20;

* 주어진 데이터 설정 파일, 모델 설정 파일, 모델 가중치 파일을 이용하여 모델을 로드하고, 검증 데이터셋을 이용하여 모델의 정확도를 검증하는 함수입니다.&#x20;
* 검증 결과인 top-1 정확도와 top-k 정확도를 출력합니다.

설명:&#x20;

* 이 함수는 Darknet 프레임워크에서 제공하는 함수로, 주어진 데이터 설정 파일, 모델 설정 파일, 모델 가중치 파일을 이용하여 모델을 로드합니다. 검증 데이터셋을 이용하여 모델의 정확도를 검증하고, top-1 정확도와 top-k 정확도를 출력합니다.
* 함수가 받는 입력으로는 데이터 설정 파일 경로, 모델 설정 파일 경로, 모델 가중치 파일 경로가 있습니다. 이 함수는 데이터 설정 파일에서 다음과 같은 정보를 읽어옵니다.
  * labels: 클래스 레이블 파일 경로
  * valid: 검증 데이터셋 파일 경로
  * classes: 클래스 수
  * top: top-k 정확도 계산에 사용할 k 값
* 이 함수는 각 이미지에 대해 다음과 같은 동작을 수행합니다.
  * 클래스 레이블 파일에서 클래스 레이블을 읽어옵니다.
  * 검증 데이터셋에서 이미지 경로를 읽어옵니다.
  * 이미지를 로드하고, 다양한 크기로 resize합니다.
  * resize된 이미지를 이용하여 모델의 예측값을 구합니다.
  * 예측값을 이용하여 top-k 정확도를 계산합니다.
  * top-1 정확도를 계산합니다.
* 마지막으로, 각 이미지의 top-1 정확도와 top-k 정확도를 평균하여 출력합니다.



## try\_classifier

```c
void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
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
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net->layers[layer_num];
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }
#ifdef GPU
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }
        /*

           printf("\n\nWeights\n");
           for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
           printf("%f\n", l.filters[i]);
           }

           printf("\n\nBiases\n");
           for(i = 0; i < l.n; ++i){
           printf("%f\n", l.biases[i]);
           }
         */

        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: try\_classifier&#x20;

입력:

* datacfg: char 포인터 타입, 데이터 구성 파일 경로
* cfgfile: char 포인터 타입, 모델 구성 파일 경로
* weightfile: char 포인터 타입, 모델 가중치 파일 경로
* filename: char 포인터 타입, 이미지 파일 경로 (선택적 입력)
* layer\_num: int 타입, 출력할 레이어 번호

동작:

* 주어진 모델 구성 파일과 가중치 파일을 사용하여 네트워크를 로드한다.
* 입력 이미지를 받아들인다 (filename으로 지정된 이미지 또는 stdin에서 입력)
* 입력 이미지를 전처리한다 (크기 조정, 자르기, 정규화)
* 전처리된 이미지를 입력으로 사용하여 네트워크를 실행한다.
* 지정된 출력 레이어의 출력 및 특정 가중치와 바이어스 값을 출력한다.
* 각 클래스에 대한 예측 확률을 출력한다.

설명:

* try\_classifier 함수는 주어진 모델로 이미지 분류기를 시도하는 함수이다.
* 이 함수는 Darknet 프레임워크를 사용하여 작성되었으며, C 언어로 작성되었다.
* 함수의 매개 변수로는 데이터 구성 파일 경로, 모델 구성 파일 경로, 모델 가중치 파일 경로, 이미지 파일 경로 (선택 사항) 및 출력 레이어 번호가 있다.
* 함수는 입력 이미지를 받아들이기 위해 stdin에서 이미지 경로를 입력하거나 filename을 사용하여 이미지 파일 경로를 직접 지정할 수 있다.
* 함수는 입력 이미지를 전처리하고 지정된 출력 레이어의 출력 값을 출력한다.
* 함수는 또한 각 클래스에 대한 예측 확률을 출력한다.



## predict\_classifier

```c
void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
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
        image r = letterbox_image(im, net->w, net->h);
        //image r = resize_min(im, 320);
        //printf("%d %d\n", r.w, r.h);
        //resize_network(net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: predict\_classifier&#x20;

입력:

* datacfg: 데이터 설정 파일의 경로를 나타내는 문자열 포인터
* cfgfile: 네트워크 구조 설정 파일의 경로를 나타내는 문자열 포인터
* weightfile: 학습된 가중치 파일의 경로를 나타내는 문자열 포인터
* filename: 이미지 파일 경로를 나타내는 문자열 포인터, 없으면 NULL
* top: 분류 결과 중 상위 몇 개의 예측 결과를 출력할지를 나타내는 정수

동작:&#x20;

* 입력된 이미지 파일을 분류하여 예측 결과를 출력하는 함수

설명:&#x20;

* 이 함수는 YOLOv3 네트워크를 이용하여 입력된 이미지 파일을 분류하고 예측 결과를 출력합니다. 함수가 호출될 때는 위에서 설명한 다섯 가지 입력값을 받게 됩니다.
* 함수의 주요 동작은 다음과 같습니다.

1. 설정 파일에서 라벨 정보를 읽어들입니다.
2. 이미지 파일을 불러들입니다.
3. 이미지를 YOLOv3 모델에 입력 가능한 크기로 변환합니다.
4. 모델을 이용하여 예측 결과를 계산합니다.
5. 계산된 예측 결과 중 상위 n개를 출력합니다.

* 위 함수의 입력값 중 filename은 선택적입니다. 이 값이 NULL이 아닌 경우에는 해당 경로의 이미지 파일을 사용하여 예측 결과를 출력합니다. filename이 NULL인 경우에는 사용자에게 이미지 파일 경로를 입력받습니다.
* 함수의 반환값은 없습니다.



## label\_classifier

```c
void label_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);

        printf("%s\n", labels[ind]);
    }
}
```

함수 이름: label\_classifier&#x20;

입력:

* char \*datacfg: 데이터 설정 파일 경로
* char \*filename: 모델 설정 파일 경로
* char \*weightfile: 모델 가중치 파일 경로

동작:&#x20;

* 주어진 데이터 설정 파일, 모델 설정 파일, 모델 가중치 파일을 이용하여 모델을 로드하고, 테스트 이미지 경로를 읽어들여 이미지를 로드한 후 모델을 이용하여 이미지를 분류하고, 해당 이미지의 클래스 이름을 출력한다.

설명:&#x20;

* label\_classifier 함수는 주어진 데이터 설정 파일(datacfg), 모델 설정 파일(filename), 모델 가중치 파일(weightfile)을 이용하여 모델을 로드한다.&#x20;
* 그리고 설정 파일에서 클래스 이름 목록(label\_list), 테스트 이미지 경로(test\_list), 클래스 개수(classes)를 읽어들인다.
* 테스트 이미지 경로에서 이미지를 읽어들여 이미지를 모델 입력 크기에 맞게 리사이즈하고, 이미지 중앙에서 모델 입력 크기만큼 크롭한 이미지를 모델에 입력하여 예측 결과를 얻는다.&#x20;
* 그리고 예측 결과 중 가장 높은 값을 가지는 클래스 인덱스를 구하고, 클래스 이름 목록에서 해당 인덱스에 해당하는 클래스 이름을 찾아 출력한다.
* 이 함수는 이미지 분류에 주로 사용되며, 단일 이미지에 대한 예측 결과를 출력한다.



## csv\_classifier

```c
void csv_classifier(char *datacfg, char *cfgfile, char *weightfile)
{
    int i,j;
    network *net = load_network(cfgfile, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int top = option_find_int(options, "top", 1);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    int *indexes = calloc(top, sizeof(int));

    for(i = 0; i < m; ++i){
        double time = what_time_is_it_now();
        char *path = paths[i];
        image im = load_image_color(path, 0, 0);
        image r = letterbox_image(im, net->w, net->h);
        float *predictions = network_predict(net, r.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);

        printf("%s", path);
        for(j = 0; j < top; ++j){
            printf("\t%d", indexes[j]);
        }
        printf("\n");

        free_image(im);
        free_image(r);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", what_time_is_it_now() - time, i+1, m);
    }
}
```

함수 이름: csv\_classifier

입력:

* datacfg: char 형식의 데이터 설정 파일 경로
* cfgfile: char 형식의 네트워크 설정 파일 경로
* weightfile: char 형식의 네트워크 가중치 파일 경로

동작:&#x20;

* csv 형식의 분류 결과를 출력하는 함수.&#x20;
* 입력된 데이터 설정 파일, 네트워크 설정 파일, 네트워크 가중치 파일을 사용하여 네트워크를 로드하고, 테스트 데이터의 경로를 가져와서 분류를 수행한 후, 결과를 csv 형식으로 출력한다.

설명:

* 입력된 데이터 설정 파일, 네트워크 설정 파일, 네트워크 가중치 파일을 사용하여 네트워크를 로드한다.
* 테스트 데이터의 경로를 가져와서 분류를 수행한다.
* 분류 결과를 csv 형식으로 출력한다. 출력되는 내용은 각 이미지 파일의 경로와 상위 n개의 클래스 인덱스이다.
* 상위 n개의 클래스 인덱스는 top\_k() 함수를 사용하여 구한다.
* 분류가 수행되는 동안 경과 시간과 처리된 이미지 수 등의 정보를 stderr에 출력한다.



## test\_classifier

```c
void test_classifier(char *datacfg, char *cfgfile, char *weightfile, int target_layer)
{
    int curr = 0;
    network *net = load_network(cfgfile, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.classes = classes;
    args.n = net->batch;
    args.m = 0;
    args.labels = 0;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(curr = net->batch; curr < m; curr += net->batch){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        if(curr < m){
            args.paths = paths + curr;
            if (curr + net->batch > m) args.n = m - curr;
            load_thread = load_data_in_thread(args);
        }
        fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);

        int i, j;
        if (target_layer >= 0){
            //layer l = net->layers[target_layer];
        }

        for(i = 0; i < pred.rows; ++i){
            printf("%s", paths[curr-net->batch+i]);
            for(j = 0; j < pred.cols; ++j){
                printf("\t%g", pred.vals[i][j]);
            }
            printf("\n");
        }

        free_matrix(pred);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr);
        free_data(val);
    }
}
```

함수 이름: test\_classifier

입력:

* datacfg: char 포인터. 데이터 구성 파일(data configuration file) 경로를 지정하는 문자열.
* cfgfile: char 포인터. 모델 구성 파일(configuration file) 경로를 지정하는 문자열.
* weightfile: char 포인터. 모델 가중치(weight) 파일 경로를 지정하는 문자열.
* target\_layer: int 타입. 특정 레이어(layer)의 출력값을 출력할 때 해당 레이어의 인덱스를 지정하는 정수.

동작:&#x20;

* 주어진 모델 파일과 가중치 파일을 사용하여 모델을 로드하고, 지정된 데이터 구성 파일을 사용하여 테스트 이미지 경로를 가져온 후, 이를 이용하여 모델을 테스트한다.&#x20;
* 배치(batch) 단위로 이미지를 불러와 모델을 통해 예측하고, 예측값을 출력한다. 특정 레이어의 출력값을 출력할 수도 있다.

설명:

* load\_network: 지정된 모델 파일과 가중치 파일을 사용하여 네트워크(network)를 로드한다.
* read\_data\_cfg: 데이터 구성 파일에서 옵션을 읽어들인다.
* option\_find\_str: 옵션 중 문자열 값을 찾는다.
* option\_find\_int: 옵션 중 정수 값을 찾는다.
* get\_paths: 이미지 경로 리스트를 가져온다.
* load\_data\_in\_thread: 이미지 데이터를 배치 단위로 로드한다.
* network\_predict\_data: 로드된 이미지 데이터를 이용하여 모델을 예측하고, 예측값을 반환한다.
* free\_data: data 구조체에서 할당된 메모리를 해제한다.



## file\_output\_classifier

```c
void file_output_classifier(char *datacfg, char *filename, char *weightfile, char *listfile)
{
    int i,j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    //char *label_list = option_find_str(options, "names", "data/labels.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(listfile);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);

        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 0, 1);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);

        printf("%s", paths[i]);
        for(j = 0; j < classes; ++j){
            printf("\t%g", pred[j]);
        }
        printf("\n");
    }
}
```

함수 이름: file\_output\_classifier

입력:

* char \*datacfg: 데이터 설정 파일 경로
* char \*filename: 모델 파일 경로
* char \*weightfile: 모델 가중치 파일 경로
* char \*listfile: 입력 이미지 경로가 포함된 파일 경로

동작:&#x20;

* 이미지 파일들이 포함된 listfile에서 이미지를 로드하고, 해당 이미지를 모델의 입력 크기로 변환한 후, 모델에 입력하여 예측값을 출력하는 함수입니다.&#x20;
* 출력은 이미지 경로와 각 클래스별 예측값으로 구성된 텍스트 파일로 출력됩니다.

설명:

1. 모델과 가중치를 로드합니다.
2. 데이터 설정 파일에서 클래스 수를 읽어옵니다.
3. 이미지 파일 경로가 포함된 listfile에서 이미지 경로를 읽어옵니다.
4. 각 이미지에 대해 다음 작업을 수행합니다.
   1. 이미지를 로드하고, 모델 입력 크기로 변환한 후 모델에 입력으로 제공합니다.
   2. 모델에서 예측값을 계산합니다.
   3. 계산된 예측값을 텍스트 파일로 출력합니다.



## threat\_classifier

```c
void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    float threat = 0;
    float roll = .2;

    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    //cvNamedWindow("Threat", CV_WINDOW_NORMAL);
    //cvResizeWindow("Threat", 512, 512);
    float fps = 0;
    int i;

    int count = 0;

    while(1){
        ++count;
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        if(!in.data) break;
        image in_s = resize_image(in, net->w, net->h);

        image out = in;
        int x1 = out.w / 20;
        int y1 = out.h / 20;
        int x2 = 2*x1;
        int y2 = out.h - out.h/20;

        int border = .01*out.h;
        int h = y2 - y1 - 2*border;
        int w = x2 - x1 - 2*border;

        float *predictions = network_predict(net, in_s.data);
        float curr_threat = 0;
        if(1){
            curr_threat = predictions[0] * 0 +
                predictions[1] * .6 +
                predictions[2];
        } else {
            curr_threat = predictions[218] +
                predictions[539] +
                predictions[540] +
                predictions[368] +
                predictions[369] +
                predictions[370];
        }
        threat = roll * curr_threat + (1-roll) * threat;

        draw_box_width(out, x2 + border, y1 + .02*h, x2 + .5 * w, y1 + .02*h + border, border, 0,0,0);
        if(threat > .97) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .02*h - 2*border,
                    x2 + .5 * w + 6*border,
                    y1 + .02*h + 3*border, 3*border, 1,0,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .02*h - 2*border,
                x2 + .5 * w + 6*border,
                y1 + .02*h + 3*border, .5*border, 0,0,0);
        draw_box_width(out, x2 + border, y1 + .42*h, x2 + .5 * w, y1 + .42*h + border, border, 0,0,0);
        if(threat > .57) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .42*h - 2*border,
                    x2 + .5 * w + 6*border,
                    y1 + .42*h + 3*border, 3*border, 1,1,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .42*h - 2*border,
                x2 + .5 * w + 6*border,
                y1 + .42*h + 3*border, .5*border, 0,0,0);

        draw_box_width(out, x1, y1, x2, y2, border, 0,0,0);
        for(i = 0; i < threat * h ; ++i){
            float ratio = (float) i / h;
            float r = (ratio < .5) ? (2*(ratio)) : 1;
            float g = (ratio < .5) ? 1 : 1 - 2*(ratio - .5);
            draw_box_width(out, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
        }
        top_predictions(net, top, indexes);
        char buff[256];
        sprintf(buff, "/home/pjreddie/tmp/threat_%06d", count);
        //save_image(out, buff);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        if(1){
            show_image(out, "Threat", 10);
        }
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}
```

함수 이름: threat\_classifier&#x20;

입력:

* datacfg: char 포인터, data 파일 경로
* cfgfile: char 포인터, 모델의 구조를 정의한 파일의 경로
* weightfile: char 포인터, 모델의 가중치가 저장된 파일의 경로
* cam\_index: int, 웹캠의 인덱스
* filename: char 포인터, 비디오 파일의 경로

동작:&#x20;

* 입력된 파일 경로와 인덱스를 통해 웹캠 또는 비디오를 열고, 모델을 로드한 후, 입력 이미지에서 특정 객체의 위협 수준을 예측하여 그에 따라 박스와 색상을 그려 출력한다.&#x20;
* 모델은 Darknet을 사용하며, OpenCV 라이브러리가 필요하다.

설명:&#x20;

* 주어진 경로에서 데이터 파일(datacfg), 모델 파일(cfgfile), 가중치 파일(weightfile)을 로드하여 모델(network)을 생성한다. 이 모델은 입력 이미지에서 객체의 위협 수준(threat)을 예측하는데 사용된다.
* top과 names는 모델을 정의하는 파일에서 읽어오며, top은 예측 결과 중 가장 높은 상위 n개의 값을 가져온다.
* 입력된 비디오 파일(filename) 또는 웹캠(cam\_index)에서 프레임을 가져와 처리한다. 이 때, 처리할 때마다 현재의 위협 수준을 기존의 위협 수준에 일정 비율(roll)을 적용하여 업데이트한다.
* 출력된 이미지는 위협 수준(threat)에 따라 박스와 색상이 그려져 있으며, OpenCV 라이브러리의 show\_image 함수를 사용하여 실시간으로 보여준다.
* 최종적으로 처리한 이미지와 FPS(Frame Per Second)를 반환한다.



## gun\_classifier

```c
void gun_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    int bad_cats[] = {218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697};

    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = resize_image(in, net->w, net->h);

        float *predictions = network_predict(net, in_s.data);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");

        int threat = 0;
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("Threat Detected!\n");
                threat = 1;
                break;
            }
        }
        if(!threat) printf("Scanning...\n");
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("%s\n", names[index]);
            }
        }

        show_image(in, "Threat Detection", 10);
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}
```

함수 이름: gun\_classifier&#x20;

입력:

* datacfg: 문자열 포인터. 데이터 구성 파일 경로.
* cfgfile: 문자열 포인터. 네트워크 구성 파일 경로.
* weightfile: 문자열 포인터. 네트워크 가중치 파일 경로.
* cam\_index: 정수. 웹캠 인덱스.
* filename: 문자열 포인터. 비디오 파일 경로.

동작:

* 지정된 네트워크 구성 파일과 가중치 파일을 사용하여 네트워크를 로드합니다.
* 네트워크의 입력 크기를 설정합니다.
* 데이터 구성 파일에서 옵션을 읽어옵니다.
* 웹캠 또는 비디오 파일에서 이미지를 얻어와 네트워크를 통해 예측합니다.
* 지정된 임계값 이상인 클래스 인덱스를 출력합니다.
* 예측된 이미지와 예측된 클래스 이름을 출력하고, 위협으로 간주되는 클래스가 있는 경우 "Threat Detected!"을 출력합니다.
* 이미지를 표시하고, FPS(초당 프레임 수)를 계산합니다.

설명:&#x20;

* 이 함수는 입력된 데이터 구성 파일, 네트워크 구성 파일, 가중치 파일을 사용하여 이미지나 비디오에서 촬영한 영상 데이터를 분석하고, 위험으로 간주되는 클래스가 있는지 감지하는 기능을 합니다.&#x20;
* 이 함수는 OpenCV 라이브러리를 사용합니다. 함수는 웹캠 또는 비디오 파일에서 이미지를 얻어와 분석을 수행하며, 분석 결과를 콘솔에 출력하고, FPS를 계산하여 이미지를 실시간으로 표시합니다.&#x20;
* 분석 시 "bad\_cats" 배열에 지정된 클래스 중, 예측된 확률 값이 0.01 이상인 클래스가 있는 경우 "Threat Detected!"을 출력하고, 위협으로 간주되는 클래스의 이름을 출력합니다.



## demo\_classifier

```c
void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    image **alphabet = load_alphabet();
    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);

    int w = 1280;
    int h = 720;
    void * cap = open_video_stream(filename, cam_index, w, h, 0);

    int top = option_find_int(options, "top", 1);

    char *label_list = option_find_str(options, "labels", 0);
    char *name_list = option_find_str(options, "names", label_list);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);

        float *predictions = network_predict(net, in_s.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        int lh = in.h*.03;
        int toph = 3*lh;

        float rgb[3] = {1,1,1};
        for(i = 0; i < top; ++i){
            printf("%d\n", toph);
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);

            char buff[1024];
            sprintf(buff, "%3.1f%%: %s\n", predictions[index]*100, names[index]);
            image label = get_label(alphabet, buff, lh);
            draw_label(in, toph, lh, label, rgb);
            toph += 2*lh;
            free_image(label);
        }

        show_image(in, base, 10);
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}
```

함수 이름: demo\_classifier&#x20;

입력:

* datacfg: 문자열 포인터, 데이터 파일 경로
* cfgfile: 문자열 포인터, 네트워크 구성 파일 경로
* weightfile: 문자열 포인터, 학습된 가중치 파일 경로
* cam\_index: 정수, 웹캠 인덱스
* filename: 문자열 포인터, 동영상 파일 경로

동작:&#x20;

* 주어진 데이터 파일, 네트워크 구성 파일, 학습된 가중치 파일을 사용하여 분류기 네트워크를 로드하고, 웹캠 또는 동영상에서 프레임을 가져와 입력 이미지로 변환합니다.&#x20;
* 그런 다음, 분류기 네트워크를 사용하여 입력 이미지에서 클래스 예측을 수행하고, 가장 높은 예측 점수를 가진 상위 클래스를 인쇄하고 화면에 표시합니다.

설명:

* 이 함수는 영상 데이터에 대한 분류기 모델의 성능을 시각적으로 평가하기 위해 사용됩니다.&#x20;
* 주어진 데이터 파일, 네트워크 구성 파일 및 학습된 가중치 파일을 사용하여 분류기 네트워크를 로드하고, 웹캠 또는 동영상 파일에서 프레임을 가져와 입력 이미지로 변환합니다.&#x20;
* 그런 다음, 분류기 네트워크를 사용하여 입력 이미지에서 클래스 예측을 수행하고, 가장 높은 예측 점수를 가진 상위 클래스를 인쇄하고 화면에 표시합니다. 이 함수는 OpenCV 라이브러리를 사용합니다.



## run\_classifier

```c
void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);


    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *layer_s = (argc > 7) ? argv[7]: 0;
    int layer = layer_s ? atoi(layer_s) : -1;
    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "fout")) file_output_classifier(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, layer);
    else if(0==strcmp(argv[2], "csv")) csv_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights);
    else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);
}
```

함수 이름: run\_classifier

입력:

* int argc : main 함수에서 전달된 명령행 인자의 개수
* char \*\*argv : main 함수에서 전달된 명령행 인자 문자열 배열

동작:

* 인자로 받은 명령행 인자를 파싱하여 해당하는 함수를 호출하는 역할을 수행

설명:

* 주어진 명령행 인자를 파싱하여 해당하는 함수를 호출
* train, test, valid 등의 옵션에 따라 다른 함수를 호출하여 해당 작업을 수행
* 다양한 옵션에 따라 다양한 동작을 수행할 수 있으며, 이를 위해 다양한 명령행 인자를 받음
* GPU 사용 여부, 카메라 인덱스, 출력 파일명 등을 인자로 받아 처리 가능

