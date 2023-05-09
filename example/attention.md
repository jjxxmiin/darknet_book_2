# attention

```c
#include "darknet.h"

#include <sys/time.h>
#include <assert.h>
```

## extend\_data\_truth

```c
void extend_data_truth(data *d, int n, float val)
{
    int i, j;
    for(i = 0; i < d->y.rows; ++i){
        d->y.vals[i] = realloc(d->y.vals[i], (d->y.cols+n)*sizeof(float));
        for(j = 0; j < n; ++j){
            d->y.vals[i][d->y.cols + j] = val;
        }
    }
    d->y.cols += n;
}
```

함수 이름: extend\_data\_truth

입력:&#x20;

* (data \*d): 데이터 구조체 포인터
* (int n): 추가할 열 수
* (float val): 추가할 값

동작:&#x20;

* 데이터 구조체 d의 y 행렬의 각 행에 대해 n개의 열을 추가하고, 해당 열의 값을 모두 val로 설정합니다.

설명:&#x20;

* 입력으로 받은 데이터 구조체 포인터 d의 y 행렬에 n개의 열을 추가하고, 각 행의 추가된 열들의 값을 모두 val로 설정하는 함수입니다.&#x20;
* 이 함수를 사용하면 데이터셋의 y값을 확장할 수 있습니다.



## network\_loss\_data

```c
matrix network_loss_data(network *net, data test)
{
    int i,b;
    int k = 1;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    float *y = calloc(net->batch*test.y.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
            memcpy(y+b*test.y.cols, test.y.vals[i+b], test.y.cols*sizeof(float));
        }

        network orig = *net;
        net->input = X;
        net->truth = y;
        net->train = 0;
        net->delta = 0;
        forward_network(net);
        *net = orig;

        float *delta = net->layers[net->n-1].output;
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            int t = max_index(y + b*test.y.cols, 1000);
            float err = sum_array(delta + b*net->outputs, net->outputs);
            pred.vals[i+b][0] = -err;
            //pred.vals[i+b][0] = 1-delta[b*net->outputs + t];
        }
    }
    free(X);
    free(y);
    return pred;   
}
```

함수 이름: network\_loss\_data

입력:

* network \*net: 신경망 모델을 나타내는 포인터
* data test: 테스트 데이터를 담고 있는 구조체 포인터

동작:

* 주어진 신경망 모델과 테스트 데이터를 이용해 모델의 손실값을 계산하는 함수
* 입력으로 주어진 신경망 모델의 가중치를 고정시키고, 입력 데이터를 통해 순전파(forward) 연산을 수행하여 출력값을 계산하고, 이를 토대로 손실값을 계산함
* 테스트 데이터는 배치(batch) 단위로 처리됨

설명:

* 입력으로 받은 뉴럴 네트워크 "net"에 대해 다음과 같은 작업을 수행합니다:
  * "net->batch" 개수 만큼 입력 데이터를 포함하는 메모리 "X"와 정답 데이터를 포함하는 메모리 "y"를 동적으로 할당합니다.
  * "test.X"와 "test.y"에서 "net->batch" 개수 만큼의 데이터를 복사하여 "X"와 "y"에 저장합니다.
  * "net->train" 변수를 0으로 설정하여 뉴럴 네트워크를 테스트 모드로 전환합니다.
  * "net->delta" 변수를 0으로 초기화합니다.
  * "forward\_network" 함수를 호출하여 뉴럴 네트워크의 출력값을 계산합니다.
  * 뉴럴 네트워크를 원래의 상태로 되돌립니다.
* 계산된 출력값 "delta"를 이용하여 다음과 같은 작업을 수행합니다:
  * 각 mini-batch에 대해 오차를 계산합니다.
  * 계산된 오차를 "matrix" 형태로 저장합니다.
* 마지막으로, 계산된 오차를 저장한 "matrix"를 반환합니다.



## train\_attention

```c
void train_attention(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i, j;

    float avg_cls_loss = -1;
    float avg_att_loss = -1;
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
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    int divs=3;
    int size=2;

    load_args args = {0};
    args.w = divs*net->w/size;
    args.h = divs*net->h/size;
    args.size = divs*net->w/size;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*args.w;
    args.max = net->max_ratio*args.w;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        data resized = resize_data(train, net->w, net->h);
        extend_data_truth(&resized, divs*divs, 0);
        data *tiles = tile_data(train, divs, size);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float aloss = 0;
        float closs = 0;
        int z;
        for (i = 0; i < divs*divs/ngpus; ++i) {
#pragma omp parallel for
            for(j = 0; j < ngpus; ++j){
                int index = i*ngpus + j;
                extend_data_truth(tiles+index, divs*divs, SECRET_NUM);
                matrix deltas = network_loss_data(nets[j], tiles[index]);
                for(z = 0; z < resized.y.rows; ++z){
                    resized.y.vals[z][train.y.cols + index] = deltas.vals[z][0];
                }
                free_matrix(deltas);
            }
        }
        int *inds = calloc(resized.y.rows, sizeof(int));
        for(z = 0; z < resized.y.rows; ++z){
            int index = max_index(resized.y.vals[z] + train.y.cols, divs*divs);
            inds[z] = index;
            for(i = 0; i < divs*divs; ++i){
                resized.y.vals[z][train.y.cols + i] = (i == index)? 1 : 0;
            }
        }
        data best = select_data(tiles, inds);
        free(inds);
        #ifdef GPU
        if (ngpus == 1) {
            closs = train_network(net, best);
        } else {
            closs = train_networks(nets, ngpus, best, 4);
        }
        #endif
        for (i = 0; i < divs*divs; ++i) {
            printf("%.2f ", resized.y.vals[0][train.y.cols + i]);
            if((i+1)%divs == 0) printf("\n");
            free_data(tiles[i]);
        }
        free_data(best);
        printf("\n");
        image im = float_to_image(64,64,3,resized.X.vals[0]);
        //show_image(im, "orig");
        //cvWaitKey(100);
        /*
           image im1 = float_to_image(64,64,3,tiles[i].X.vals[0]);
           image im2 = float_to_image(64,64,3,resized.X.vals[0]);
           show_image(im1, "tile");
           show_image(im2, "res");
         */
#ifdef GPU
        if (ngpus == 1) {
            aloss = train_network(net, resized);
        } else {
            aloss = train_networks(nets, ngpus, resized, 4);
        }
#endif
        for(i = 0; i < divs*divs; ++i){
            printf("%f ", nets[0]->output[1000 + i]);
            if ((i+1) % divs == 0) printf("\n");
        }
        printf("\n");

        free_data(resized);
        free_data(train);
        if(avg_cls_loss == -1) avg_cls_loss = closs;
        if(avg_att_loss == -1) avg_att_loss = aloss;
        avg_cls_loss = avg_cls_loss*.9 + closs*.1;
        avg_att_loss = avg_att_loss*.9 + aloss*.1;

        printf("%ld, %.3f: Att: %f, %f avg, Class: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, aloss, avg_att_loss, closs, avg_cls_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
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
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
```

함수 이름: train\_attention&#x20;

입력:

* datacfg (str): 데이터 구성을 위한 설정 파일
* cfgfile (str): YOLOv3 아키텍처를 위한 설정 파일
* weightfile (str): YOLOv3의 가중치 파일
* gpus (list of int): 사용할 GPU의 번호를 담은 배열
* ngpus (int): 사용할 GPU의 개수
* clear (bool): 누적된 그래디언트를 지울지 여부를 지정하는 플래그

동작:

* YOLOv3 네트워크를 `load_network` 함수를 사용하여 로드한다.
* 학습률을 `ngpus * learning_rate`로 설정하고, 이때 `learning_rate`는 설정 파일에 정의된 하이퍼파라미터이다.
* `train_list`에서 학습 데이터를 로드한다.
* 학습 데이터의 크기를 조정하고, `tile_data` 함수를 사용하여 작은 타일로 나누어 네트워크에서 학습한다.
* 최대 배치 수가 도달하거나 설정 파일의 `max_batches` 매개변수가 0으로 설정된 경우, 학습은 무기한으로 계속된다.
* 각 반복에서, 별도의 스레드를 사용하여 데이터 배치를 로드하고 크기를 조정한 후, `divs*divs` 앵커 박스를 포함하는 진실 값을 확장하여 선택된 타일로 네트워크를 학습하고 그래디언트를 업데이트한다.
* 최상의 타일을 선택하고 다른 타일을 해제한 후, 선택된 타일의 어텐션 맵을 출력하고 입력 데이터와 어텐션 맵을 시각화하기 위해 이미지로 변환한다.
* 함수는 반환값이 없다.



## validate\_attention\_single

```c
void validate_attention_single(char *datacfg, char *filename, char *weightfile)
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
    int divs = 4;
    int size = 2;
    int extra = 0;
    float *avgs = calloc(classes, sizeof(float));
    int *inds = calloc(divs*divs, sizeof(int));

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
        image resized = resize_min(im, net->w*divs/size);
        image crop = crop_image(resized, (resized.w - net->w*divs/size)/2, (resized.h - net->h*divs/size)/2, net->w*divs/size, net->h*divs/size);
        image rcrop = resize_image(crop, net->w, net->h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, rcrop.data);
        //pred[classes + 56] = 0;
        for(j = 0; j < divs*divs; ++j){
            printf("%.2f ", pred[classes + j]);
            if((j+1)%divs == 0) printf("\n");
        }
        printf("\n");
        copy_cpu(classes, pred, 1, avgs, 1);
        top_k(pred + classes, divs*divs, divs*divs, inds);
        show_image(crop, "crop");
        for(j = 0; j < extra; ++j){
            int index = inds[j];
            int row = index / divs;
            int col = index % divs;
            int y = row * crop.h / divs - (net->h - crop.h/divs)/2;
            int x = col * crop.w / divs - (net->w - crop.w/divs)/2;
            printf("%d %d %d %d\n", row, col, y, x);
            image tile = crop_image(crop, x, y, net->w, net->h);
            float *pred = network_predict(net, tile.data);
            axpy_cpu(classes, 1., pred, 1, avgs, 1);
            show_image(tile, "tile");
            //cvWaitKey(10);
        }
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        if(rcrop.data != resized.data) free_image(rcrop);
        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}
```

함수 이름: validate\_attention\_single

입력:

* datacfg: 데이터 설정 파일 경로를 가리키는 문자열 포인터
* filename: 네트워크 구조 파일 경로를 가리키는 문자열 포인터
* weightfile: 네트워크 가중치 파일 경로를 가리키는 문자열 포인터

동작:&#x20;

* 주어진 네트워크 모델에 대해 입력 데이터를 이용하여 검증을 수행하는 함수입니다.&#x20;
* 주어진 경로에 있는 설정 파일을 읽어 데이터와 네트워크 구조를 초기화한 후, 입력 이미지를 로드하고 크기를 조정하며 네트워크 모델을 적용합니다.&#x20;
* 결과를 평가하고 출력합니다.
* 이 함수는 다음과 같은 동작을 수행합니다:
  1. 데이터 설정 파일(datacfg)을 읽어들입니다.
  2. 입력 이미지의 클래스 수(classes), 상위 K(topk) 값을 설정합니다.
  3. 입력 이미지를 로드하고, 크기를 조정합니다.
  4. 크기를 조정한 이미지를 crop합니다.
  5. crop한 이미지를 이용하여 네트워크 모델을 적용합니다.
  6. 네트워크 모델의 출력 값을 이용하여 정확도를 평가합니다.
  7. 평가 결과를 출력합니다.

설명:&#x20;

* 이 함수는 Darknet 프레임워크의 코드 중 하나입니다.&#x20;
* 주어진 경로의 데이터 설정 파일(datacfg), 네트워크 구조 파일(filename), 네트워크 가중치 파일(weightfile)을 이용하여 네트워크 모델을 초기화하고, 입력 데이터를 이용하여 검증을 수행합니다.&#x20;
* 검증 결과는 평균 정확도(avg\_acc)와 평균 상위 K(topk) 정확도(avg\_topk)를 출력합니다.





## validate\_attention\_multi

```c
void validate_attention_multi(char *datacfg, char *filename, char *weightfile)
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
    int scales[] = {224, 288, 320, 352, 384};
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
            image r = resize_min(im, scales[j]);
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

함수 이름: validate\_attention\_multi&#x20;

입력:

* char\* datacfg: 데이터 설정 파일 경로
* char\* filename: 네트워크 구성 파일 경로
* char\* weightfile: 학습된 모델의 가중치 파일 경로&#x20;

동작:&#x20;

* 다중 크기의 이미지를 사용하여 지정된 네트워크를 검증하는 함수입니다.&#x20;
* 네트워크가 정확하게 분류하는 이미지의 비율과 상위 k개 분류 중 올바르게 분류하는 이미지의 비율을 출력합니다.&#x20;

설명:

* 네트워크 구성 파일과 가중치 파일을 로드하고, 배치 크기를 1로 설정합니다.
* 데이터 설정 파일에서 레이블 목록, 검증 이미지 목록, 클래스 수, 상위 k개를 가져옵니다.
* 검증 이미지를 읽어들이고, 다중 크기를 사용하여 이미지를 조정하고, 네트워크를 통해 예측합니다.
* 예측 결과에서 상위 k개 분류를 선택하고, 정확도와 상위 k개의 정확도를 계산합니다.
* 모든 검증 이미지에 대해 정확도와 상위 k개의 정확도를 계산하고 출력합니다.



## predict\_attention

```c
void predict_attention(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
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
        //resize_network(&net, r.w, r.h);
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

함수 이름: predict\_attention&#x20;

입력:

* char \*datacfg: 데이터 구성 파일의 경로를 나타내는 문자열 포인터
* char \*cfgfile: YOLO 네트워크 구성 파일의 경로를 나타내는 문자열 포인터
* char \*weightfile: YOLO 네트워크 가중치 파일의 경로를 나타내는 문자열 포인터
* char \*filename: 이미지 파일 경로를 나타내는 문자열 포인터 (선택적)
* int top: 예측 결과에서 상위 N개의 클래스를 보여줄지 결정하는 정수

동작:

* YOLO 네트워크를 로드하고 입력 이미지에서 객체를 감지하여 예측 결과를 출력하는 함수
* 입력 이미지 경로를 직접 입력하거나 인자로 전달받은 이미지 파일 경로를 사용
* 예측 결과에서 상위 N개의 클래스를 보여줌
* 함수가 종료되기 전까지 무한 루프를 실행하여 계속해서 입력 이미지를 받을 수 있음

설명:

* 함수는 YOLO 네트워크를 로드하고, 이를 이용하여 입력 이미지에서 객체를 감지하고 예측 결과를 출력함
* 예측 결과에서 보여줄 클래스의 수(top)는 입력으로 받거나, 데이터 구성 파일에서 찾거나 기본값 1로 설정됨
* 함수는 무한 루프를 실행하며, 사용자가 직접 이미지 경로를 입력하거나 인자로 전달받은 이미지 파일 경로를 사용하여 예측을 수행함
* 예측 결과에서는 각 클래스의 이름과 해당 클래스일 확률이 백분율로 출력됨
* 함수가 실행되면 먼저 입력 이미지를 로드하고, YOLO 네트워크에서 사용되는 크기로 이미지를 조절함
* 이후, 예측을 수행하고 예측 결과를 출력함
* 예측 결과는 클래스의 이름과 확률값으로 이루어져 있으며, 상위 N개의 클래스만 출력함
* 함수가 종료되기 전까지 무한 루프를 실행하여 계속해서 입력 이미지를 받을 수 있음



## run\_attention

```c
void run_attention(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);


    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *layer_s = (argc > 7) ? argv[7]: 0;
    if(0==strcmp(argv[2], "predict")) predict_attention(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "train")) train_attention(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_attention_single(data, cfg, weights);
    else if(0==strcmp(argv[2], "validmulti")) validate_attention_multi(data, cfg, weights);
}
```

함수 이름: run\_attention

입력:

* argc: int, 프로그램 실행 시 전달된 인자의 개수
* argv: char\*\*, 프로그램 실행 시 전달된 인자들의 배열

동작:

* 주어진 인자들을 분석하여 해당하는 작업을 실행함
* "predict" 인자가 주어지면, predict\_attention 함수를 호출하여 이미지 파일의 경로를 입력받고 해당 이미지에 대한 객체 인식 결과를 출력함
* "train" 인자가 주어지면, train\_attention 함수를 호출하여 지정된 데이터셋으로부터 모델을 학습시킴
* "valid" 인자가 주어지면, validate\_attention\_single 함수를 호출하여 지정된 데이터셋으로부터 모델의 성능을 평가함
* "validmulti" 인자가 주어지면, validate\_attention\_multi 함수를 호출하여 지정된 데이터셋으로부터 모델의 성능을 평가함

설명:

* gpu\_list: char\*, 사용할 GPU 목록을 지정함
* ngpus: int, 사용할 GPU의 개수
* gpus: int\*, 사용할 GPU의 인덱스
* top: int, 객체 인식 결과 중 상위 몇 개의 결과를 출력할지 지정함
* clear: int, 학습을 시작하기 전에 이전에 학습된 내용을 삭제할지 여부를 지정함
* data: char\*, 데이터셋의 경로를 지정함
* cfg: char\*, 모델의 설정 파일 경로를 지정함
* weights: char\*, 모델의 가중치 파일 경로를 지정함 (생략 가능)
* filename: char\*, 객체 인식을 수행할 이미지 파일의 경로를 지정함 (생략 가능)
* layer\_s: char\*, 모델에서 출력할 레이어 이름을 지정함 (생략 가능)

