# data

## get\_paths

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}
```

함수 이름: get\_paths

입력:&#x20;

* char \*filename: 파일 이름

동작:

1. pthread\_mutex\_t mutex를 PTHREAD\_MUTEX\_INITIALIZER로 초기화
2. 입력받은 파일 이름으로 파일을 열고, 파일이 없는 경우에는 file\_error 함수를 호출하여 오류 메시지 출력
3. 빈 리스트(lines)를 만들기 위해 make\_list() 함수 호출
4. 파일에서 한 줄씩 읽어오면서 리스트에 추가(list\_insert 함수 사용)
5. 파일 닫기
6. 생성된 리스트(lines) 반환

설명:

* get\_paths 함수는 파일 이름을 입력받아 해당 파일에서 한 줄씩 읽어와 리스트에 추가하는 함수이다.&#x20;
* 이 함수는 파일이 없는 경우에 오류 메시지를 출력하고, 파일에서 읽어온 내용을 리스트에 추가하여 반환한다.&#x20;
* 이 함수에서는 뮤텍스(mutex)를 사용하여 스레드 간의 경쟁 상황을 막아 안전하게 리스트를 수정할 수 있다.



## get\_random\_paths

```c
char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
        //if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
```

함수 이름: get\_random\_paths

입력:

* paths: 문자열 배열 포인터
* n: 반환할 랜덤 경로의 개수
* m: 전체 경로의 개수

동작:

* paths 배열에서 랜덤으로 n개의 경로를 선택하여 새로운 배열 random\_paths에 저장한다.
* 선택된 경로는 paths 배열에서의 인덱스를 이용하여 가져온다.
* pthread\_mutex\_t 타입의 mutex를 이용하여 스레드 간 경쟁 상황을 방지한다.

설명:

* 함수는 문자열 배열 포인터 paths와 반환할 랜덤 경로의 개수 n, 그리고 전체 경로의 개수 m을 입력 받는다.
* n개의 랜덤 경로를 저장하기 위해 char 타입의 이중 포인터인 random\_paths를 calloc 함수를 이용하여 할당한다.
* mutex를 이용하여 스레드 간 경쟁 상황을 방지한다.
* for 루프를 이용하여 n개의 랜덤 경로를 선택하여 random\_paths에 저장한다.
* 인덱스는 rand 함수를 이용하여 랜덤으로 생성하며, 이를 이용하여 paths 배열에서 해당 인덱스의 경로를 선택하여 random\_paths에 저장한다.
* 모든 랜덤 경로 선택이 끝나면 random\_paths를 반환한다.



## find\_replace\_paths

```c
char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}
```

함수 이름: find\_replace\_paths

입력:

* char \*\*paths: 문자열 배열
* int n: 문자열 배열의 길이
* char \*find: 찾을 문자열
* char \*replace: 대체할 문자열

동작:

* 문자열 배열 paths에서 찾을 문자열 find을 대체할 문자열 replace로 대체하여 replace\_paths라는 새로운 문자열 배열을 만듭니다.
* replace\_paths 배열은 동적으로 할당됩니다.

설명:

* 입력된 문자열 배열 paths에서 각각의 문자열을 대체하기 위해 for문을 사용합니다.
* 각 문자열을 대체하기 위해 find\_replace 함수를 사용합니다.
* 대체한 문자열은 동적으로 할당된 replace\_paths 배열에 저장됩니다.
* 마지막으로 replace\_paths 배열을 반환합니다.



## load\_image\_paths\_gray

```c
matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}
```

함수 이름: load\_image\_paths\_gray

입력:

* char \*\*paths: 이미지 파일 경로를 저장한 문자열 배열
* int n: 이미지 파일 경로 개수
* int w: 이미지의 폭
* int h: 이미지의 높이

동작:&#x20;

* 주어진 경로에서 이미지 파일을 읽어들이고, grayscale로 변환한 후, 픽셀값을 float 형식으로 변환하여 2차원 배열(matrix) X에 저장한다.

설명:&#x20;

* load\_image\_paths\_gray 함수는 char 타입의 이중포인터 paths, int 타입의 n, w, h를 입력값으로 받아들인다.&#x20;
* 이 함수는 이미지 파일 경로에서 이미지를 읽어들인 후 grayscale로 변환하고, 픽셀값을 float 형식으로 변환하여 2차원 배열(matrix) X에 저장한 후, 이를 반환한다.&#x20;
* 이 때, X는 matrix 타입으로, X.vals는 n개의 포인터를 저장할 수 있는 메모리 공간을 calloc을 통해 할당받으며, X.cols는 0으로 초기화된다. 이미지 파일은 load\_image 함수를 통해 읽어들인 후, grayscale\_image 함수를 통해 grayscale로 변환한다.&#x20;
* 그리고 나서, free\_image 함수를 이용해 메모리를 해제하고, 변환된 grayscale 이미지를 다시 im에 할당한다. ㅊ변환된 이미지의 픽셀값을 float 형식으로 변환하여 X의 i번째 요소에 저장하고, X.cols에는 해당 이미지의 가로, 세로, 채널 수에 해당하는 값을 저장한다. 마지막으로, 변환된 X를 반환한다.



## load\_image\_paths

```c
matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}
```

함수 이름: load\_image\_paths&#x20;

입력:

* paths (char \*\*): 이미지 파일 경로 배열
* n (int): 이미지 파일 개수
* w (int): 이미지의 가로 크기
* h (int): 이미지의 세로 크기

동작:&#x20;

* 입력된 이미지 파일 경로들을 이용하여 이미지 데이터를 메모리에 로드하고, 해당 데이터를 이용하여 행렬을 생성하여 반환하는 함수입니다. 이 함수는 이미지를 grayscale로 변환하지 않습니다.

설명:&#x20;

* 이 함수는 입력된 이미지 파일 경로 배열에서 이미지 데이터를 로드하여, 해당 데이터를 행렬로 생성하여 반환합니다.&#x20;
* 이 함수에서 생성된 행렬은 이미지 데이터를 1차원 배열 형태로 저장하며, 각 이미지의 데이터는 행렬의 한 행으로 저장됩니다.&#x20;
* 이 함수는 이미지를 grayscale로 변환하지 않으며, RGB 색상 모델을 사용하여 이미지를 로드합니다.



## load\_image\_augment\_paths

```c
matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop;
        if(center){
            crop = center_crop_image(im, size, size);
        } else {
            crop = random_augment_image(im, angle, aspect, min, max, size, size);
        }
        int flip = rand()%2;
        if (flip) flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        /*
        show_image(im, "orig");
        show_image(crop, "crop");
        cvWaitKey(0);
        */
        //grayscale_image_3c(crop);
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}
```

함수 이름: load\_image\_augment\_paths

입력:

* paths (char \*\*): 이미지 파일 경로 배열
* n (int): 이미지 파일 경로의 개수
* min (int): 이미지 축소 비율의 최소값
* max (int): 이미지 축소 비율의 최대값
* size (int): 이미지 크기
* angle (float): 이미지 회전 각도 범위
* aspect (float): 이미지 확장 비율 범위
* hue (float): 이미지 색상 조정 범위
* saturation (float): 이미지 채도 조정 범위
* exposure (float): 이미지 노출 조정 범위
* center (int): 이미지 중앙으로부터 자를지 여부 (1: 중앙 자르기, 0: 무작위 자르기)

동작:

* 주어진 이미지 경로에서 이미지를 로드하고, 크기를 조정하고, 색상 및 명암 조절을 통해 이미지를 보강한다.
* 이미지 배열을 만들어 이미지 데이터를 저장하고, 이를 반환한다.

설명:

* 이미지 파일 경로 배열 `paths`에서 이미지를 로드하고, 이미지 크기를 `size`로 조정한다.
* `center` 값에 따라 이미지 중앙으로부터 자를지, 무작위로 자를지 결정한다.
* 이미지를 무작위로 회전하고, 확대 또는 축소하며, 이미지 색상, 명암을 무작위로 조절한다.
* 보강된 이미지 데이터를 2차원 행렬로 저장하고, 이를 반환한다.



## read\_boxes

```c
box_label *read_boxes(char *filename, int *n)
{
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    int size = 64;
    box_label *boxes = calloc(size, sizeof(box_label));
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        if(count == size) {
            size = size * 2;
            boxes = realloc(boxes, size*sizeof(box_label));
        }
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}
```

함수 이름: read\_boxes&#x20;

입력:

* filename (char\*): 박스 정보가 저장된 파일의 경로
* n (int\*): 박스의 개수를 저장하기 위한 변수의 포인터

동작:

* 입력된 파일에서 박스 정보를 읽어와 box\_label 구조체에 저장하고, 이들을 배열에 저장
* 필요에 따라 배열의 크기를 조절하면서 박스 정보를 저장하는 동작 수행
* 박스의 개수를 입력받은 포인터를 통해 반환

설명:

* 이 함수는 YOLO 형식의 박스 정보가 담긴 텍스트 파일을 읽어와 box\_label 구조체에 저장하는 기능을 수행한다.
* 파일의 내용은 "id x y w h" 형식으로 구성되며, 각각은 박스의 클래스, 중심점(x, y), 너비(w), 높이(h)를 나타낸다.
* 파일을 읽으면서 box\_label 구조체에 저장하고, 이들을 동적으로 할당된 배열에 추가한다.
* 배열의 크기가 부족해지면, 필요에 따라 두 배씩 늘려가면서 박스 정보를 저장한다.
* 마지막으로, 박스의 개수를 반환한다.



## randomize\_boxes

```c
void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}
```

함수 이름: randomize\_boxes&#x20;

입력:&#x20;

* box\_label \*b: 박스 라벨 포인터
* int n: 박스 라벨의 수&#x20;

동작:&#x20;

* 주어진 박스 라벨 배열에서 무작위로 박스를 선택하여 위치를 바꾼다.&#x20;

설명:&#x20;

* 입력으로 받은 박스 라벨 배열에서 무작위로 두 개의 박스를 선택하여 위치를 서로 바꿔주는 작업을 n번 반복한다.&#x20;
* 이 함수는 객체 검출 알고리즘에서 데이터 증강을 위해 사용될 수 있다.



## correct\_boxes

```c
void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}
```

함수 이름: correct\_boxes

입력:

* box\_label \*boxes: bounding box 정보가 담긴 구조체 포인터 배열
* int n: bounding box의 개수
* float dx: x축으로 이동할 값
* float dy: y축으로 이동할 값
* float sx: x축으로 scale할 값
* float sy: y축으로 scale할 값
* int flip: 좌우 반전 여부를 나타내는 플래그

동작:&#x20;

* 주어진 bounding box 정보를 이동하고 scale한 후, 좌우 반전 여부를 고려하여 수정합니다.

설명:

* 각 bounding box 정보는 (left, top), (right, bottom) 좌표값과 해당 영역의 클래스 정보를 포함합니다.
* 입력으로 주어진 dx, dy, sx, sy는 모두 bounding box 좌표값을 수정하기 위한 값입니다.
* boxes 배열의 각 원소에 대해, 먼저 (left, top), (right, bottom) 좌표값을 이동하고 scale합니다.
* 이후 flip 값이 1일 경우, 좌우 반전이 적용됩니다.
* 마지막으로, 수정된 bounding box 정보를 다시 (x, y, w, h) 형태로 변환합니다.



## fill\_truth\_swag

```c
void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 90; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}
```

함수 이름: fill\_truth\_swag

입력:

* char \*path: 이미지 경로
* float \*truth: 각 객체에 대한 ground truth 정보를 담은 배열
* int classes: 분류할 클래스 수
* int flip: 이미지를 수평으로 뒤집을지 여부를 나타내는 플래그
* float dx: x 방향으로 이동할 값
* float dy: y 방향으로 이동할 값
* float sx: 이미지를 x 방향으로 확대 또는 축소할 배율
* float sy: 이미지를 y 방향으로 확대 또는 축소할 배율

동작:

* 입력받은 이미지의 label 파일 경로를 찾아서 해당 파일로부터 객체의 위치와 크기 정보를 읽어옴
* 객체의 위치와 크기 정보를 무작위로 섞은 후, 이미지의 변환값에 따라 위치와 크기 정보를 수정함
* ground truth 배열에 객체의 정보를 저장함

설명:

* 입력으로 받은 이미지 파일의 label 파일 경로를 찾아서 해당 파일로부터 객체의 위치와 크기 정보를 읽어옴
* 읽어온 정보를 무작위로 섞은 후, 이미지의 변환값(dx, dy, sx, sy)에 따라 위치와 크기 정보를 수정함
* 수정된 정보를 바탕으로 ground truth 배열에 객체의 정보를 저장함
* ground truth 배열에는 각 객체의 x, y 좌표, 너비, 높이 정보와 클래스 정보가 저장됨
* 객체가 클래스 중 어떤 것인지 나타내는 클래스 정보는 one-hot encoding 방식으로 저장됨



## fill\_truth\_region

```c
void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);

    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .005 || h < .005) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}
```

함수 이름: fill\_truth\_region&#x20;

입력:

* path: char 포인터 타입. 라벨 파일 경로를 지정하는 문자열.
* truth: float 포인터 타입. 실제 값(truth)이 채워질 배열.
* classes: int 타입. 클래스(class) 수.
* num\_boxes: int 타입. 각 셀(cell)의 bounding box 개수.
* flip: int 타입. 이미지를 수평으로 뒤집을지 여부를 결정하는 변수.
* dx, dy, sx, sy: float 타입. 이미지를 변환(transform)하는 데 사용되는 변수들.

동작:

* 주어진 경로에서 라벨 파일을 찾아 읽고, bounding box들을 무작위로 섞음(randomize).
* bounding box들을 이미지 변환에 맞게 수정(correct)하고, 각 셀에 해당하는 인덱스(index)를 계산함.
* 각 bounding box의 실제 값(truth)을 계산하고, truth 배열에 저장함.

설명:&#x20;

* 이 함수는 YOLO 알고리즘에서 bounding box를 처리하기 위한 함수입니다.&#x20;
* 이미지 경로를 받아 해당 이미지의 라벨 파일을 찾아서 bounding box 정보를 읽은 후, 각 bounding box를 실제 값(truth)으로 변환하여 truth 배열에 저장합니다.&#x20;
* 이때, 이미지의 변환과 셀(cell)의 개수(num\_boxes)를 고려하여 bounding box들이 어떤 셀에 속하는지 계산합니다.&#x20;
* 이렇게 계산된 bounding box의 실제 값은 (x,y,w,h) 형태로 저장되며, 각 bounding box의 클래스(class) 정보는 one-hot 인코딩(one-hot encoding)으로 저장됩니다.



## load\_rle

```c
void load_rle(image im, int *rle, int n)
{
    int count = 0;
    int curr = 0;
    int i,j;
    for(i = 0; i < n; ++i){
        for(j = 0; j < rle[i]; ++j){
            im.data[count++] = curr;
        }
        curr = 1 - curr;
    }
    for(; count < im.h*im.w*im.c; ++count){
        im.data[count] = curr;
    }
}
```

함수 이름: load\_rle&#x20;

입력:&#x20;

* im: image 타입의 이미지
* rle: RLE 인코딩된 데이터
* n: RLE 데이터 길이&#x20;

동작:&#x20;

* RLE 인코딩된 데이터를 디코딩하여 이미지 데이터로 변환해주는 함수&#x20;

설명:

* RLE 인코딩된 데이터를 디코딩하여 이미지 데이터로 변환한다.
* 인코딩된 데이터(rle)는 0과 1로 번갈아 가며 나타나며, 이를 디코딩하여 이미지 데이터로 변환한다.
* curr 변수는 현재 값을 나타내며, rle\[i] 값만큼 curr 값을 반복하여 im.data에 저장한다.
* count 변수는 이미지 데이터의 현재 위치를 나타내며, 이미지 데이터의 크기(im.h_im.w_im.c)가 될 때까지 curr 값을 반복하여 im.data에 저장한다.



## or\_image

```c
void or_image(image src, image dest, int c)
{
    int i;
    for(i = 0; i < src.w*src.h; ++i){
        if(src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
    }
}
```

함수 이름: or\_image&#x20;

입력:&#x20;

* image src: 소스 이미지
* image dest: 대상 이미지
* int c: 채널

동작:&#x20;

* 소스 이미지에서 값이 0이 아닌 모든 픽셀은 대상 이미지에서 주어진 채널에 대해 1로 설정됩니다.&#x20;

설명:

* 소스 이미지(src)에서 값이 0이 아닌 모든 픽셀은 대상 이미지(dest)에서 주어진 채널(c)에 대해 1로 설정됩니다.
* 즉, 소스 이미지에서 흰색 부분은 dest 이미지에서 해당 채널에만 1로 나타나게 됩니다.
* 이 함수는 이미지를 이진화할 때 사용될 수 있습니다. 예를 들어 객체 검출과 같은 작업을 수행할 때, 물체를 검출하기 위해 입력 이미지를 이진화할 필요가 있습니다.



## exclusive\_image

```c
void exclusive_image(image src)
{
    int k, j, i;
    int s = src.w*src.h;
    for(k = 0; k < src.c-1; ++k){
        for(i = 0; i < s; ++i){
            if (src.data[k*s + i]){
                for(j = k+1; j < src.c; ++j){
                    src.data[j*s + i] = 0;
                }
            }
        }
    }
}
```

함수 이름: exclusive\_image

입력:&#x20;

* 이미지 구조체 포인터 src

동작:

* 입력으로 받은 src 이미지의 채널 중, 오직 하나의 채널에만 값이 있는 픽셀들만 남기고, 나머지 채널의 값을 0으로 만든다.

설명:

* src 이미지의 너비, 높이, 채널 정보를 사용하여 픽셀 데이터를 일렬로 나열한 후, 여러 채널 중에서 값이 있는 픽셀을 선택하여 그 외의 채널의 값을 모두 0으로 만든다.
* 예를 들어, 입력 이미지가 RGB 3채널로 구성되어 있다면, 빨간색 채널에만 값이 있는 픽셀들만 선택하여 나머지 채널의 값을 0으로 만든다. 이 과정을 초록색, 파란색 채널에 대해서도 반복한다.
* 최종적으로 입력 이미지는 하나의 채널만 값을 가지는 이진 이미지로 변환된다.



## bound\_image

```c
box bound_image(image im)
{
    int x,y;
    int minx = im.w;
    int miny = im.h;
    int maxx = 0;
    int maxy = 0;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            if(im.data[y*im.w + x]){
                minx = (x < minx) ? x : minx;
                miny = (y < miny) ? y : miny;
                maxx = (x > maxx) ? x : maxx;
                maxy = (y > maxy) ? y : maxy;
            }
        }
    }
    box b = {minx, miny, maxx-minx + 1, maxy-miny + 1};
    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
    return b;
}
```

함수 이름: bound\_image

입력:&#x20;

* image: 이미지

동작:&#x20;

* 이진화된 이미지에서 바운딩 박스 (bounding box)를 계산합니다. 바운딩 박스는 이미지 내에 있는 객체가 포함되어 있는 최소한의 사각형입니다.

설명:

* 바운딩 박스를 계산하기 위해, 이미지 내의 모든 픽셀을 탐색하면서, 픽셀의 값이 1 (흰색)인 경우에만 바운딩 박스를 계산합니다.
* 이미지의 너비와 높이를 이용해, 최소 x, y 좌표와 최대 x, y 좌표를 계산합니다.
* 계산된 좌표를 이용해, 바운



## fill\_truth\_iseg

```c
void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    int j;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);

        image mask = resize_image(sized, mw, mh);
        truth[i*(mw*mh+1)] = id;
        for(j = 0; j < mw*mh; ++j){
            truth[i*(mw*mh + 1) + 1 + j] = mask.data[j];
        }
        ++i;

        free_image(mask);
        free_image(sized);
        free(rle);
    }
    if(i < num_boxes) truth[i*(mw*mh+1)] = -1;
    fclose(file);
    free_image(part);
}
```

함수 이름: fill\_truth\_iseg

입력:

* path: 이미지 파일 경로
* num\_boxes: 이미지 내 객체의 수
* truth: 객체의 정보를 담을 배열
* classes: 클래스 수
* w, h: 이미지의 너비와 높이
* aug: 객체 증강(augmentation)을 위한 인자값들
* flip: 이미지를 수평으로 뒤집을지 여부
* mw, mh: 객체의 마스크(mask) 이미지의 너비와 높이

동작:

* 입력으로 받은 이미지 파일 경로(path)를 이용하여 해당 이미지의 객체 정보를 포함하는 파일을 열고, 객체의 정보를 truth 배열에 저장하는 함수
* 객체 정보를 포함하는 파일은 이미지 파일 경로에서 'images'를 'mask'로 변경하여 해당 경로에서 찾는다.
* 이미지 파일 경로가 JPEGImages를 포함하는 경우에도 'mask'로 변경한다.
* 객체 정보를 포함하는 파일은 이미지 파일 경로에서 확장자를 .txt로 변경한 파일이다.
* 파일을 열고, 객체의 정보를 한 줄씩 읽어와 객체의 마스크 이미지를 생성하고 크기를 조정한 뒤 truth 배열에 객체 정보를 저장한다.
* num\_boxes 수만큼의 객체 정보를 저장하거나, 파일을 다 읽어들인 경우에는 나머지 truth 배열 공간을 -1로 채운다.

설명:

* 이 함수는 객체 검출 모델에서 주로 사용되는 함수로, 이미지 내 객체의 정보를 포함하는 마스크(mask) 이미지를 생성하는 작업을 수행한다.
* 객체 검출 모델에서는 입력 이미지의 크기와 비율이 다양하게 존재하므로, 객체 증강(augmentation)을 수행하여 데이터를 다양하게 만들어 학습 성능을 높이는 것이 일반적이다.
* 객체 증강(augmentation)은 입력 이미지를 회전, 이동, 확대/축소, 수평/수직 반전 등의 변환을 가하여 여러 가지 형태의 데이터를 만드는 것을 의미한다.
* 이 함수에서는 객체 증강(augmentation)에 필요한 인자값들을 aug 인자로 받는다.
* 객체의 정보를 포함하는 파일은 COCO 데이터셋에서 사용되는 'Run Length Encoding' 방식을 이용하여 객체의 위치와 크기를 저장한 파일이다.
* 객체의 정보를 포함하는 파일에서 읽어들인 문자열은 read\_intlist 함수를 이용하여 RLE(Run Length Encoding) 방식으로 저장된 값을 읽어들인다.
* RLE 방식은 객체의 마스크(mask) 이미지를 압축하여 저장하는 방식으로, 객체 영역 내 픽셀들의 연속된 값의 길이를 저장하여 이미지 데이터를 효율적으로 저장할 수 있다.



## fill\_truth\_mask

```c
void fill_truth_mask(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);
        box b = bound_image(sized);
        if(b.w > 0){
            image crop = crop_image(sized, b.x, b.y, b.w, b.h);
            image mask = resize_image(crop, mw, mh);
            truth[i*(4 + mw*mh + 1) + 0] = (b.x + b.w/2.)/sized.w;
            truth[i*(4 + mw*mh + 1) + 1] = (b.y + b.h/2.)/sized.h;
            truth[i*(4 + mw*mh + 1) + 2] = b.w/sized.w;
            truth[i*(4 + mw*mh + 1) + 3] = b.h/sized.h;
            int j;
            for(j = 0; j < mw*mh; ++j){
                truth[i*(4 + mw*mh + 1) + 4 + j] = mask.data[j];
            }
            truth[i*(4 + mw*mh + 1) + 4 + mw*mh] = id;
            free_image(crop);
            free_image(mask);
            ++i;
        }
        free_image(sized);
        free(rle);
    }
    fclose(file);
    free_image(part);
}
```

함수 이름: fill\_truth\_mask&#x20;

입력:

* path: 이미지 경로
* num\_boxes: 이미지에 포함된 bounding box 수
* truth: bounding box 정보와 segmentation mask 정보를 담은 배열
* classes: 클래스 수
* w: 이미지 가로 길이
* h: 이미지 세로 길이
* aug: 이미지 augmentation 정보
* flip: 이미지를 수평으로 뒤집을지 여부
* mw: segmentation mask 가로 길이
* mh: segmentation mask 세로 길이

동작:&#x20;

* 이미지 경로에서 해당 이미지에 대한 segmentation mask 정보 파일을 찾아서 열고, bounding box 정보와 함께 truth 배열에 정보를 채운다.

설명:

* 이미지 경로에서 해당 이미지에 대한 segmentation mask 정보 파일 경로를 찾는다.
* 파일을 열고, 파일이 없을 경우 에러를 출력한다.
* 이미지의 일부분을 자른 이미지(part)를 만들고, 해당 이미지의 RLE 형태의 segmentation mask 정보를 읽어온다.
* part 이미지에 대해 augmentation을 적용한 후, flip 여부에 따라 이미지를 수평으로 뒤집는다.
* part 이미지에 대한 bounding box 정보를 얻어온다.
* bounding box가 있는 경우, 해당 영역을 crop하여 mask 이미지를 만들고, 이를 resize하여 mw x mh 크기의 segmentation mask로 변환한다.
* bounding box의 중심점과 크기, 그리고 segmentation mask 정보를 truth 배열에 저장한다.
* i를 증가시키고, part 이미지와 rle 배열을 해제한다.
* num\_boxes에 도달할 때까지 위의 과정을 반복하며, 파일을 닫고 part 이미지를 해제한다.



## fill\_truth\_detection

```c
void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);

    find_replace(labelpath, "raw", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if(count > num_boxes) count = num_boxes;
    float x,y,w,h;
    int id;
    int i;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if ((w < .001 || h < .001)) {
            ++sub;
            continue;
        }

        truth[(i-sub)*5+0] = x;
        truth[(i-sub)*5+1] = y;
        truth[(i-sub)*5+2] = w;
        truth[(i-sub)*5+3] = h;
        truth[(i-sub)*5+4] = id;
    }
    free(boxes);
}
```

함수 이름: fill\_truth\_detection

입력:

* path: 라벨 파일 경로를 포함하는 이미지 파일 경로
* num\_boxes: 이미지에서 가져올 박스 수
* truth: 실제 값 배열
* classes: 클래스 수
* flip: 이미지 뒤집기 여부
* dx, dy, sx, sy: 이미지 확대 및 이동에 대한 인자

동작:

* 라벨 파일을 읽어들여서 무작위로 섞은 후, 박스 위치를 수정한다.
* num\_boxes 개수만큼 박스를 가져와서 실제 값(truth) 배열에 저장한다.
* 저장된 값은 각각 x, y, w, h, id 로 이루어진다.

설명:

* 함수는 주어진 이미지 파일 경로와 라벨 파일 경로를 기반으로 라벨 파일을 읽어들인다.
* 박스 정보를 수정하여 데이터 증강을 수행한다.
* 이후 num\_boxes 개수만큼 박스를 가져와서 실제 값 배열에 저장한다.
* 저장된 값은 각각 박스의 x, y 좌표, 너비 w, 높이 h, 그리고 클래스 id 로 이루어진다.



## print\_letters

```c
#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}
```

함수 이름: print\_letters&#x20;

입력:

* float \*pred: 글자 인식 모델에서 예측된 문자 확률 값을 담은 1차원 배열
* int n: 예측된 문자 개수&#x20;

동작:

* pred 배열에서 n개의 문자를 각각 추출하여 예측된 문자 확률이 가장 높은 인덱스를 찾음
* 찾은 인덱스를 int\_to\_alphanum 함수를 통해 문자로 변환하여 출력
* 개행 문자를 출력하여 줄바꿈&#x20;

설명:

* 입력받은 pred 배열은 NUMCHARS(37)개의 글자에 대한 확률 값을 가짐
* 배열을 n개의 문자씩 잘라서 예측 결과를 출력함
* 출력된 문자열은 예측된 문자열을 의미하며, 해당 함수는 예측 결과를 쉽게 확인하기 위해 사용될 수 있음



## fill\_truth\_captcha

```c
void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}
```

함수 이름: fill\_truth\_captcha&#x20;

입력:

* char \*path: 레이블 파일 경로
* int n: 레이블 문자 수
* float \*truth: 레이블 데이터가 저장될 실수형 배열

동작:&#x20;

* 주어진 레이블 파일 경로에서 레이블 데이터를 추출하여 주어진 실수형 배열에 저장합니다.&#x20;
* 레이블 파일의 이름에서 문자를 추출하고, 각 문자의 인덱스를 계산하여 실수형 배열에서 해당 인덱스에 대한 값을 1로 설정합니다.&#x20;
* 레이블 파일 이름에서 문자를 추출하지 못한 경우에는 나머지 레이블 데이터의 값을 1로 설정합니다.

설명:&#x20;

* 이 함수는 captcha 모델에서 사용되며, captcha 이미지의 레이블을 추출하여 실수형 배열에 저장합니다.&#x20;
* 함수는 문자열에서 문자를 추출하고, 각 문자의 인덱스를 계산하여 실수형 배열에서 해당 인덱스에 대한 값을 1로 설정합니다.&#x20;
* 예를 들어, 문자 "A"의 인덱스는 0이며, 문자 "Z"의 인덱스는 25입니다.&#x20;
* 추출한 문자열의 길이가 주어진 레이블 문자 수보다 작을 경우, 나머지 레이블 데이터의 값을 1로 설정합니다.



## load\_data\_captcha

```c
data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}
```

함수 이름: load\_data\_captcha&#x20;

입력:

* char \*\*paths: 이미지 경로들을 담은 문자열 배열
* int n: 이미지의 개수
* int m: 무작위 샘플링할 이미지의 개수
* int k: 캡차의 글자 수
* int w: 이미지의 너비
* int h: 이미지의 높이&#x20;

동작:&#x20;

* 캡차 데이터를 로드하고, 입력 이미지와 정답(label)을 담은 행렬을 반환한다.
* get\_random\_paths() 함수를 이용하여 paths 배열에서 m개의 이미지를 무작위로 선택한다. m이 0인 경우, 모든 이미지를 사용한다.
* load\_image\_paths() 함수를 이용하여 paths 배열에 있는 이미지들을 로드하고, d.X에 할당한다.
* 정답을 저장할 d.y 행렬을 만들고, fill\_truth\_captcha() 함수를 이용하여 각 이미지의 정답(label)을 설정한다.
* 만약 m이 0이 아닌 경우, get\_random\_paths() 함수를 통해 할당된 paths 배열을 해제한다.
* 완성된 데이터를 담고 있는 data 타입의 구조체 d를 반환한다.&#x20;

설명:

* 입력으로 주어진 이미지들을 불러와 캡차 데이터를 만들어 반환하는 함수이다.
* 무작위 샘플링을 사용하여 데이터를 가져오는 것으로, 학습 데이터의 다양성을 높이고 overfitting을 방지할 수 있다.
* load\_image\_paths() 함수와 fill\_truth\_captcha() 함수를 사용하여 이미지 데이터와 정답 데이터를 생성하고 할당한다.



## load\_data\_captcha\_encode

```c
data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}
```

함수 이름: load\_data\_captcha\_encode

입력:

* paths: char 형식의 이차원 문자열 배열. 데이터 경로가 저장되어 있다.
* n: int 형식. paths 배열의 크기이자 로드할 이미지 수.
* m: int 형식. paths 배열에서 무작위로 선택할 이미지의 수. 0이면 모든 이미지를 사용한다.
* w: int 형식. 로드할 이미지의 가로 크기.
* h: int 형식. 로드할 이미지의 세로 크기.

동작:&#x20;

* paths 배열에서 이미지를 로드하여 반환하는 함수이다. m이 0이 아닌 경우, paths 배열에서 m개의 이미지를 무작위로 선택하여 로드한다.&#x20;
* 로드된 이미지를 이차원 행렬 형태로 저장하며, X와 y에 모두 할당한다. X와 y는 같은 값을 가진다.

설명:&#x20;

* 이 함수는 지정된 경로에서 이미지를 로드하여 반환하는 함수이다. 데이터 증강을 수행하지 않고, 그대로 로드한 이미지를 반환한다.&#x20;
* 이 함수는 주로 CAPTCHA 이미지를 분류하기 위한 데이터셋을 로드할 때 사용된다. 반환된 값은 data 구조체 형태로 반환된다.



## fill\_truth

```c
void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
            //printf("%s %s %d\n", path, labels[i], i);
        }
    }
    if(count != 1 && (k != 1 || count != 0)) printf("Too many or too few labels: %d, %s\n", count, path);
}
```

함수 이름: fill\_truth

입력:

* path: 이미지 파일 경로를 가리키는 문자열 포인터
* labels: 레이블 배열을 가리키는 문자열 포인터 배열
* k: 레이블 개수

동작:

* truth 배열에 해당 이미지의 레이블 값을 채움
* 해당 이미지 파일 경로(path)에 레이블(labels)이 포함되어 있으면 해당 레이블의 인덱스 위치의 truth 값을 1로 설정함
* 레이블(labels) 중 해당 이미지 파일 경로(path)에 포함된 레이블이 2개 이상 또는 0개인 경우 에러 메시지 출력

설명:

* 이 함수는 이미지 파일의 실제 레이블 값을 가져와 truth 배열에 저장하는 함수입니다.
* 해당 이미지 파일 경로(path)에서 레이블(labels) 중 포함된 레이블을 찾아 해당 레이블의 인덱스 위치의 truth 값을 1로 설정합니다.
* 이때 레이블(labels)의 개수(k)가 1인 경우 해당 이미지 파일 경로(path)에서 레이블이 발견되지 않는 경우도 허용됩니다.
* 그러나 레이블(labels)의 개수(k)가 2개 이상인 경우 해당 이미지 파일 경로(path)에서 레이블이 2개 이상 발견되거나 0개 발견되면 에러 메시지를 출력합니다.



## fill\_hierarchy

```c
void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}
```

함수 이름: fill\_hierarchy&#x20;

입력:

* float \*truth: 실제 레이블 값들의 배열 포인터
* int k: 레이블의 개수
* tree \*hierarchy: 레이블들의 계층 구조를 나타내는 트리 구조체 포인터&#x20;

동작:&#x20;

* 실제 레이블 값을 계층 구조에 맞게 변환하는 함수로, 계층 구조를 따라 부모 노드들도 모두 1로 채우고, 하위 그룹에 속한 레이블들이 하나도 존재하지 않으면 그룹 전체를 SECRET\_NUM 값으로 채움&#x20;

설명:&#x20;

* 입력으로 받은 truth 배열 포인터에 대해 계층 구조에 맞게 변환하는 함수입니다.&#x20;
* 계층 구조에서 하위 레이블은 상위 레이블을 모두 포함하므로, 상위 레이블이 존재하는 경우 해당 레이블도 1로 채웁니다.&#x20;
* 이후 하위 그룹에 속한 레이블들이 하나도 존재하지 않으면 그룹 전체를 SECRET\_NUM 값으로 채웁니다.



## load\_regression\_labels\_paths

```c
matrix load_regression_labels_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i,j;
    for(i = 0; i < n; ++i){
        char labelpath[4096];
        find_replace(paths[i], "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".BMP", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);
        find_replace(labelpath, ".JPG", ".txt", labelpath);
        find_replace(labelpath, ".JPeG", ".txt", labelpath);
        find_replace(labelpath, ".Jpeg", ".txt", labelpath);
        find_replace(labelpath, ".PNG", ".txt", labelpath);
        find_replace(labelpath, ".TIF", ".txt", labelpath);
        find_replace(labelpath, ".bmp", ".txt", labelpath);
        find_replace(labelpath, ".jpeg", ".txt", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".png", ".txt", labelpath);
        find_replace(labelpath, ".tif", ".txt", labelpath);

        FILE *file = fopen(labelpath, "r");
        for(j = 0; j < k; ++j){
            fscanf(file, "%f", &(y.vals[i][j]));
        }
        fclose(file);
    }
    return y;
}
```

함수 이름: load\_regression\_labels\_paths&#x20;

입력:

* paths: char \*\* 타입, 레이블 파일 경로가 들어있는 문자열 배열
* n: int 타입, 경로의 개수
* k: int 타입, 레이블의 개수

동작:&#x20;

* 레이블 파일 경로에서 레이블 값을 읽어와서 n x k 크기의 행렬 y에 저장한다.

설명:

* make\_matrix 함수를 이용하여 n x k 크기의 행렬 y를 생성한다.
* paths 배열에서 각 경로를 읽어서 해당 이미지에 대한 레이블 파일 경로를 생성한다.
* 생성된 레이블 파일 경로에서 레이블 값을 읽어와서 y에 저장한다.
* 이 과정을 모든 경로에 대해 반복하고, 최종적으로 y를 반환한다.



## load\_labels\_paths

```c
matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy){
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}
```

함수 이름: load\_labels\_paths

입력:

* paths: 이미지 경로 배열(char\*\*)
* n: 이미지 경로 배열(paths)의 길이(int)
* labels: 클래스 레이블 배열(char\*\*)
* k: 클래스 레이블 배열(labels)의 길이(int)
* hierarchy: 클래스 레이블 계층 구조(tree\*)

동작:&#x20;

* 이미지 경로 배열(paths)와 클래스 레이블 배열(labels)을 입력 받아 각 이미지의 클래스 레이블을 추출하여 k개의 클래스에 대한 one-hot 인코딩된 레이블 행렬(matrix) y를 생성한다.&#x20;
* 클래스 레이블 계층 구조가 있다면 해당 계층 구조(hierarchy)를 이용하여 y의 각 행에 대해 상위 클래스에 대한 one-hot 인코딩된 레이블로도 채운다.

설명:&#x20;

* 입력된 이미지 경로 배열(paths)와 클래스 레이블 배열(labels)을 이용하여 각 이미지의 클래스 레이블을 추출하고, 추출된 클래스 레이블 정보를 이용하여 k개의 클래스에 대한 one-hot 인코딩된 레이블 행렬(matrix) y를 생성한다.&#x20;
* 만약 클래스 레이블 계층 구조(hierarchy)가 있다면, 해당 계층 구조를 이용하여 y의 각 행에 대해 상위 클래스에 대한 one-hot 인코딩된 레이블로도 채운다.&#x20;
* 최종적으로 생성된 레이블 행렬(matrix) y를 반환한다.



## load\_tags\_paths

```c
matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    //int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "images", "labels", label);
        find_replace(label, ".jpg", ".txt", label);
        FILE *file = fopen(label, "r");
        if (!file) continue;
        //++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    //printf("%d/%d\n", count, n);
    return y;
}
```

함수 이름: load\_tags\_paths

입력:

* char \*\*paths: 이미지 파일 경로를 저장한 문자열 배열
* int n: 이미지 파일의 개수
* int k: 태그의 개수

동작:

* n개의 이미지 파일 경로를 받아서 해당 이미지 파일에 대한 태그를 읽어들여서 k차원의 one-hot encoding된 벡터로 변환하여 matrix 타입의 y에 저장한다.
* 파일을 열지 못한 경우(태그 파일이 없는 경우) 건너뛴다.

설명:

* 주어진 이미지 파일 경로 paths에서 ".jpg"를 ".txt"로 바꾸어서 해당 이미지에 대한 태그 파일 경로 label을 생성한다.
* 해당 label 파일을 열고, 한 줄씩 읽으면서 태그가 있으면 one-hot encoding된 벡터 y.vals\[i]에 저장한다.
* 이미지 파일이 존재하지 않거나, 파일을 열지 못한 경우, 해당 이미지 파일은 건너뛴다.
* 최종적으로 n개의 이미지에 대한 태그들이 one-hot encoding된 matrix y를 반환한다.



## get\_labels

```c
char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}
```

함수 이름: get\_labels

입력:&#x20;

* filename: 문자열

동작:&#x20;

* 주어진 filename으로부터 이미지 파일들에 대응하는 라벨 파일들의 경로들을 가져와서, 해당 경로들을 문자열 배열로 변환한 후 반환함.

설명:

* get\_paths 함수를 호출하여 주어진 filename에서 이미지 파일들의 경로들을 가져옴.
* 가져온 경로들을 이용하여 각 이미지 파일에 대응하는 라벨 파일들의 경로를 생성함.
* 생성된 라벨 파일 경로들을 문자열 배열로 변환하여 반환함.
* 함수 내부에서 사용된 list\_to\_array와 free\_list 함수는 주어진 연결 리스트를 배열로 변환하고, 변환된 배열을 사용한 후 메모리 해제를 수행하는 함수들임.



## free\_data

```c
void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}
```

함수 이름: free\_data

입력:&#x20;

* data: data 구조체

동작:&#x20;

* 주어진 data 구조체의 메모리를 해제함. 만약 d 구조체가 shallow이 아니면, X와 y의 메모리를 해제하고, shallow이면 vals 배열을 해제함.

설명:

* data 구조체는 입력 데이터를 담는 구조체임
* X는 입력 데이터를 담고 있는 matrix 구조체
* y는 출력 데이터(정답)를 담고 있는 matrix 구조체
* shallow은 X와 y의 vals 배열을 공유하고 있는지 여부를 나타냄
* 만약 shallow이 아니면, X와 y의 vals 배열을 개별적으로 할당하여 사용하고 있으므로, 메모리 해제 시에는 각각의 메모리를 해제해야 함
* 하지만 shallow이면 X와 y가 같은 vals 배열을 공유하고 있으므로, vals 배열만 해제하면 됨.



## get\_segmentation\_image

```c
image get_segmentation_image(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}
```

함수 이름: load\_data\_mask

입력:

* n (int): 데이터셋에 있는 이미지 수
* paths (char \*\*): 이미지 파일 경로 배열
* m (int): 무작위로 선택할 이미지의 수
* w (int): 이미지 가로 길이
* h (int): 이미지 세로 길이
* classes (int): 객체 클래스 수
* boxes (int): 하나의 이미지에서 처리할 객체 수
* coords (int): 객체의 좌표 수
* min (int): 이미지 크기를 무작위로 변형하기 위한 최소 비율
* max (int): 이미지 크기를 무작위로 변형하기 위한 최대 비율
* angle (float): 이미지 회전 각도 범위
* aspect (float): 이미지 크기 비율 범위
* hue (float): 이미지 색상 변형 범위
* saturation (float): 이미지 채도 변형 범위
* exposure (float): 이미지 노출 변형 범위

동작:

* 무작위 이미지 경로를 선택하고 무작위 증강 기법을 적용하여 이미지 데이터를 로드하고, 레이블 데이터를 만든다.
* 이미지 데이터와 레이블 데이터를 담은 구조체를 반환한다.

설명:

* 입력으로 받은 이미지 경로에서 n개의 이미지를 로드한다.
* m이 0보다 크면, 이미지 경로 배열에서 무작위로 m개의 이미지를 선택한다.
* 로드한 이미지 데이터를 무작위로 증강하여 크기, 회전, 비율, 색조, 채도, 노출 등의 변형을 적용한다.
* 이미지 데이터를 로드하고 증강하는 동안 각 이미지에 대한 레이블을 만든다.
* 레이블 데이터는 각 객체마다 좌표값과 객체의 클래스를 담은 벡터로 이루어져 있으며, 모든 객체의 벡터는 하나의 행렬로 묶여 반환된다.
* 최종적으로, 로드한 이미지 데이터와 레이블 데이터를 담은 data 구조체를 반환한다.



## get\_segmentation\_image2

```c
image get_segmentation_image2(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes+1);
    int i;
    for(i = 0; i < w*h; ++i){
        mask.data[w*h*classes + i] = 1;
    }
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        for(i = 0; i < w*h; ++i){
            if(part.data[i]) mask.data[w*h*classes + i] = 0;
        }
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}
```





## load\_data\_seg

```
data load_data_seg(int n, char **paths, int m, int w, int h, int classes, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, int div)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    d.y.rows = n;
    d.y.cols = h*w*classes/div/div;
    d.y.vals = calloc(d.X.rows, sizeof(float*));

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        image mask = get_segmentation_image(random_paths[i], orig.w, orig.h, classes);
        //image mask = make_image(orig.w, orig.h, classes+1);
        image sized_m = rotate_crop_image(mask, a.rad, a.scale/div, a.w/div, a.h/div, a.dx/div, a.dy/div, a.aspect);

        if(flip) flip_image(sized_m);
        d.y.vals[i] = sized_m.data;

        free_image(orig);
        free_image(mask);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}
```





## load\_data\_iseg

```
data load_data_iseg(int n, char **paths, int m, int w, int h, int classes, int boxes, int div, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (((w/div)*(h/div))+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, "image");

        fill_truth_iseg(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, w/div, h/div);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}
```





## load\_data\_mask

```c
data load_data_mask(int n, char **paths, int m, int w, int h, int classes, int boxes, int coords, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (coords+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, "image");

        fill_truth_mask(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, 14, 14);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}
```

함수 이름: load\_data\_mask

입력:

* n (int): 데이터의 개수
* paths (char \*\*): 이미지 파일 경로가 들어있는 문자열 배열
* m (int): paths 배열의 길이
* w (int): 리사이즈할 이미지의 가로 길이
* h (int): 리사이즈할 이미지의 세로 길이
* classes (int): 객체 클래스의 개수
* boxes (int): 이미지 당 객체의 최대 개수
* coords (int): 객체 좌표의 개수
* min (int): 객체가 차지하는 최소 면적
* max (int): 객체가 차지하는 최대 면적
* angle (float): 회전 각도 범위
* aspect (float): 가로 세로 비율 범위
* hue (float): 색상 변화 범위
* saturation (float): 채도 변화 범위
* exposure (float): 밝기 변화 범위

동작:

* 입력으로 들어온 이미지 파일 경로에서 이미지를 읽어와 데이터를 생성
* 데이터는 resized 및 distorted된 이미지와 객체 좌표 정보를 담은 행렬로 구성됨

설명:

* 입력된 n개의 이미지 파일에서 데이터를 읽어옴
* 각 이미지마다 객체 좌표 정보를 담은 행렬을 생성하며, 행렬의 크기는 (coords+1)\*boxes임
* 객체의 좌표 정보는 (x, y, w, h) 형태로 저장됨
* 객체의 최소 면적과 최대 면적을 지정하여 이 범위 내에서 무작위로 객체의 크기를 조절하며, 회전과 가로 세로 비율도 무작위로 변형함
* 색상, 채도, 밝기를 무작위로 변형하여 데이터를 augment함



## load\_data\_region

```c
data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    int k = size*size*(5+classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = rand_uniform(-dw, dw);
        int pright = rand_uniform(-dw, dw);
        int ptop   = rand_uniform(-dh, dh);
        int pbot   = rand_uniform(-dh, dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = rand()%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}
```

함수 이름: load\_data\_region&#x20;

입력:

* int n: 데이터셋의 이미지 개수
* char \*\*paths: 이미지 파일 경로 배열
* int m: 이미지 파일 경로 배열의 길이
* int w: 리사이즈할 이미지의 가로 크기
* int h: 리사이즈할 이미지의 세로 크기
* int size: YOLO 네트워크에서 사용되는 그리드 셀의 크기
* int classes: 클래스 개수
* float jitter: 이미지를 잘라내기 위한 임의의 jittering 크기
* float hue: 이미지 색상 조정을 위한 hue 변화 비율
* float saturation: 이미지 색상 조정을 위한 saturation 변화 비율
* float exposure: 이미지 색상 조정을 위한 exposure 변화 비율

동작:&#x20;

* 입력으로 받은 이미지 파일 경로 배열에서 이미지를 읽어들인 후, 해당 이미지를 jittering하고 리사이즈한 뒤 YOLO 네트워크에서 사용할 수 있는 형태로 변환하여 반환하는 함수입니다.&#x20;
* 반환값으로는 data 구조체가 사용되며, 이 구조체에는 리사이즈 및 변환된 이미지 데이터와 해당 이미지에 대한 ground truth 정보가 포함됩니다.

설명:&#x20;

* 이 함수는 YOLO 네트워크를 학습시키기 위한 데이터를 로드하는 함수입니다.&#x20;
* 입력으로 데이터셋의 이미지 개수, 이미지 파일 경로 배열, 이미지 리사이즈 크기, YOLO 네트워크에서 사용되는 그리드 셀 크기, 클래스 개수 등을 받습니다.&#x20;
* 함수는 입력으로 받은 이미지 파일 경로 배열에서 이미지를 읽어들인 후, 해당 이미지를 jittering하고 리사이즈합니다.&#x20;
* 이후에는 해당 이미지에 대한 ground truth 정보를 생성하고, 리사이즈한 이미지 데이터와 함께 data 구조체에 저장하여 반환합니다.&#x20;
* 이 구조체는 YOLO 네트워크에서 학습에 사용됩니다.



## load\_data\_compare

```c
data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = calloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char imlabel1[4096];
        char imlabel2[4096];
        find_replace(paths[i*2],   "imgs", "labels", imlabel1);
        find_replace(imlabel1, "jpg", "txt", imlabel1);
        FILE *fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        find_replace(paths[i*2+1], "imgs", "labels", imlabel2);
        find_replace(imlabel2, "jpg", "txt", imlabel2);
        FILE *fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }

        for (j = 0; j < classes; ++j){
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5){
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            } else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5){
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            } else {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}
```

함수 이름: load\_data\_compare

입력:

* n (int): 이미지 쌍의 수
* paths (char \*\*): 이미지 경로 배열
* m (int): 이미지 경로 배열의 길이, 0이면 경로 배열은 n개의 이미지 경로를 포함
* classes (int): 분류 클래스 수
* w (int): 이미지의 가로 크기
* h (int): 이미지의 세로 크기

동작:&#x20;

* 이미지 경로에서 랜덤으로 이미지 쌍을 가져온 후 각 이미지를 읽어들인다.&#x20;
* 읽어들인 이미지들을 하나의 데이터 행렬로 만든다.&#x20;
* 각 이미지 쌍에 대해 ground truth(정답 레이블)을 가져와, 적절한 형식으로 변환한 후, 데이터 행렬과 함께 리턴한다.

설명:&#x20;

* load\_data\_compare 함수는 이미지 쌍 데이터를 읽어들이고 ground truth(정답 레이블)을 가져와, 적절한 형식으로 변환하여 리턴하는 함수이다.&#x20;
* 함수의 입력으로는 이미지 쌍의 수, 이미지 경로 배열, 이미지 경로 배열의 길이, 분류 클래스 수, 이미지의 가로 크기, 이미지의 세로 크기가 들어간다.
* 이미지 경로 배열에서 랜덤으로 이미지 쌍을 가져오고, 각 이미지를 읽어들인 후, 하나의 데이터 행렬로 만든다.&#x20;
* ground truth(정답 레이블)을 가져와, 적절한 형식으로 변환한 후, 데이터 행렬과 함께 리턴한다.&#x20;
* 이 때, classes의 수만큼 ground truth(정답 레이블)을 만들어야 하므로, 2\*(classes) 크기의 행렬을 만든다.&#x20;
* 이 행렬의 짝수 인덱스에는 첫 번째 이미지의 ground truth(정답 레이블), 홀수 인덱스에는 두 번째 이미지의 ground truth(정답 레이블)을 저장한다. 각 ground truth(정답 레이블)은 이미지 쌍의 id와 iou 값으로 이루어져 있다. 이 값들을 적절한 형식으로 변환하고, 데이터 행렬과 함께 리턴한다.



## load\_data\_swag

```c
data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = rand()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*90;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = rand()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}
```

함수 이름: load\_data\_swag

입력:

* paths: 이미지 경로 배열, char\*\*
* n: 이미지 경로 배열의 크기, int
* classes: 객체 클래스 수, int
* jitter: 이미지 자르기 정도, float

동작:

* paths에서 임의의 이미지를 선택하고 해당 이미지를 읽어들인다.
* 선택된 이미지를 jitter를 이용하여 자르고 크기를 변경한다.
* 자른 이미지에서 랜덤하게 crop을 수행하고, crop된 이미지를 flip한다.
* crop 및 flip된 이미지와 해당 이미지의 label을 반환한다.

설명:

* load\_data\_swag는 주어진 이미지 경로 중 임의의 이미지를 선택하여 자르고 크기를 변경하며, 그에 따른 label을 반환하는 함수이다.
* 입력으로는 이미지 경로 배열, 이미지 경로 배열의 크기, 객체 클래스 수, 이미지 자르기 정도를 받는다.
* 먼저, paths에서 랜덤하게 이미지를 선택하고 선택된 이미지를 읽어들인다.
* 그 다음, jitter를 이용하여 이미지를 자르고 크기를 변경한다.
* 자른 이미지 중 랜덤하게 crop을 수행하고, crop된 이미지를 flip한다.
* crop 및 flip된 이미지와 해당 이미지의 label을 matrix 형태로 반환한다.



## load\_data\_detection

```c
data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, 5*boxes);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);

        float dw = jitter * orig.w;
        float dh = jitter * orig.h;

        float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
        //float scale = rand_uniform(.25, 2);
        float scale = 1;

        float nw, nh;

        if(new_ar < 1){
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }

        float dx = rand_uniform(0, w - nw);
        float dy = rand_uniform(0, h - nh);

        place_image(orig, nw, nh, dx, dy, sized);

        random_distort_image(sized, hue, saturation, exposure);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        d.X.vals[i] = sized.data;


        fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h);

        free_image(orig);
    }
    free(random_paths);
    return d;
}
```

함수 이름: load\_data\_detection&#x20;

입력:

* n: int 형태의 데이터 개수
* paths: char\*\* 형태의 데이터 경로 배열
* m: int 형태의 데이터 경로 개수
* w: int 형태의 이미지 가로 크기
* h: int 형태의 이미지 세로 크기
* boxes: int 형태의 박스 개수
* classes: int 형태의 클래스 개수
* jitter: float 형태의 이미지 jittering 범위
* hue: float 형태의 이미지 hue 변화 범위
* saturation: float 형태의 이미지 saturation 변화 범위
* exposure: float 형태의 이미지 exposure 변화 범위

동작:&#x20;

* 입력된 데이터 경로에서 이미지를 불러와서 data 구조체를 만들어 반환하는 함수입니다.&#x20;
* 각 이미지는 jittering, hue, saturation, exposure 변화 등의 처리를 거칩니다.&#x20;
* 또한 각 이미지에서는 ground truth 정보를 추출하고, 이를 y 값으로 저장합니다.

설명:&#x20;

* 입력된 데이터 경로에서 이미지를 불러와서 각 이미지에 대해 다음과 같은 동작을 수행합니다.

1. 불러온 이미지를 jittering하여 크기를 랜덤하게 변화시킵니다.
2. 변환된 이미지를 랜덤하게 flip 합니다.
3. flip된 이미지를 hue, saturation, exposure 변화를 주어 random distort 합니다.
4. ground truth 정보를 추출하여 y 값으로 저장합니다.
5. 처리된 이미지를 data 구조체 X 값에 할당합니다.
6. 처리된 data 구조체를 반환합니다.



## load\_thread

```c
void *load_thread(void *ptr)
{
    //printf("Loading data: %d\n", rand());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == REGRESSION_DATA){
        *a.d = load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center);
    } else if (a.type == SUPER_DATA){
        *a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
    } else if (a.type == WRITING_DATA){
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    } else if (a.type == ISEG_DATA){
        *a.d = load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == INSTANCE_DATA){
        *a.d = load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == SEGMENTATION_DATA){
        *a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.scale);
    } else if (a.type == REGION_DATA){
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == DETECTION_DATA){
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == SWAG_DATA){
        *a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    } else if (a.type == COMPARE_DATA){
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    } else if (a.type == LETTERBOX_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    } else if (a.type == TAG_DATA){
        *a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}
```

함수 이름: load\_thread

입력:&#x20;

* void \*ptr (void 포인터)

동작:&#x20;

* 스레드에서 데이터를 로드하는 함수입니다.&#x20;
* load\_args 구조체에 저장된 입력 매개 변수를 통해 어떤 유형의 데이터를 로드할지 결정하고 해당 데이터를 로드합니다.&#x20;
* 로드된 데이터는 메모리에 할당되고, load\_args 구조체에서 전달된 포인터에 저장됩니다.

설명:

* load\_args 구조체는 다양한 데이터 유형에 대한 입력 매개 변수를 저장합니다.
* 입력 매개 변수는 다양한 형태의 데이터 로드 함수에 전달됩니다.
* load\_args 구조체에서 포인터로 전달된 데이터는 해당 유형에 대한 데이터 구조체에 저장됩니다.
* 함수가 끝나면, load\_args 구조체가 동적으로 할당 해제됩니다.



## load\_data\_in\_thread

```c
pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}
```

함수 이름: load\_data\_in\_thread

입력:&#x20;

* args: 데이터 로드에 필요한 인자들을 담은 구조체 포인터

동작:&#x20;

* 데이터를 로드하는 쓰레드를 생성하고, 생성된 쓰레드를 반환함.

설명:

* 데이터 로드를 위해 필요한 인자들을 담은 구조체 포인터를 인자로 받음.
* 인자로 받은 구조체를 동적으로 할당한 메모리에 복사함.
* 생성된 쓰레드를 반환함.



## load\_threads

```c
void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);
    data *buffers = calloc(args.threads, sizeof(data));
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    free(threads);
    return 0;
}
```

함수 이름: load\_threads

입력:&#x20;

* void 포인터 ptr

동작:&#x20;

* 주어진 입력 인수에 따라 이미지 데이터를 비동기적으로 로드하고, 이를 스레드 수에 따라 분할하여 병렬 처리한다.&#x20;
* 각 스레드에서 로드된 데이터는 개별적인 데이터 구조체에 저장되고, 이들을 합쳐서 하나의 데이터 구조체로 반환한다.

설명:

* args: 이미지 데이터를 로드하기 위한 매개변수를 담고 있는 구조체
* out: 로드된 이미지 데이터가 저장될 구조체
* total: 전체 이미지 데이터 개수
* buffers: 스레드마다 로드된 이미지 데이터가 저장될 구조체 배열
* threads: 생성된 스레드들의 배열
* load\_data\_in\_thread: 스레드에서 실행되는 함수, 로드된 이미지 데이터를 buffers에 저장한다.
* concat\_datas: buffers에 저장된 이미지 데이터들을 하나의 데이터 구조체로 병합한다.
* free\_data: 이미지 데이터를 메모리에서 해제한다.



## load\_data\_blocking

```c
void load_data_blocking(load_args args)
{
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    load_thread(ptr);
}
```

함수 이름: load\_data\_blocking

입력:&#x20;

* args: load\_args (구조체 포인터)

동작:&#x20;

* 입력으로 받은 구조체 포인터 args의 정보를 이용해 데이터를 로드하는 작업을 수행한다.&#x20;
* 이 함수는 스레드를 사용하지 않고 블로킹 방식으로 데이터를 로드한다.

설명:&#x20;

* load\_args 구조체 포인터를 입력으로 받아서 데이터를 로드하는 작업을 수행하는 함수이다.&#x20;
* 이 함수는 스레드를 사용하지 않고 블로킹 방식으로 데이터를 로드하므로, 함수가 실행되는 동안 다른 작업은 수행할 수 없다.



## load\_data

```c
pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}
```

함수 이름: load\_data

입력:&#x20;

* args: load\_args 구조체

동작:&#x20;

* 새로운 스레드를 생성하여 load\_threads 함수를 실행하는데, 이때 load\_args 구조체를 인자로 넘겨준다.&#x20;
* 생성된 스레드의 ID를 반환한다.

설명:

* load\_args: 이미지 파일 경로, 라벨 파일 경로, 배치 크기 등을 담고 있는 구조체
* pthread\_t: POSIX 스레드 ID를 나타내는 데이터 타입
* calloc: 동적 할당된 메모리를 초기화하는 함수
* pthread\_create: POSIX 스레드를 생성하는 함수
* load\_threads: 이미지와 라벨 데이터를 읽어들이는 함수 (멀티스레드로 구현)
* error: 오류 메시지를 출력하고 프로그램을 종료하는 함수



## load\_data\_writing

```c
data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}
```

함수 이름: load\_data\_writing

입력:

* char \*\*paths: 이미지 파일 경로 배열
* int n: 경로 배열의 길이
* int m: 무작위 샘플링을 할 때 사용하는 샘플링 수
* int w: 이미지의 너비
* int h: 이미지의 높이
* int out\_w: 출력 이미지의 너비
* int out\_h: 출력 이미지의 높이

동작:

* paths 배열에서 이미지를 로드하여 X 행렬에 저장하고, "-label.png"로 끝나는 파일명을 찾아 y 행렬에 저장
* m이 0이 아니면, paths 배열에서 m개의 무작위 경로를 선택하여 로드
* X와 y는 float 형식의 이미지 데이터를 저장하는 구조체
* y는 gray-scale 이미지 데이터를 저장하는 구조체

설명:

* 이 함수는 "writing" 이미지 데이터를 로드하는 데 사용됩니다.
* 입력 이미지와 출력 이미지 모두 png 형식이며, 출력 이미지는 입력 이미지와 동일한 크기가 아닐 수 있습니다.
* 입력 이미지의 레이블은 입력 이미지 파일 이름과 "-label.png"를 붙인 파일에 저장되어 있습니다.



## load\_data\_old

```c
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0);
    if(m) free(paths);
    return d;
}
```

함수 이름: load\_data\_old&#x20;

입력:

* paths: char\*\* 타입. 이미지 파일 경로 배열.
* n: int 타입. 이미지 파일 경로 개수.
* m: int 타입. 무작위 샘플링할 이미지 파일 경로 개수. (0이면 샘플링하지 않음)
* labels: char\*\* 타입. 이미지 파일에 대한 레이블 배열.
* k: int 타입. 레이블 개수.
* w: int 타입. 이미지 가로 길이.
* h: int 타입. 이미지 세로 길이.

동작:&#x20;

* 이미지 파일과 해당 이미지에 대한 레이블을 로드하여 data 구조체에 저장한다.

설명:&#x20;

* 이미지 파일과 해당 이미지에 대한 레이블을 로드하는 함수이다.&#x20;
* 이미지 파일 경로와 레이블 배열, 그리고 이미지의 크기를 입력으로 받는다.&#x20;
* 만약 무작위 샘플링할 이미지 파일 경로 개수(m)가 0이 아니면 get\_random\_paths 함수를 사용하여 m개의 이미지 파일 경로를 무작위로 선택한다.&#x20;
* 그리고 load\_image\_paths 함수를 사용하여 이미지 파일을 로드하고, load\_labels\_paths 함수를 사용하여 해당 이미지에 대한 레이블을 로드한다.&#x20;
* 마지막으로, 로드한 이미지와 레이블을 data 구조체에 저장하여 반환한다.



## load\_data\_super

```c
data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
   if(m) paths = get_random_paths(paths, n, m);
   data d = {0};
   d.shallow = 0;

   int i;
   d.X.rows = n;
   d.X.vals = calloc(n, sizeof(float*));
   d.X.cols = w*h*3;

   d.y.rows = n;
   d.y.vals = calloc(n, sizeof(float*));
   d.y.cols = w*scale * h*scale * 3;

   for(i = 0; i < n; ++i){
       image im = load_image_color(paths[i], 0, 0);
       image crop = random_crop_image(im, w*scale, h*scale);
       int flip = rand()%2;
       if (flip) flip_image(crop);
       image resize = resize_image(crop, w, h);
       d.X.vals[i] = resize.data;
       d.y.vals[i] = crop.data;
       free_image(im);
   }

   if(m) free(paths);
   return d;
}
```

함수 이름: load\_data\_super

입력:

* char \*\*paths: 이미지 파일 경로가 담긴 문자열 배열
* int n: 데이터셋 크기
* int m: 이미지에서 임의로 선택할 이미지 수
* int w: 입력 이미지 너비
* int h: 입력 이미지 높이
* int scale: 상위 해상도 이미지와 하위 해상도 이미지 비율

동작:&#x20;

* 주어진 이미지 경로에서 이미지를 불러와서 상위 해상도 이미지를 생성하는 데이터셋을 로드합니다.&#x20;
* 불러온 이미지는 임의의 크기로 자르고, 뒤집어서 상하좌우 대칭을 만들고, 상위 해상도와 하위 해상도 이미지를 만듭니다. 생성된 이미지는 데이터셋의 입력 값(X)과 목표 값(y)으로 사용됩니다.

설명:

* data 구조체를 초기화하고, X와 y 값이 할당된 메모리를 가리키는 포인터 변수를 설정합니다.
* 입력 이미지의 개수(n)를 X와 y 행의 크기로 지정합니다.
* X와 y 열의 크기를 각각 입력 이미지의 너비(w), 높이(h), 채널 수(3)의 곱과 하위 해상도 이미지 크기(w_scale, h_scale, 3)의 곱으로 지정합니다.
* 입력된 이미지 경로에서 이미지를 불러옵니다.
* 불러온 이미지를 임의의 크기로 자르고, 상하좌우 대칭을 만듭니다.
* 상위 해상도 이미지와 하위 해상도 이미지를 생성합니다.
* 생성된 이미지를 각각 X와 y 값에 할당합니다.
* 모든 이미지를 불러온 후, 메모리를 해제하고, 생성된 데이터셋을 반환합니다.



## load\_data\_regression

```c
data load_data_regression(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_regression_labels_paths(paths, n, k);
    if(m) free(paths);
    return d;
}
```

함수 이름: load\_data\_regression

입력:

* paths: char 타입의 경로 배열
* n: int 타입의 데이터 개수
* m: int 타입의 mini-batch 개수
* k: int 타입의 라벨 개수
* min: int 타입의 이미지 픽셀 최소값
* max: int 타입의 이미지 픽셀 최대값
* size: int 타입의 이미지 크기
* angle: float 타입의 이미지 회전 각도
* aspect: float 타입의 이미지 가로 세로 비율
* hue: float 타입의 이미지 색조 변경 값
* saturation: float 타입의 이미지 채도 변경 값
* exposure: float 타입의 이미지 노출 변경 값

동작:&#x20;

* 회귀 분석용 데이터를 로드하고, 이미지에 대한 augmentation 작업을 수행함

설명:&#x20;

* 이 함수는 입력받은 경로 배열(paths)를 이용하여 회귀 분석용 데이터를 로드합니다.&#x20;
* 만약 mini-batch 개수(m)가 0보다 크면, 경로 배열을 이용하여 랜덤한 mini-batch 경로를 가져옵니다.&#x20;
* 그리고 data 구조체를 초기화하고, 입력 이미지에 대한 augmentation 작업을 수행합니다. 마지막으로 로드한 데이터와 라벨을 data 구조체에 할당하고 반환합니다.



## select\_data

```c
data select_data(data *orig, int *inds)
{
    data d = {0};
    d.shallow = 1;
    d.w = orig[0].w;
    d.h = orig[0].h;

    d.X.rows = orig[0].X.rows;
    d.y.rows = orig[0].X.rows;

    d.X.cols = orig[0].X.cols;
    d.y.cols = orig[0].y.cols;

    d.X.vals = calloc(orig[0].X.rows, sizeof(float *));
    d.y.vals = calloc(orig[0].y.rows, sizeof(float *));
    int i;
    for(i = 0; i < d.X.rows; ++i){
        d.X.vals[i] = orig[inds[i]].X.vals[i];
        d.y.vals[i] = orig[inds[i]].y.vals[i];
    }
    return d;
}
```

함수 이름: tile\_data&#x20;

입력:

* data orig: 변환할 데이터셋
* int divs: 원본 이미지를 나눌 수
* int size: 나눈 이미지의 크기

동작:

* 원본 이미지를 divs x divs 개수로 나눈다.
* 각 나눈 이미지를 size x size 크기로 조정한다.
* 조정된 이미지를 가지고 새로운 데이터셋을 만든다.

설명:&#x20;

* 이 함수는 이미지를 나누고 조정하여 새로운 데이터셋을 만들어주는 함수이다.&#x20;
* orig에는 원본 이미지가 들어오며, divs는 원본 이미지를 얼마나 나눌지를 결정하고 size는 각 나눈 이미지의 크기를 결정한다.
* 함수는 divs x divs 개수로 원본 이미지를 나누고, 각 나눈 이미지를 size x size 크기로 조정한다. 그리고 각각 조정된 이미지를 가지고 새로운 데이터셋을 만든다.&#x20;
* 이 때, shallow 값은 0으로 설정되며, d.X.vals와 d.y는 원본 데이터셋에서 복사된다.
* 최종적으로 변환된 데이터셋은 data 포인터 배열 형태로 리턴된다.



## tile\_data

```c
data *tile_data(data orig, int divs, int size)
{
    data *ds = calloc(divs*divs, sizeof(data));
    int i, j;
#pragma omp parallel for
    for(i = 0; i < divs*divs; ++i){
        data d;
        d.shallow = 0;
        d.w = orig.w/divs * size;
        d.h = orig.h/divs * size;
        d.X.rows = orig.X.rows;
        d.X.cols = d.w*d.h*3;
        d.X.vals = calloc(d.X.rows, sizeof(float*));

        d.y = copy_matrix(orig.y);
#pragma omp parallel for
        for(j = 0; j < orig.X.rows; ++j){
            int x = (i%divs) * orig.w / divs - (d.w - orig.w/divs)/2;
            int y = (i/divs) * orig.h / divs - (d.h - orig.h/divs)/2;
            image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[j]);
            d.X.vals[j] = crop_image(im, x, y, d.w, d.h).data;
        }
        ds[i] = d;
    }
    return ds;
}
```





## resize\_data

```c
data resize_data(data orig, int w, int h)
{
    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;
    int i;
    d.X.rows = orig.X.rows;
    d.X.cols = w*h*3;
    d.X.vals = calloc(d.X.rows, sizeof(float*));

    d.y = copy_matrix(orig.y);
#pragma omp parallel for
    for(i = 0; i < orig.X.rows; ++i){
        image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[i]);
        d.X.vals[i] = resize_image(im, w, h).data;
    }
    return d;
}
```

함수 이름: resize\_data

입력:

* data orig: 원래 데이터
* int w: 가로 크기
* int h: 세로 크기

동작:

* orig의 이미지 데이터 크기를 w x h 크기로 조정(resize)
* orig의 라벨 데이터는 그대로 복사하여 반환

설명:

* 입력으로 받은 orig 데이터의 이미지 데이터 크기를 w x h 크기로 조정하여 새로운 데이터 d를 생성하여 반환하는 함수입니다.
* 새로 생성된 데이터 d의 shallow 멤버 변수는 0으로 설정됩니다.
* orig의 라벨 데이터는 그대로 복사되어 반환됩니다.
* orig의 이미지 데이터를 각각 float\_to\_image 함수로 이미지로 변환한 후, resize\_image 함수로 크기를 조정합니다.
* 생성된 이미지 데이터를 새로운 데이터 d의 이미지 데이터로 할당합니다.



## load\_data\_augment

```c
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.w=size;
    d.h=size;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy);
    if(m) free(paths);
    return d;
}
```

함수 이름: load\_data\_augment

입력:

* paths: 이미지 파일 경로를 담은 문자열 배열
* n: 이미지 파일 경로 개수
* m: 무작위 이미지 선정 수 (선택 사항)
* labels: 클래스 레이블을 담은 문자열 배열
* k: 클래스 수
* hierarchy: 클래스 계층 구조를 저장한 tree 구조체
* min: 이미지 크기를 조절할 때 최소 크기
* max: 이미지 크기를 조절할 때 최대 크기
* size: 출력 이미지 크기
* angle: 이미지 회전 각도 범위
* aspect: 이미지 비율 변경 비율 범위
* hue: 이미지 색상 범위
* saturation: 이미지 채도 범위
* exposure: 이미지 밝기 범위
* center: 이미지 중앙에서 자를 영역 크기 (선택 사항)

동작:

* 지정된 경로에서 이미지를 불러옴
* 지정된 크기로 이미지 크기를 조절하고 지정된 augmentation을 수행하여 데이터 증강을 함
* 클래스 레이블을 불러옴
* 계층 구조가 지정되어 있으면 계층 구조를 이용하여 클래스 레이블을 적절하게 수정함
* 결과 데이터셋을 반환함

설명:&#x20;

* 주어진 이미지 경로에서 이미지를 불러오고 augmentation을 수행하여 데이터를 증강시키는 함수입니다.&#x20;
* 이 함수는 클래스 레이블과 계층 구조를 이용하여 클래스 레이블을 적절하게 수정합니다.&#x20;
* 함수의 입력으로는 이미지 파일 경로, 클래스 레이블, 데이터셋 크기 등이 주어집니다.&#x20;
* 이 함수는 데이터셋을 반환합니다.



## load\_data\_tag

```c
data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.w = size;
    d.h = size;
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_tags_paths(paths, n, k);
    if(m) free(paths);
    return d;
}
```

함수 이름: load\_data\_tag&#x20;

입력:

* char \*\*paths: 이미지 파일 경로를 담은 배열
* int n: 배열 paths의 길이
* int m: 랜덤으로 이미지를 선택하여 로드할 개수, 0이면 모든 이미지를 로드
* int k: 클래스 개수
* int min: 이미지 크기를 조정할 때 사용할 최소 비율
* int max: 이미지 크기를 조정할 때 사용할 최대 비율
* int size: 로드할 이미지의 크기
* float angle: 이미지 회전 각도 범위
* float aspect: 이미지 가로/세로 비율 범위
* float hue: 이미지 hue 채도 범위
* float saturation: 이미지 채도 범위
* float exposure: 이미지 노출 범위

동작:

* paths 배열에서 이미지를 로드하고 augment를 적용하여 data 구조체를 반환한다.
* m이 0이 아니면, paths 배열에서 m개의 이미지를 랜덤으로 선택하여 로드한다.
* 이미지를 로드할 때, 최소 비율과 최대 비율을 사용하여 이미지 크기를 조정한다.
* 이미지를 로드한 후, augment를 적용하여 이미지를 변환한다.
* 이미지의 태그를 로드하고 one-hot encoding하여 y 행렬에 저장한다.

설명:

* 입력으로 받은 이미지 파일 경로를 이용하여 이미지 데이터를 로드하고, augment를 적용하여 학습 데이터를 만드는 함수이다.
* 데이터가 많아서 모든 이미지를 한번에 로드하기에는 메모리 용량이 부족할 때, 일부 이미지만 로드하고 싶을 때 사용할 수 있는 기능도 제공한다.



## concat\_matrix

```c
matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}
```

함수 이름: concat\_matrix

입력:&#x20;

* 두 개의 matrix(m1, m2)

동작:&#x20;

* 두 개의 matrix를 행 방향으로 이어 붙인 새로운 matrix를 반환한다.

설명:

* m1의 행과 m2의 행을 합한 새로운 matrix m을 만든다.
* m의 열은 m1의 열과 같다.
* m1의 각 행을 m의 행에 복사하고, 그 다음 m2의 각 행을 m의 행에 복사한다.
* m을 반환한다.



## concat\_data

```c
data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    d.w = d1.w;
    d.h = d1.h;
    return d;
}
```

함수 이름: concat\_data

입력:&#x20;

* data 타입의 두 변수 d1과 d2

동작:

* 두 개의 data 변수 d1과 d2를 입력으로 받는다.
* 새로운 data 변수 d를 초기화한다.
* d1과 d2의 X와 y matrix를 각각 concat\_matrix 함수를 이용하여 합쳐서 d의 X와 y matrix에 저장한다.
* d1의 w와 h 값을 d의 w와 h 값으로 대입한다.
* d의 shallow 값에 1을 대입한다.

설명:

* concat\_data 함수는 두 개의 data 변수를 입력으로 받아서 이 두 변수의 X와 y matrix를 합쳐서 새로운 data 변수를 생성하는 함수이다.
* 합쳐진 X와 y matrix는 각각 concat\_matrix 함수를 이용하여 새로운 matrix로 만들어진다.
* d1의 w와 h 값을 d의 w와 h 값으로 대입하는 것은 두 matrix가 합쳐질 때 w와 h값이 동일하기 때문이다.
* d의 shallow 값이 1인 이유는 d 변수에서 X와 y matrix를 복사하지 않고, 이전의 d1과 d2 변수에서 사용되던 matrix를 참조하기 때문이다. 따라서 shallow copy가 일어나는 것이다.



## concat\_datas

```c
data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        free_data(out);
        out = new;
    }
    return out;
}
```

함수 이름: concat\_datas

입력:&#x20;

* d: data 구조체 배열
* n: 배열의 크기

동작:

* 새로운 빈 data 구조체 out을 생성
* d 배열의 각 원소에 대해 concat\_data 함수를 호출하여 out과 합침
* 합쳐진 결과를 out에 저장하고 이전에 생성된 데이터를 해제
* n번 반복 후, 합쳐진 데이터 out을 반환

설명:

* 여러 개의 데이터셋을 하나로 합치는 함수
* 입력으로 받은 data 구조체 배열 d를 하나씩 concat\_data 함수를 호출하여 하나의 data 구조체 out으로 합침
* d 배열의 첫 번째 원소는 out과 합쳐지며, 이후 배열의 각 원소는 out에 이전 데이터가 합쳐진 상태에서 추가로 합쳐짐
* 합쳐진 결과는 새로운 data 구조체로 저장되며, 이전에 생성된 데이터는 메모리 해제됨



## load\_categorical\_data\_csv

```c
data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}
```

함수 이름: load\_categorical\_data\_csv

입력:

* filename: CSV 파일 이름
* target: 타겟 변수 열의 인덱스
* k: 클래스 수

동작:

* CSV 파일을 읽어와서 2차원 행렬 X를 만듦
* target 인덱스에 해당하는 열을 제거하고 그 값을 1차원 배열 truth\_1d에 저장
* truth\_1d를 one-hot 인코딩하여 k개의 열을 가진 2차원 행렬 truth를 만듦
* y 행렬을 생성하고, rows는 X.rows와 같고 cols는 k로 설정하고, truth를 값으로 가짐
* d 구조체에 X와 y를 할당하고 truth\_1d 메모리를 해제하고, d를 반환함

설명:

* load\_categorical\_data\_csv 함수는 CSV 파일에서 데이터를 읽어와서 카테고리컬 변수를 one-hot 인코딩한 결과를 반환하는 함수입니다.&#x20;
* 이 함수는 입력으로 CSV 파일의 이름, 타겟 변수 열의 인덱스, 그리고 클래스 수를 받습니다.&#x20;
* 함수는 먼저 csv\_to\_matrix 함수를 사용하여 CSV 파일을 읽어와서 2차원 행렬 X를 만듭니다.&#x20;
* 그리고 pop\_column 함수를 사용하여 target 인덱스에 해당하는 열을 1차원 배열 truth\_1d로 제거하고 그 값을 저장합니다.&#x20;
* 그 다음 one\_hot\_encode 함수를 사용하여 truth\_1d를 one-hot 인코딩하여 k개의 열을 가진 2차원 행렬 truth를 만듭니다.&#x20;
* y 행렬을 생성하고, rows는 X.rows와 같고 cols는 k로 설정하고, truth를 값으로 가집니다.&#x20;
* 마지막으로, d 구조체에 X와 y를 할당하고 truth\_1d 메모리를 해제하고, d를 반환합니다.



## load\_cifar10\_data

```c
data load_cifar10_data(char *filename)
{
    data d = {0};
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}
```

함수 이름: load\_cifar10\_data

입력:

* char \*filename: CIFAR-10 데이터 파일 경로

동작:

* CIFAR-10 데이터 파일을 로드하여 데이터셋을 생성한다.
* 데이터셋은 10000개의 이미지로 구성되며, 각 이미지는 32x32 크기의 RGB 이미지이다.
* 입력 이미지 데이터는 3072차원의 벡터로 변환되어 X 행렬에 저장된다.
* 출력 레이블은 10차원의 원-핫 인코딩 벡터로 변환되어 y 행렬에 저장된다.
* X, y 행렬을 담은 데이터 구조체를 반환한다.

설명:

* 함수는 CIFAR-10 데이터 파일을 읽어서 데이터셋을 생성한다.
* 데이터셋은 X, y 두 개의 행렬로 이루어진다. X는 입력 이미지 데이터를 저장하는 행렬이고, y는 출력 레이블을 저장하는 행렬이다.
* CIFAR-10 데이터 파일에서는 각 이미지마다 먼저 출력 레이블(클래스) 정보가 주어지고, 이후에는 3072개의 픽셀 정보가 주어진다.
* 파일에서 한 번에 3073바이트씩 읽어서, 첫 번째 바이트에서 출력 레이블 정보를 가져와 y 행렬에 저장한다.
* 나머지 3072바이트는 입력 이미지 데이터를 구성하는 픽셀 정보이므로, X 행렬에 저장한다.
* 마지막으로, 입력 이미지 데이터를 0과 1 사이로 스케일링하고, X 행렬을 정규화한다.
* 스케일링과 정규화는 입력 이미지 데이터를 모델이 더 잘 학습할 수 있도록 전처리하는 작업이다.



## get\_random\_batch

```c
void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}
```

함수 이름: get\_random\_batch

입력:

* data d: 입력 데이터셋
* int n: 가져올 배치 크기
* float \*X: 배치 이미지 데이터를 저장할 포인터
* float \*y: 배치 레이블 데이터를 저장할 포인터

동작:&#x20;

* 주어진 데이터셋 d에서 랜덤하게 배치를 추출하여 X와 y에 저장합니다.

설명:

* for 루프를 이용하여 n개의 랜덤한 인덱스를 추출합니다.
* 추출된 인덱스를 이용하여 d.X와 d.y에서 해당하는 데이터를 복사하여 X와 y에 저장합니다.



## get\_next\_batch

```c
void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}
```

함수 이름: get\_next\_batch

입력:

* data d: 학습 데이터와 레이블을 저장하는 구조체
* int n: 배치 크기(batch size)
* int offset: 현재 배치의 시작 인덱스(offset index)
* float \*X: 입력 데이터를 저장할 float형 포인터 변수
* float \*y: 출력 레이블을 저장할 float형 포인터 변수

동작:

* 현재 배치의 시작 인덱스(offset index)부터 배치 크기(batch size)만큼의 데이터와 레이블을 구조체에서 가져와서
* 입력 데이터 포인터 변수 X와 출력 레이블 포인터 변수 y에 복사한다.

설명:

* 학습할 때 데이터를 일정한 배치 크기만큼 나누어서 처리하는 미니배치(mini-batch) 학습 방식에서 사용되는 함수이다.
* 입력 데이터와 출력 레이블을 일정한 크기로 잘라서 배치(batch) 단위로 가져오는 역할을 한다.
* 구조체 data에는 입력 데이터와 출력 레이블이 각각 2차원 float 배열 형태로 저장되어 있다.
* 현재 배치의 시작 인덱스(offset index)와 배치 크기(batch size)를 이용하여 필요한 데이터와 레이블을 복사한다.
* y 포인터 변수는 NULL일 수 있으므로, y가 NULL이 아닐 경우에만 출력 레이블을 가져온다.



## smooth\_data

```c
void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}
```

함수 이름: smooth\_data&#x20;

입력:&#x20;

* d: data 구조체 변수

동작:&#x20;

* 입력으로 들어온 데이터의 y 값들을 부드럽게 만드는(smooth) 함수입니다.&#x20;
* 이를 위해 각 열(column)의 합이 1이 되도록 하는 L1 정규화(L1 normalization)를 수행합니다.&#x20;
* 이 과정에서 각 y 값에 작은 상수 epsilon을 더하고(값의 평균으로부터 일정한 거리를 두기 위해), 이를 전체의 (1 - epsilon) 만큼 고유값으로 보정해줍니다.&#x20;

설명:&#x20;

* 입력 데이터의 y 값들은 확률 분포(probability distribution)로 사용될 수 있도록 부드럽게 만들어야 할 때가 있습니다. 이 함수는 그러한 목적으로 사용될 수 있습니다.



## load\_all\_cifar10

```c
data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class = bytes[0];
            y.vals[i+b*10000][class] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}
```

함수 이름: load\_all\_cifar10

입력:&#x20;

* 없음

동작:

* CIFAR-10 데이터 세트를 읽어들여서 데이터 행렬(X)과 레이블 행렬(y)을 생성한다.
* 5개의 데이터 파일(data\_batch\_1.bin \~ data\_batch\_5.bin)에서 이미지와 레이블 데이터를 읽어들여서 X와 y에 저장한다.
* 이미지 데이터는 0~~255 범위의 값으로 저장되어 있으며, 이를 0~~1 범위의 값으로 스케일링한다.
* 데이터를 랜덤하게 섞는(smooth\_data) 전처리를 수행한다.
* 생성된 데이터를 data 구조체에 담아서 반환한다.

설명:

* CIFAR-10 데이터 세트는 10개의 클래스(각각 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭)로 이루어진 32x32 크기의 컬러 이미지 데이터 세트이다.
* 이 함수는 CIFAR-10 데이터 세트를 읽어들여서 해당 데이터를 처리하는 데 필요한 작업을 수행하고, 처리된 데이터를 반환한다.
* 데이터 파일은 data/cifar/cifar-10-batches-bin 디렉토리에 위치하며, 각 데이터 파일은 10000개의 이미지와 레이블 데이터를 가지고 있다.
* 이미지 데이터는 각 픽셀(R, G, B)별로 0\~255 사이의 값을 가진다.
* scale\_data\_rows 함수는 데이터 행렬의 각 행을 0\~1 범위로 스케일링하는 함수이다.
* smooth\_data 함수는 데이터를 랜덤하게 섞는 전처리를 수행하는 함수이다.
* 반환되는 data 구조체는 데이터 행렬(X)과 레이블 행렬(y)을 멤버로 가지며, shallow 변수는 0으로 초기화된다.



## load\_go

```c
data load_go(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp))){
        int i;
        if(count == X.rows){
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, "%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i){
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}
```

함수 이름: load\_go

입력:

* filename: 읽어들일 파일 이름을 나타내는 문자열 포인터

동작:

* filename으로 지정된 파일을 읽어들여서 데이터를 처리하고, 처리된 데이터를 반환함.
* 파일에서 한 줄씩 읽어들임.
* 읽어들인 줄의 첫 번째와 두 번째 문자열을 정수로 변환해서, 해당 좌표에 해당하는 y 행렬의 값을 1로 설정함.
* 읽어들인 줄의 세 번째 문자열부터 361개의 문자를 읽어들여서, 해당하는 X 행렬의 값을 설정함.
* 모든 줄을 읽어들인 후에, X와 y 행렬을 resize함.
* 처리된 데이터를 저장하고 있는 data 구조체를 초기화해서, 처리된 데이터를 저장하고 있는 행렬들을 포함시킴.
* 처리된 데이터가 저장된 data 구조체를 반환함.

설명:

* load\_go 함수는 지정된 파일에서 데이터를 읽어들여서 처리하는 함수입니다.
* 함수는 filename으로 지정된 파일을 "rb" 모드로 열고, 열기에 실패하면 file\_error 함수를 호출합니다.
* 함수는 while 루프를 돌면서 파일에서 한 줄씩 읽어들입니다.
* label 포인터 변수에 fgetl 함수를 사용해서 파일에서 한 줄씩 읽어들입니다.
* 만약, X의 행 개수와 count가 같아지면, X와 y 행렬을 resize해서 크기를 2배로 늘립니다.
* label 포인터 변수에서 읽어들인 문자열을 sscanf 함수를 사용해서 row와 col 변수로 분리해냅니다.
* fgetl 함수를 사용해서 다음 줄에서 보드의 상태를 나타내는 문자열을 읽어들입니다.
* index 변수에 row와 col을 이용해서 y 행렬에서 해당하는 인덱스를 계산해서, 해당하는 위치의 값을 1로 설정합니다.
* for 루프를 돌면서 보드의 상태를 나타내는 문자열에서 읽어들인 값을 float 형태로 변환해서, X 행렬에 저장합니다.
* count 값을 증가시키고, label과 board 포인터 변수를 free 함수를 사용해서 메모리를 해제합니다.
* 모든 줄을 읽어들인 후에, X와 y 행렬을 resize해서, 처리된 데이터를 저장하고 있는 행렬들의 크기를 줄입니다.
* data 구조체를 초기화해서, 처리된 데이터를 저장하고 있는 행렬들을 포함시키고, 해당하는 data 구조체를 반환합니다.
* 함수 실행이 끝나면, 파일을 닫습니다.



## randomize\_data

```c
void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = rand()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}
```

함수 이름: randomize\_data

입력:&#x20;

* data d: 데이터

동작:&#x20;

* 입력받은 데이터의 X와 y 값을 무작위로 섞음

설명:&#x20;

* 입력으로 받은 데이터의 X와 y 값을 무작위로 섞는 함수입니다.&#x20;
* 이를 위해 먼저 입력된 데이터의 X와 y 배열을 역순으로 순회하면서 현재 인덱스와 랜덤으로 선택된 인덱스의 값을 서로 바꿔줍니다.&#x20;
* 이를 모든 인덱스에 대해 반복하면 X와 y 값이 무작위로 섞인 데이터가 만들어집니다.



## scale\_data\_rows

```c
void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}
```

함수 이름: scale\_data\_rows

입력:

* data d: 변환할 학습 데이터를 담고 있는 data 구조체
* float s: 스케일링(scale)할 비율

동작:&#x20;

* 이 함수는 입력으로 받은 data 구조체(d)의 X 필드(입력 데이터)에 대해, 각 행(row)의 값을 s만큼 스케일링합니다.

설명:

* 함수 내부에서는 입력으로 받은 data 구조체(d)의 X 필드에 대해 각 행(row)의 값을 스케일링합니다.
* 스케일링(scale)할 비율(s)이 1보다 작을 경우, 각 원소(element)의 값을 s만큼 감소시킵니다.
* 스케일링(scale)할 비율(s)이 1보다 클 경우, 각 원소(element)의 값을 s만큼 증가시킵니다.
* 입력으로 받은 data 구조체(d)는 변경되며, 반환값은 없습니다.



## translate\_data\_rows

```c
void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}
```

함수 이름: translate\_data\_rows

입력:

* data d: 변환할 학습 데이터를 담고 있는 data 구조체
* float s: 이동 거리

동작:&#x20;

* 이 함수는 입력으로 받은 data 구조체(d)의 X 필드(입력 데이터)에 대해, 각 행(row)의 값을 s만큼 이동시킵니다.

설명:

* 함수 내부에서는 입력으로 받은 data 구조체(d)의 X 필드에 대해 각 행(row)의 값을 이동시킵니다.
* 이동 거리(s)가 양수일 경우, 각 원소(element)의 값을 s만큼 증가시킵니다.
* 이동 거리(s)가 음수일 경우, 각 원소(element)의 값을 s만큼 감소시킵니다.
* 입력으로 받은 data 구조체(d)는 변경되며, 반환값은 없습니다.



## copy\_data

```c
data copy_data(data d)
{
    data c = {0};
    c.w = d.w;
    c.h = d.h;
    c.shallow = 0;
    c.num_boxes = d.num_boxes;
    c.boxes = d.boxes;
    c.X = copy_matrix(d.X);
    c.y = copy_matrix(d.y);
    return c;
}
```

함수 이름: copy\_data&#x20;

입력:&#x20;

* data d: 복사할 학습 데이터를 담고 있는 data 구조체

동작:&#x20;

* 이 함수는 입력으로 받은 data 구조체(d)를 복사한 새로운 data 구조체를 생성하여 반환합니다. 이때, 입력으로 받은 data 구조체(d)와 반환할 data 구조체는 서로 다른 메모리 공간을 참조하게 됩니다.

설명:

* 함수 내부에서는 입력으로 받은 data 구조체(d)의 필드값들을 새로운 data 구조체(c)에 복사합니다.
* 이때, data 구조체의 X와 y 필드는 copy\_matrix 함수를 사용하여 복사합니다. copy\_matrix 함수는 입력으로 받은 행렬(matrix)을 새로운 메모리 공간에 복사한 후, 복사된 행렬을 가리키는 새로운 메모리 주소를 반환합니다.
* 최종적으로, 복사된 data 구조체(c)가 반환됩니다.



## normalize\_data\_rows

```c
void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}
```

함수 이름: normalize\_data\_rows&#x20;

입력:&#x20;

* data d: 학습 데이터를 담고 있는 data 구조체

동작:&#x20;

* 이 함수는 입력으로 받은 data 구조체(d)의 특징값(feature)을 정규화(normalization)합니다.&#x20;
* 정규화는 각 특징값이 0과 1사이의 범위에 있도록 값을 스케일링(scale)하는 것을 말합니다.&#x20;
* 여기서는 각각의 특징값을 해당 특징에서 최대값으로 나누어서 스케일링합니다.

설명:

* 함수 내부에서는 for 루프를 이용해 각각의 특징값(feature)을 스케일링합니다.
* 루프에서는 normalize\_array 함수를 호출하여, 한 개의 특징값(feature) 배열을 정규화합니다. 이때, normalize\_array 함수는 입력으로 받은 배열에서 최대값을 찾아서, 해당 배열의 모든 원소를 최대값으로 나누어주는 작업을 수행합니다.
* 최종적으로, 입력으로 받은 data 구조체(d)는 내부의 특징값(feature)이 스케일링된 상태로 변경됩니다.



## get\_data\_part

```c
data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}
```

함수 이름: get\_data\_part&#x20;

입력:

* data d: 전체 학습 데이터를 담고 있는 data 구조체
* int part: 추출할 부분 데이터의 번호 (0부터 시작)
* int total: 전체 추출할 부분 데이터 개수

동작:&#x20;

* 이 함수는 입력으로 받은 전체 학습 데이터(data d)를 total 개수로 분할한 후, part 번째 분할된 부분 데이터를 추출하여 새로운 data 구조체(p)에 담아 반환합니다.

설명:

* 함수 내부에서 새로 생성한 data 구조체(p)는 shallow 멤버 변수가 1로 설정되어 있습니다. 이것은 p가 가리키는 메모리가 새로운 메모리가 아니라 d의 일부를 공유한다는 것을 의미합니다.
* p.X, p.y는 각각 추출된 부분 데이터의 특징값(feature)과 라벨값(label)을 저장할 메모리 공간입니다. p.X.rows, p.X.cols는 특징값 행렬의 크기를 나타냅니다. 마찬가지로, p.y.rows, p.y.cols는 라벨값 행렬의 크기를 나타냅니다.
* p.X.vals, p.y.vals는 특징값과 라벨값을 저장할 포인터 배열입니다. 이전 분할된 부분 데이터까지의 특징값과 라벨값의 개수를 d.X.rows, d.y.rows로 나누어서 분할된 부분 데이터의 시작 포인터를 계산합니다. 이렇게 계산된 포인터를 각각 p.X.vals, p.y.vals에 저장합니다.
* 최종적으로, 새로 생성된 data 구조체(p)를 반환합니다.



## get\_random\_data

```c
data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = calloc(num, sizeof(float *));
    r.y.vals = calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i){
        int index = rand()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}
```

함수 이름: get\_random\_data&#x20;

입력:

* data d: 학습 데이터를 담고 있는 data 구조체
* int num: 무작위로 추출할 데이터의 개수

동작:&#x20;

* 이 함수는 입력으로 받은 학습 데이터(data d)에서 무작위로(num 개수만큼) 데이터를 추출하여 새로운 data 구조체(r)에 담아 반환합니다.

설명:

* 함수 내부에서 새로 생성한 data 구조체(r)는 shallow 멤버 변수가 1로 설정되어 있습니다. 이것은 r이 가리키는 메모리가 새로운 메모리가 아니라 d의 일부를 공유한다는 것을 의미합니다.
* r.X, r.y는 각각 추출된 데이터의 특징값(feature)과 라벨값(label)을 저장할 메모리 공간입니다. r.X.rows, r.X.cols는 특징값 행렬의 크기를 나타냅니다. 마찬가지로, r.y.rows, r.y.cols는 라벨값 행렬의 크기를 나타냅니다.
* r.X.vals, r.y.vals는 특징값과 라벨값을 저장할 포인터 배열입니다. num 개수만큼 동적으로 할당되어 각 포인터는 각각 무작위로 추출된 데이터의 특징값과 라벨값을 가리키게 됩니다.
* 무작위 데이터 추출을 위해서 rand() 함수를 사용하며, d.X.rows(전체 데이터 개수)를 범위로 하는 난수를 생성합니다. 이렇게 생성된 난수로부터 추출할 데이터의 인덱스를 계산하여, 해당 데이터의 특징값과 라벨값을 r.X.vals, r.y.vals에 저장합니다.
* 최종적으로, 새로 생성된 data 구조체(r)를 반환합니다.



## split\_data

```c
data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}
```

함수 이름: split\_data&#x20;

입력:

* data d: 학습 데이터셋과 검증 데이터셋으로 분할할 전체 데이터셋
* int part: 현재 분할하려는 데이터셋의 인덱스
* int total: 전체 데이터셋을 분할한 데이터셋의 개수

동작:&#x20;

* 전체 데이터셋을 part와 total의 값에 따라 학습 데이터셋과 검증 데이터셋으로 분할하여 반환함

설명:

* data 구조체는 입력 데이터와 레이블 데이터를 저장하는 두 개의 행렬(X와 y)로 구성됨
* split\_data 함수는 전체 데이터셋 d를 part와 total의 값에 따라 train과 test 데이터셋으로 나누어 반환함
* train과 test 데이터셋은 data 구조체의 포인터이며, split 배열에 저장되어 반환됨
* start와 end 변수는 현재 분할하려는 데이터셋의 시작과 끝 인덱스를 계산함
* train 데이터셋은 start 이전의 데이터와 end 이후의 데이터를 모두 포함함
* test 데이터셋은 start부터 end 이전의 데이터를 포함함
* train과 test 데이터셋은 행렬의 크기와 값들을 복사하여 생성함
* 반환되는 split\[0]은 train 데이터셋을, split\[1]은 test 데이터셋을 가리키는 포인터이며, split 배열은 calloc 함수를 사용하여 동적으로 할당됨

