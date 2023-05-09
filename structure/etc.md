---
description: 구조체
---

# etc

## 구조체

* 사용되는 구조체를 모았습니다.

***

## src/include/darknet.h

### node

```c
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;
```

### list

```c
typedef struct list{
    int size;
    node *front;
    node *back;
} list;
```

### matrix

```c
typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;
```

### box

```c
typedef struct{
    float x, y, w, h;
} box;
```

### box\_label

```c
typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;
```

### data

```c
typedef struct{
    int w, h;
    matrix X; // X.cols = total pixel , rows = batch(64) , vals = each pixel's value (we can find information in data.c's load_data_detection)
    matrix y; // y.cols = 5*boxes( boxes default 90 ) , y.rows = batch(64) ,
    int shallow;
    int *num_boxes;
    box **boxes;
} data;
```

X : cols, rows

* cols : 이미지 한장의 픽셀
* rows : 이미지의 개수 batch size
* vals : 각 픽셀의 값

Y : cols, rows

* cols : 이미지 한장에서의 box의 개수
* rows : 이미지의 개수 batch size

### detection

```c
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;
```

검출한 데이터에 대한 정보를 담고 있습니다.

***

## /src/option\_list.h

### kvp 구조체

```c
typedef struct{
    char *key;
    char *val;
    int used;
} kvp;
```

***

## /src/parser.c

### section

```c
typedef struct{
    char *type;
    list *options;
}section;
```

### size\_params

```c
typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;
```

## /src/box.h

### dbox

```c
typedef struct{
    float dx, dy, dw, dh;
} dbox;
```

* box의 미분 값을 저장합니다.

## 내장/time.h

### timeval

```c
struct timeval {
  long tv_sec;       // 1초
  long tv_usec;      // 1/1000000초
};
```
