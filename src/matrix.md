# matrix

## free\_matrix

```c
void free_matrix(matrix m)
{
    int i;
    for(i = 0; i < m.rows; ++i) free(m.vals[i]);
    free(m.vals);
}
```

함수 이름: free\_matrix

입력:&#x20;

* matrix m (2차원 배열)

동작:&#x20;

* 2차원 배열 m의 할당된 메모리를 해제하는 함수입니다.&#x20;
* 행마다 할당된 메모리를 우선 해제한 뒤, 마지막으로 2차원 배열 자체의 메모리를 해제합니다.

설명:&#x20;

* 이 함수는 Darknet 라이브러리에서 사용되는 함수로, 2차원 배열로 이루어진 행렬(matrix)의 메모리를 해제합니다.&#x20;
* 이 함수는 C언어에서 동적으로 할당한 메모리를 해제하는 함수 중 하나인 free() 함수를 사용합니다.&#x20;
* Darknet 라이브러리에서는 행렬(matrix)을 사용하여 다양한 계산을 수행하므로, 행렬 계산을 마치고 나서는 메모리를 해제해주어야 합니다.



## matrix\_topk\_accuracy

```c
float matrix_topk_accuracy(matrix truth, matrix guess, int k)
{
    int *indexes = calloc(k, sizeof(int));
    int n = truth.cols;
    int i,j;
    int correct = 0;
    for(i = 0; i < truth.rows; ++i){
        top_k(guess.vals[i], n, k, indexes);
        for(j = 0; j < k; ++j){
            int class = indexes[j];
            if(truth.vals[i][class]){
                ++correct;
                break;
            }
        }
    }
    free(indexes);
    return (float)correct/truth.rows;
}
```

함수 이름: matrix\_topk\_accuracy

입력:

* truth: 참 값 행렬(matrix) (float 타입)
* guess: 예측 값 행렬(matrix) (float 타입)
* k: 상위 k개의 클래스를 가져오기 위한 값 (int 타입)

동작:

* 예측 값 행렬에서 각 샘플마다 가장 높은 k개의 값이 들어있는 인덱스를 가져온다.
* 참 값 행렬에서 해당 샘플의 클래스가 k개 중 하나인 경우 정확한 예측으로 간주하고 정확한 예측 수를 계산한다.
* 모든 샘플에 대한 정확도를 계산하여 반환한다.

설명:

* 이 함수는 상위 k개의 클래스에 대해 정확도를 계산하는 데 사용된다.
* 입력된 truth와 guess 행렬은 예측 모델의 출력 값과 실제 참 값을 나타낸다.
* k는 가져올 상위 클래스의 수를 정의한다. 예를 들어, k=1인 경우, 가장 높은 값이 들어있는 인덱스를 가져와서 하나의 클래스로 예측을 수행한다.
* 이 함수는 모든 샘플에 대해 예측과 참 값이 얼마나 일치하는지를 계산하여 정확도를 반환한다.



## scale\_matrix

```c
void scale_matrix(matrix m, float scale)
{
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.vals[i][j] *= scale;
        }
    }
}
```

함수 이름: scale\_matrix

입력:&#x20;

* matrix m (스케일링을 적용할 행렬)
* float scale (적용할 스케일 값)

동작:&#x20;

* 주어진 행렬의 모든 원소에 주어진 스케일 값을 곱해 스케일링을 적용함

설명:&#x20;

* 입력으로 주어진 행렬 m의 모든 원소에 스케일 값을 곱해 행렬을 스케일링하는 함수입니다.&#x20;
* 스케일링이란, 행렬의 모든 원소에 일정한 값을 곱하는 연산으로, 행렬을 확대 또는 축소시키는 효과를 줄 수 있습니다.&#x20;
* 이 함수에서는 주어진 스케일 값만큼 모든 원소를 곱하여 스케일링을 적용합니다.



## resize\_matrix

```c
matrix resize_matrix(matrix m, int size)
{
    int i;
    if (m.rows == size) return m;
    if (m.rows < size) {
        m.vals = realloc(m.vals, size*sizeof(float*));
        for (i = m.rows; i < size; ++i) {
            m.vals[i] = calloc(m.cols, sizeof(float));
        }
    } else if (m.rows > size) {
        for (i = size; i < m.rows; ++i) {
            free(m.vals[i]);
        }
        m.vals = realloc(m.vals, size*sizeof(float*));
    }
    m.rows = size;
    return m;
}
```

함수 이름: resize\_matrix

입력:&#x20;

* matrix m (크기를 조정할 행렬)
* int size (조정된 행렬의 행 개수)

동작:&#x20;

* 입력으로 주어진 행렬 m의 행 개수를 size로 조정하고, 그 결과를 반환한다.&#x20;
* size가 m.rows보다 작으면, m의 마지막 size \~ m.rows-1 행을 제거한다.&#x20;
* size가 m.rows보다 크면, m의 행 개수를 size로 늘리고, 새로 추가된 행은 0으로 초기화한다.

설명:&#x20;

* 입력으로 주어진 행렬 m의 행 개수를 조정하는 함수이다.&#x20;
* 행렬의 크기를 조정할 때, realloc 함수를 사용하여 메모리를 할당하거나 해제한다.&#x20;
* 새로운 행은 0으로 초기화하기 위해 calloc 함수를 사용한다.



## matrix\_add\_matrix

```c
void matrix_add_matrix(matrix from, matrix to)
{
    assert(from.rows == to.rows && from.cols == to.cols);
    int i,j;
    for(i = 0; i < from.rows; ++i){
        for(j = 0; j < from.cols; ++j){
            to.vals[i][j] += from.vals[i][j];
        }
    }
}
```

함수 이름: matrix\_add\_matrix

입력:

* matrix from: 더해지는 행렬
* matrix to: 더해지는 대상 행렬

동작:

* from 행렬의 각 요소들을 to 행렬의 해당 요소들과 더한 후, 그 결과를 to 행렬의 해당 요소에 다시 저장한다.

설명:

* from과 to 행렬의 크기가 같아야 한다.
* from과 to 행렬은 함수 내에서 변경되므로, 원본 행렬을 보존해야 하는 경우 복사본을 만들어서 사용해야 한다.



## copy\_matrix

```c
matrix copy_matrix(matrix m)
{
    matrix c = {0};
    c.rows = m.rows;
    c.cols = m.cols;
    c.vals = calloc(c.rows, sizeof(float *));
    int i;
    for(i = 0; i < c.rows; ++i){
        c.vals[i] = calloc(c.cols, sizeof(float));
        copy_cpu(c.cols, m.vals[i], 1, c.vals[i], 1);
    }
    return c;
}
```

함수 이름: copy\_matrix&#x20;

입력:&#x20;

* matrix m (복사할 행렬)&#x20;

동작:&#x20;

* 입력된 행렬 m을 복사하여 새로운 행렬 c를 생성하고 반환함. 새로운 행렬 c는 입력된 행렬 m과 같은 크기를 가지며, 동일한 값을 가지도록 함.&#x20;

설명:

* 함수는 입력된 행렬 m을 복사하여 새로운 행렬 c를 생성하고 반환함.
* 새로운 행렬 c는 입력된 행렬 m과 같은 크기를 가지며, 동일한 값을 가지도록 함.
* 입력된 행렬 m과 새로운 행렬 c는 다른 메모리 공간에 저장됨.
* 함수 내부에서는 메모리 할당을 위해 calloc 함수를 사용함.



## make\_matrix

```c
matrix make_matrix(int rows, int cols)
{
    int i;
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.vals = calloc(m.rows, sizeof(float *));
    for(i = 0; i < m.rows; ++i){
        m.vals[i] = calloc(m.cols, sizeof(float));
    }
    return m;
}
```

함수 이름: make\_matrix

입력:&#x20;

* (int) rows: 생성할 행의 수
* (int) cols - 생성할 열의 수

동작:&#x20;

* rows와 cols 크기의 matrix를 생성하고 0으로 초기화

설명:&#x20;

* 입력으로 주어진 크기(rows \* cols)로 matrix를 생성하고, 행렬의 값을 0으로 초기화한 후 생성된 matrix를 반환하는 함수입니다.



## hold\_out\_matrix

```c
matrix hold_out_matrix(matrix *m, int n)
{
    int i;
    matrix h;
    h.rows = n;
    h.cols = m->cols;
    h.vals = calloc(h.rows, sizeof(float *));
    for(i = 0; i < n; ++i){
        int index = rand()%m->rows;
        h.vals[i] = m->vals[index];
        m->vals[index] = m->vals[--(m->rows)];
    }
    return h;
}
```

함수 이름: hold\_out\_matrix

입력:&#x20;

* matrix \*m (포인터)
* int n

동작:&#x20;

* 입력으로 받은 matrix 포인터 m에서 무작위로 n개의 샘플을 추출하여 그 샘플들로 이루어진 새로운 matrix h를 생성하고 반환한다. 이때, m에서 추출된 샘플들은 m에서 제거된다.

설명:&#x20;

* hold-out 기법은 머신러닝 모델의 성능을 평가하기 위해 데이터셋을 학습 데이터와 테스트 데이터로 나누는 방법 중 하나이다.&#x20;
* 이 함수는 입력으로 받은 matrix 포인터 m에서 무작위로 n개의 샘플을 추출하여 테스트 데이터셋으로 사용하기 위한 matrix h를 생성하고, 이러한 샘플들을 m에서 제거함으로써 학습 데이터셋을 구성하는 데 사용한다.&#x20;
* 반환되는 matrix h는 테스트 데이터셋으로 사용되며, 학습 데이터셋은 입력으로 받은 matrix 포인터 m에서 추출된 샘플을 제외한 나머지 샘플들로 구성된다.



## pop\_column

```c
float *pop_column(matrix *m, int c)
{
    float *col = calloc(m->rows, sizeof(float));
    int i, j;
    for(i = 0; i < m->rows; ++i){
        col[i] = m->vals[i][c];
        for(j = c; j < m->cols-1; ++j){
            m->vals[i][j] = m->vals[i][j+1];
        }
    }
    --m->cols;
    return col;
}
```

함수 이름: pop\_column

입력:&#x20;

* matrix \*m (포인터 변수, 삭제될 열을 포함하는 행렬)
* int c (정수, 삭제할 열의 인덱스)

동작:&#x20;

* 입력된 행렬에서 해당 열의 데이터를 꺼내어 배열 형태로 리턴하고, 입력된 행렬에서 해당 열의 데이터를 삭제한다.

설명:&#x20;

* 입력된 행렬의 열을 하나 제거하고 그 열의 데이터를 배열 형태로 리턴하는 함수이다.&#x20;
* 입력된 행렬의 c번째 열의 데이터를 col 배열에 저장하고, 해당 열을 제거한 후 열의 개수를 감소시킨다.&#x20;
* 삭제된 열 이후의 열은 모두 왼쪽으로 한 칸씩 이동하여 메모리 상에 유지된다.



## csv\_to\_matrix

```c
matrix csv_to_matrix(char *filename)
{
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);

    matrix m;
    m.cols = -1;

    char *line;

    int n = 0;
    int size = 1024;
    m.vals = calloc(size, sizeof(float*));
    while((line = fgetl(fp))){
        if(m.cols == -1) m.cols = count_fields(line);
        if(n == size){
            size *= 2;
            m.vals = realloc(m.vals, size*sizeof(float*));
        }
        m.vals[n] = parse_fields(line, m.cols);
        free(line);
        ++n;
    }
    m.vals = realloc(m.vals, n*sizeof(float*));
    m.rows = n;
    return m;
}
```

함수 이름: csv\_to\_matrix

입력:&#x20;

* char\* filename: 읽어들일 CSV 파일 이름

동작:&#x20;

* CSV 파일을 읽어들여 matrix 구조체로 변환하는 함수입니다.&#x20;
* 파일을 읽어들일 때, 각 라인의 컬럼 수를 파악하고, 필드 값을 파싱하여 matrix 구조체에 저장합니다.

설명:&#x20;

* 입력받은 파일 이름으로 파일을 열어서 파일이 없으면 에러를 발생시키고, 파일을 성공적으로 열었을 때, matrix 구조체를 초기화합니다.&#x20;
* 그 다음, 파일에서 한 줄씩 읽어들여 각 라인의 컬럼 수를 파악합니다. 라인의 컬럼 수가 처음 읽어들인 경우, matrix 구조체의 컬럼 수로 설정합니다.&#x20;
* 이후, 각 라인의 필드 값을 파싱하여 matrix 구조체에 저장합니다. 만약, 필드 값 파싱 도중 에러가 발생하면 프로그램이 종료됩니다.&#x20;
* 모든 라인을 읽어들인 후, 메모리를 최적화하기 위해 matrix 구조체가 저장된 메모리의 크기를 조정합니다.&#x20;
* 마지막으로, matrix 구조체의 행 수를 저장하고, matrix 구조체를 반환합니다.



## matrix\_to\_csv

```c
void matrix_to_csv(matrix m)
{
    int i, j;

    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            if(j > 0) printf(",");
            printf("%.17g", m.vals[i][j]);
        }
        printf("\n");
    }
}
```

```
0,0,0
0,0,0
0,0,0
```

* 위와 같은 형태로 출력됩니다.

함수 이름: matrix\_to\_csv

입력:&#x20;

* matrix m (CSV 파일로 저장할 행렬)

동작:&#x20;

* 입력으로 주어진 행렬을 CSV 파일 형식으로 출력합니다. 각 행과 열은 쉼표로 구분되며, 각 행의 끝에는 개행 문자가 포함됩니다.

설명:&#x20;

* 함수는 주어진 행렬을 인자로 받아서, 각 원소를 CSV 파일 형식으로 출력합니다. 이 함수는 주로 행렬의 데이터를 저장하거나 출력하는 데 사용됩니다.



## print\_matrix

```c
void print_matrix(matrix m)
{
    int i, j;
    printf("%d X %d Matrix:\n",m.rows, m.cols);
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.vals[i][j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}
```

```
__                                               __
|                                                   |
|        0.0000000       0.0000000       0.0000000  |
|        0.0000000       0.0000000       0.0000000  |
|        0.0000000       0.0000000       0.0000000  |
|__                                               __|
```

* 위와 같은 형태로 출력됩니다.

함수 이름: print\_matrix

입력:&#x20;

* matrix m (출력하고자 하는 행렬)

동작:&#x20;

* 입력으로 받은 행렬을 예쁘게 포맷팅하여 출력한다. 각 요소는 15.7f 형식으로 출력되며, 행과 열의 경계에는 선으로 구분된 테두리가 그려진다.

설명:&#x20;

* 이 함수는 주어진 행렬을 예쁘게 출력하기 위해 만들어졌다.&#x20;
* 입력으로 받은 행렬을 이중 for 루프를 통해 순회하며, 각 요소를 15.7f 형식으로 출력한다.&#x20;
* 이때 각 행과 열의 경계에는 선으로 구분된 테두리가 그려진다.

