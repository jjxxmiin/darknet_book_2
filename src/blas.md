# blas

## blas 란?

벡터의 덧셈, 내적, 선형 조합, 행렬 곱셈과 같은 일반적인 선형 대수 연산을 수행하기 위한 역할을 합니다.

* Basic Linear Algebra Subprograms의 약자입니다.
* 크게 3개의 level(`vector-vector`, `matrix-vector`, `matrix-matrix`)로 구분되어 집니다.

***

### copy\_cpu

```c
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}
```

함수 이름: copy\_cpu

입력:

* N: 복사할 요소 수
* X: 복사할 데이터의 포인터
* INCX: X의 인덱스 간격
* Y: 복사 대상 데이터의 포인터
* INCY: Y의 인덱스 간격

동작:

* X의 요소를 Y에 복사합니다.
* 각각의 요소는 X의 인덱스 간격 INCX와 Y의 인덱스 간격 INCY를 사용하여 복사됩니다.

설명:&#x20;

* 위 코드는 copy\_cpu 함수입니다. 이 함수는 X 포인터가 가리키는 데이터를 Y 포인터가 가리키는 위치에 복사합니다. 이때 INCX와 INCY를 이용하여 X와 Y의 요소 간격을 조절할 수 있습니다.&#x20;
* 이 함수는 주로 딥 러닝에서 사용되는 배열 연산에서 많이 활용됩니다.

### mean\_cpu

```c
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}
```

함수 이름: mean\_cpu

입력:

* x: 입력 데이터의 포인터
* batch: 배치 크기
* filters: 필터 수
* spatial: 공간 크기
* mean: 평균값을 저장할 포인터

동작:

* x의 요소들을 이용하여 각 필터의 평균값을 계산합니다.
* mean 포인터가 가리키는 위치에 각 필터의 평균값을 저장합니다.

설명:&#x20;

* 위 코드는 mean\_cpu 함수입니다. 이 함수는 입력 데이터 x의 요소들을 이용하여 각 필터의 평균값을 계산하고, 그 결과를 mean 포인터가 가리키는 위치에 저장합니다.&#x20;
* 평균값은 배치 크기, 필터 수, 공간 크기를 고려하여 계산되며, 배치 크기와 공간 크기를 곱한 값의 역수를 scale 변수에 저장하여 평균값 계산에 사용합니다.&#x20;
* 이 함수는 주로 딥 러닝에서 사용되는 배열 연산에서 많이 활용됩니다.



### variance\_cpu

```c
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}
```

함수 이름: variance\_cpu

입력:

* x: 입력 데이터
* mean: 입력 데이터의 평균 값
* batch: 배치 크기
* filters: 필터의 개수
* spatial: 공간 차원(너비와 높이)

동작:

* 모든 배치, 필터 및 공간 요소에 대해, 입력 데이터 x의 해당 필터 요소와 평균 값 간의 차이를 계산하고, 그 차이의 제곱을 누적하여 분산 값을 계산합니다.
* 계산된 분산 값을 배치 크기 및 공간 차원에서 감소된 자유도에 따라 스케일링합니다.

설명:&#x20;

* 위 코드는 입력 데이터의 분산 값을 계산하는 함수인 variance\_cpu입니다. 분산은 데이터가 얼마나 분산되어 있는지를 나타내는 지표이며, 입력 데이터의 분산 값은 딥 러닝에서 Batch Normalization 등과 같은 기술에서 자주 사용됩니다.&#x20;
* 이 함수는 입력 데이터 x와 해당 필터의 평균 값을 이용하여 분산 값을 계산하고, 스케일링을 통해 반환합니다.



### normalize\_cpu

```c
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}
```

함수 이름: normalize\_cpu

입력:

* x: 정규화할 데이터의 포인터
* mean: 평균 값의 포인터
* variance: 분산 값의 포인터
* batch: 배치 크기
* filters: 필터 개수
* spatial: 공간 크기

동작:

* 입력으로 받은 데이터 x를 정규화하여 각 요소에 대해 평균 값을 빼고, 표준 편차 값으로 나누어 저장합니다.
* 데이터 x는 입력 인덱스(b, f, i)에 따라서 정규화되어 저장됩니다.

설명:&#x20;

* 위 코드는 딥 러닝에서 많이 사용되는 정규화 함수인 normalize\_cpu입니다. 이 함수는 입력된 데이터의 평균과 분산을 이용하여 데이터를 정규화하는데 사용됩니다.&#x20;
* 이 함수는 각 요소를 해당 필터의 평균 값으로 뺀 후, 해당 필터의 분산 값으로 나누어 정규화합니다.&#x20;
* 이 함수를 사용하면 모델이 더 빠르고 안정적으로 수렴하게 되므로, 딥 러닝 모델에서 많이 활용됩니다.



### axpy\_cpu

```c
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}
```

함수 이름: axpy\_cpu

입력:

* N: X 및 Y에서 복사해야 할 요소 수
* ALPHA: X에서 Y로 복사될 때 곱해지는 스칼라 값
* X: 복사할 데이터가 저장된 원본 배열
* INCX: X의 인덱스 간격
* Y: 데이터가 복사될 대상 배열
* INCY: Y의 인덱스 간격

동작:

* X의 요소를 Y에 복사하고 ALPHA 값으로 곱한 다음 Y에 더합니다.
* 각각의 요소는 X의 인덱스 간격 INCX와 Y의 인덱스 간격 INCY를 사용하여 복사됩니다.

설명:&#x20;

* 위 코드는 axpy\_cpu 함수입니다. 이 함수는 X 포인터가 가리키는 데이터를 ALPHA 값을 곱한 후 Y 포인터가 가리키는 위치에 더합니다. 이때 INCX와 INCY를 이용하여 X와 Y의 요소 간격을 조절할 수 있습니다.&#x20;
* 이 함수는 주로 딥 러닝에서 사용되는 배열 연산에서 많이 활용됩니다.



### scal\_cpu

```c
void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}
```

함수 이름: scal\_cpu

입력:

* N: 스칼라를 적용할 벡터 X의 크기
* ALPHA: 스칼라 값
* X: 스칼라 값을 적용할 벡터의 포인터
* INCX: X의 인덱스 간격

동작:&#x20;

* 벡터 X의 각 요소에 스칼라 값 ALPHA를 곱해 갱신합니다. 이때 INCX를 이용하여 X의 인덱스 간격을 조절할 수 있습니다.

설명:&#x20;

* 위 코드는 scal\_cpu 함수입니다. 이 함수는 주로 딥 러닝에서 사용되는 배열 연산에서 많이 활용됩니다.&#x20;
* 벡터 X의 각 요소에 스칼라 값을 곱해 갱신하는 것은 배열의 크기를 변경하는 것이 아니라, 벡터를 스칼라 값으로 스케일링하는 것을 의미합니다. 따라서 이 함수는 주로 데이터 전처리나 최적화 알고리즘에서 사용됩니다.



### fill\_cpu

```c
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}
```

함수 이름: fill\_cpu

입력:

* N: 채워질 요소 수
* ALPHA: 배열에 할당할 값
* X: 채워질 데이터의 포인터
* INCX: X의 인덱스 간격

동작:

* X 배열에 ALPHA 값을 채웁니다.
* 각 요소는 X의 인덱스 간격 INCX를 사용하여 채워집니다.

설명:

* 위 코드는 fill\_cpu 함수입니다. 이 함수는 X 포인터가 가리키는 배열에 ALPHA 값으로 모든 요소를 채웁니다. 이때 INCX를 이용하여 X 배열의 요소 간격을 조절할 수 있습니다.
* 이 함수는 주로 딥 러닝에서 초기화나 전처리 과정에서 사용됩니다.



### mul\_cpu

```c
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}
```

함수 이름: mul\_cpu

입력:

* N: 요소 수
* X: 곱해질 데이터의 포인터
* INCX: X의 인덱스 간격
* Y: 곱하는 대상 데이터의 포인터
* INCY: Y의 인덱스 간격

동작:

* Y의 각 요소에 X의 해당 요소를 곱합니다.
* 각 요소는 X와 Y의 인덱스 간격 INCX와 INCY를 사용하여 계산됩니다.

설명:&#x20;

* 위 코드는 mul\_cpu 함수입니다. 이 함수는 X 포인터가 가리키는 데이터와 Y 포인터가 가리키는 대상 데이터를 요소별로 곱합니다. 이때 INCX와 INCY를 이용하여 X와 Y의 요소 간격을 조절할 수 있습니다.&#x20;
* 이 함수는 주로 딥 러닝에서 사용되는 배열 연산에서 많이 활용됩니다.

### pow\_cpu

```c
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}
```

함수 이름: pow\_cpu

입력:

* N: 처리할 데이터 개수
* ALPHA: 거듭제곱 계산에 사용할 지수
* X: 입력 데이터 배열 포인터
* INCX: 입력 데이터 배열에서 원소 사이의 간격
* Y: 출력 데이터 배열 포인터
* INCY: 출력 데이터 배열에서 원소 사이의 간격

동작:&#x20;

* 입력 데이터 배열 X에서 원소를 ALPHA 지수만큼 거듭제곱하여 출력 데이터 배열 Y에 저장함

설명:&#x20;

* pow\_cpu 함수는 입력 데이터 배열 X에서 원소를 ALPHA 지수만큼 거듭제곱하여 출력 데이터 배열 Y에 저장하는 함수입니다.&#x20;
* 이 함수는 N 개의 원소를 처리하며, INCX와 INCY를 이용해 X와 Y 배열의 포인터를 이동하면서 원소에 접근합니다.

### deinter\_cpu

```c
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}
```

함수 이름: deinter\_cpu

입력:

* NX: 인터리브된 X 데이터의 수
* X: 인터리브된 X 데이터 (NX\*B 길이)
* NY: 인터리브된 Y 데이터의 수
* Y: 인터리브된 Y 데이터 (NY\*B 길이)
* B: 배치 크기
* OUT: 인터리브된 출력 데이터 (NX_NY_B 길이)

동작:

* 인터리브된 출력 데이터에서 X와 Y 데이터를 추출하여 X와 Y에 더해줌
* OUT은 인터리브된 형태로 저장된 출력 데이터이며, B개의 배치에 대한 X와 Y 데이터가 번갈아가며 저장되어 있다.

설명:

* 이 함수는 인터리브(interleave)된 데이터를 디인터리브(deinterleave)하여 X와 Y 데이터로 다시 나누어 주는 함수이다.
* 예를 들어, 인터리브된 데이터는 \[x1, y1, x2, y2, x3, y3, ...]와 같이 X와 Y 데이터가 번갈아가며 저장되어 있으며, 이를 X=\[x1, x2, x3, ...], Y=\[y1, y2, y3, ...]로 나누어 주는 것이다.
* 이 함수는 X와 Y를 입력으로 받아서 각각 NX와 NY의 길이를 가지는 데이터를 B개 만큼 받아온다. 이때, OUT은 인터리브된 형태로 저장된 출력 데이터이다.
* 함수는 B개의 배치에 대해 for 루프를 수행하며, 각 배치에서 NX개의 X 데이터와 NY개의 Y 데이터를 추출하여, 인터리브된 형태로 저장된 OUT에서 순서대로 읽어와서 X와 Y에 더해준다.



### inter\_cpu

```c
void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}
```

함수 이름: inter\_cpu

입력:

* NX: int 타입 변수. 배열 X의 요소 개수.
* X: float 타입 배열. NX개의 요소를 가진 배열.
* NY: int 타입 변수. 배열 Y의 요소 개수.
* Y: float 타입 배열. NY개의 요소를 가진 배열.
* B: int 타입 변수. 배치 크기.
* OUT: float 타입 배열. NX+NY개의 요소를 가진 배열.

동작:

* 입력으로 주어진 X와 Y 배열을 섞어서 OUT 배열에 저장한다.
* 배치 사이즈 B만큼 X, Y 배열을 섞어서 OUT 배열에 저장한다.
* OUT 배열의 크기는 (NX+NY)\*B이다.

설명:

* 이 함수는 YOLOv3 딥러닝 네트워크에서 사용되는 함수 중 하나이다.
* 이 함수는 배열 X와 배열 Y의 각 요소를 번갈아가며 OUT 배열에 저장하는 역할을 한다.
* 배열 X와 배열 Y는 각각 다른 데이터를 가지고 있으며, 이 함수는 이들을 하나의 배열로 섞어서 사용한다.
* 배열 X와 Y는 입력 이미지를 처리하기 위한 레이어에서 사용되며, inter\_cpu 함수는 이들을 결합해주는 역할을 한다.
* 이 함수는 입력 데이터를 처리하는 데 있어서 필수적인 역할을 수행하며, 딥러닝 네트워크의 정확도와 속도에 직접적인 영향을 미친다.



### mult\_add\_into\_cpu

```c
void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}
```

함수 이름: mult\_add\_into\_cpu

입력:

* N: 정수값으로, X, Y, Z 배열의 길이
* X: float 형태의 배열 포인터
* Y: float 형태의 배열 포인터
* Z: float 형태의 배열 포인터

동작:&#x20;

* Z 배열에 X\[i]\*Y\[i] 값을 더한다. i는 0부터 N-1까지 반복한다.

설명:&#x20;

* mult\_add\_into\_cpu 함수는 X, Y, Z 배열에서 같은 인덱스 값을 곱한 다음, 그 결과를 Z 배열의 같은 인덱스에 더하는 함수이다.&#x20;
* 이 함수는 벡터 내적과 비슷한 역할을 하며, 두 배열의 요소 곱의 합을 계산하는 것이다.&#x20;
* 이 함수는 다양한 수학 연산에서 사용될 수 있으며, 행렬 연산에서 행렬의 각 행과 열을 곱한 뒤 합한 값을 계산하는 데에도 사용될 수 있다.



### smooth\_l1\_cpu

```c
void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}
```

함수 이름: smooth\_l1\_cpu

입력:&#x20;

* n: 배열의 길이
* pred: 예측값
* truth: 실제값
* delta: gradient
* error: 에러

동작:&#x20;

* 예측값(pred)과 실제값(truth) 사이의 차이(diff)를 구하고, 차이의 절댓값(abs\_val)이 1보다 작으면 error와 delta를 계산하여 반환하고, 그렇지 않으면 error와 delta를 다시 계산하여 반환한다.

설명:&#x20;

* 이 함수는 smooth L1 손실 함수를 계산하는 데 사용된다.&#x20;
* 이 손실 함수는 회귀(regression) 문제에서 많이 사용되며, 주어진 입력 값에 대해 실제 값과 예측 값을 비교하여 오차를 계산한다.&#x20;
* smooth L1 손실 함수는 L2 손실 함수(MSE)와 L1 손실 함수(MAE)를 결합한 것으로, 오차가 크면 L1 손실 함수처럼 절댓값을 사용하고, 작으면 L2 손실 함수처럼 제곱을 사용한다.&#x20;
* 이 함수는 CPU에서 동작하며, 각각의 입력값에 대해 error와 delta를 계산하여 반환한다.



### l1\_cpu

```c
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}
```

함수 이름: l1\_cpu

입력:

* n: 데이터 샘플 수
* pred: 예측값 배열
* truth: 실제값 배열
* delta: 역전파 시 계산되는 델타값이 저장될 배열
* error: 손실값이 저장될 배열

동작:

* 예측값 배열 pred와 실제값 배열 truth 간의 L1 손실을 계산하여 error 배열에 저장하고,
* 예측값과 실제값의 차이가 양수인 경우 delta 배열에 1, 음수인 경우 -1을 저장함으로써 역전파 시 계산되는 델타값을 구한다.

설명:&#x20;

* L1 손실 함수는 예측값과 실제값 간의 차이의 절대값을 손실값으로 사용한다.&#x20;
* L1 손실은 이상치(Outlier)에 대한 민감도가 높아 이상치가 적은 데이터에서는 MSE 손실보다 더 좋은 성능을 보인다.&#x20;
* 이 함수는 CPU에서 동작하며, 입력으로 주어진 예측값과 실제값의 차이를 계산하여 손실값과 델타값을 구한다.



### softmax\_x\_ent\_cpu

```c
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}
```

함수 이름: softmax\_x\_ent\_cpu

입력:

* n: 예측값(pred)과 실제값(truth)의 개수
* pred: 예측값 포인터
* truth: 실제값 포인터
* delta: 역전파 시 오차값을 저장할 포인터
* error: 손실값을 저장할 포인터

동작:&#x20;

* 소프트맥스(softmax) 함수와 크로스 엔트로피(cross-entropy) 손실 함수를 계산하여 오차값과 손실값을 계산합니다.

설명:&#x20;

* 이 함수는 딥러닝에서 분류(classification) 문제에서 사용하는 손실 함수 중 하나인 크로스 엔트로피(cross-entropy) 손실 함수를 계산합니다. 이 손실 함수는 소프트맥스 함수와 함께 분류 문제에서 널리 사용됩니다.
* 함수의 동작은 다음과 같습니다. 예측값(pred)과 실제값(truth)이 주어졌을 때, 먼저 소프트맥스 함수를 이용해 예측값을 확률값으로 변환합니다. 그 다음, 확률값과 실제값을 이용해 크로스 엔트로피 손실 함수를 계산합니다. 손실 함수의 값이 작을수록 모델의 예측값이 실제값에 가깝다는 것을 의미합니다.
* 이 함수는 역전파(backpropagation)를 위해 오차값(delta)도 함께 계산합니다. 역전파를 통해 이 오차값이 모델의 파라미터를 업데이트할 때 사용됩니다.



### logistic\_x\_ent\_cpu

```c
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}
```

함수 이름: logistic\_x\_ent\_cpu

입력:

* n: 예측 값(pred)과 실제 값(truth)의 길이
* pred: 예측 값 배열 포인터
* truth: 실제 값 배열 포인터
* delta: 델타 값 배열 포인터
* error: 오차 값 배열 포인터

동작:&#x20;

* 로지스틱 회귀에서 크로스 엔트로피 손실 함수의 값을 계산하고, 그 값을 이용해 델타 값을 계산하는 함수이다.

설명:&#x20;

이 함수는 n개의 원소를 갖는 pred, truth 배열의 원소 값들을 이용하여 로지스틱 회귀에서 크로스 엔트로피 손실 함수의 값을 계산하고, 그 값을 이용해 델타 값을 계산한다. 이 함수는 다음과 같은 작업을 수행한다.

* i번째 원소에 대해서, t는 실제 값(truth\[i]), p는 예측 값(pred\[i])이다.
* error\[i]에는 크로스 엔트로피 손실 함수의 i번째 원소 값이 저장된다.
* delta\[i]에는 델타 값(i번째 원소에 대한 예측 값과 실제 값의 차이)이 저장된다.

주의할 점은 pred와 truth는 확률값으로 간주되어야 한다는 것이다. 즉, 0과 1 사이의 값이어야 한다.



### l2\_cpu

```c
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}
```

함수 이름: l2\_cpu

입력:

* n: 예측 값(pred)과 실제 값(truth)의 원소 개수
* \*pred: 예측 값 배열
* \*truth: 실제 값 배열
* \*delta: 예측 값과 실제 값의 차이
* \*error: 예측 값과 실제 값의 차이의 제곱

동작:&#x20;

* 예측 값(pred)과 실제 값(truth)을 이용하여 예측 값과 실제 값의 차이(diff)를 계산하고, 해당 차이(diff)를 이용하여 예측 값과 실제 값의 차이의 제곱(error)과 차이(delta)를 계산한다.

설명:&#x20;

* l2(제곱근 오차) 손실 함수를 계산하는 함수로, 딥러닝에서 사용된다.&#x20;
* l2 손실 함수는 예측 값과 실제 값의 차이를 제곱한 값들의 합을 구하는 방식으로 손실을 계산한다.&#x20;
* 이를 이용하여 예측 값과 실제 값 사이의 오차를 계산하고, 이를 역전파(backpropagation)를 통해 학습을 진행한다.



### dot\_cpu

```c
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}
```

함수 이름: dot\_cpu

입력:&#x20;

* N(벡터의 길이)
* X(첫번째 벡터)
* INCX(첫번째 벡터의 증분)
* Y(두번째 벡터)
* INCY(두번째 벡터의 증분)

동작:&#x20;

* 두 벡터 X와 Y의 내적(dot product)을 계산한다. 즉, X\[0]\*Y\[0] + X\[INCX]\*Y\[INCY] + ... + X\[(N-1)\*INCX]\*Y\[(N-1)\*INCY]를 계산한다.

설명:&#x20;

* 두 벡터의 내적은 선형 대수학에서 자주 사용되는 연산 중 하나이다.&#x20;
* 이 함수는 N개의 원소를 가지는 두 벡터 X와 Y의 내적을 계산한다. X와 Y는 각각 INCX, INCY만큼 증분된다. 이 함수는 내적 결과를 반환한다.



### softmax

```c
void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}
```

함수 이름: softmax

입력:

* input: 소프트맥스 함수를 적용할 입력 배열 포인터
* n: 입력 배열의 크기
* temp: 소프트맥스 함수의 온도 매개변수
* stride: 입력 배열에서 요소 간의 간격
* output: 소프트맥스 함수의 출력 배열 포인터

동작:

* 입력 배열의 각 요소에 대해 소프트맥스 함수를 적용하여 출력 배열을 계산
* 입력 배열에서 가장 큰 값(largest)을 찾음
* 출력 배열의 각 요소를 계산할 때, largest로부터 temp로 나눈 값을 지수 함수의 입력으로 사용하여 지수 함수 계산
* 모든 요소의 지수 함수 값을 더하고, 각 요소의 지수 함수 값을 총합으로 나눠 정규화
* 최종적으로 정규화된 값을 출력 배열에 저장

설명:&#x20;

* 소프트맥스 함수는 입력 배열을 확률 분포로 변환하는 함수이며, 각 입력 요소를 정규화된 확률 값으로 변환한다. softmax 함수는 크게 세 부분으로 구성된다.&#x20;
* 먼저 입력 배열에서 가장 큰 값을 찾아 largest에 저장한다.&#x20;
* 그 다음, 입력 배열의 각 요소에 대해 지수 함수를 계산하여 출력 배열을 구한다. 마지막으로 출력 배열의 모든 값을 더하고 총합으로 나눠 정규화한다.&#x20;
* 이를 통해 출력 배열의 값은 모두 0과 1 사이에 있으며, 총합은 1이 된다.&#x20;
* 이렇게 구해진 출력 배열은 입력 배열의 확률 분포를 나타낸다. 소프트맥스 함수는 딥러닝에서 주로 출력층에서 사용되며, 다중 클래스 분류 문제에서 사용된다.



### softmax\_cpu

```c
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}
```

함수 이름: softmax\_cpu

입력:

* input: 입력 배열의 포인터
* n: softmax를 수행할 원소 수
* batch: 배치 크기
* batch\_offset: 배치 간격
* groups: 그룹 수
* group\_offset: 그룹 간격
* stride: 스트라이드 크기
* temp: softmax scaling 매개변수
* output: 출력 배열의 포인터

동작:&#x20;

* 배치와 그룹에 대해 softmax 함수를 적용합니다.&#x20;
* 각각의 배치와 그룹에서, 입력 배열에서 해당 배치와 그룹을 선택하고, softmax 함수를 수행하여 출력 배열에 결과를 저장합니다.

설명:&#x20;

* softmax 함수는 주어진 입력 배열의 값들을 정규화하여 출력 배열의 합이 1이 되도록 합니다. 이 함수는 주로 분류 문제에서 출력층에서 사용됩니다. softmax 함수는 각 원소의 지수값을 취한 후 전체 원소의 합으로 나누어 값을 정규화합니다. softmax\_cpu 함수는 이러한 softmax 함수를 배치와 그룹 단위로 수행합니다.
* batch와 groups는 배열을 분할하는 데 사용됩니다. batch\_offset은 한 배치의 원소 수와 stride를 곱한 값입니다. group\_offset은 한 그룹의 원소 수와 stride를 곱한 값입니다. 이러한 배치 및 그룹의 설정은 입력 배열을 다룰 때 유용합니다.
* temp 매개변수는 softmax 함수의 스케일을 조절하는 데 사용됩니다. 출력 배열에서의 값을 모두 같은 범위 내에 유지하기 위해 사용됩니다. 이 값이 작으면 출력 배열의 차이가 커집니다.
* softmax\_cpu 함수는 입력 배열의 각 배치와 그룹에 대해 softmax 함수를 수행하고 결과를 출력 배열에 저장합니다. 출력 배열의 크기는 입력 배열과 동일합니다.



### upsample\_cpu

```c
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

// if stride : 2
// [1, 2, 3, 4]  -->  [1, 1, 2, 2, 3, 3, 4, 4]
```

함수 이름: upsample\_cpu

입력:

* float \*in: 입력 데이터 포인터
* int w: 입력 데이터의 너비
* int h: 입력 데이터의 높이
* int c: 입력 데이터의 채널 수
* int batch: 배치 크기
* int stride: 업샘플링 스트라이드
* int forward: 순방향 여부(1이면 순방향, 0이면 역방향)
* float scale: 스케일링 계수
* float \*out: 출력 데이터 포인터

동작:&#x20;

* 입력 데이터를 업샘플링하여 출력 데이터를 생성합니다.&#x20;
* 입력 데이터의 각 픽셀은 스트라이드 크기만큼 연속된 출력 데이터의 픽셀들에 복사됩니다.&#x20;
* 업샘플링 스트라이드가 2인 경우, 입력 데이터의 (0,0) 위치는 출력 데이터의 (0,0), (0,1), (1,0), (1,1) 위치에 복사됩니다.&#x20;
* 역방향 계산을 수행하는 경우, 출력 데이터의 변화를 입력 데이터에 역으로 전파합니다.

설명:&#x20;

* upsample\_cpu 함수는 입력 데이터를 업샘플링하여 출력 데이터를 생성하는 함수입니다.&#x20;
* 이미지 분석 분야에서 영상 크기를 확대하거나 해상도를 높이는 등의 용도로 자주 사용됩니다.&#x20;
* 입력 데이터는 4차원 텐서(batch, 채널, 높이, 너비)로 표현되며, 출력 데이터 역시 같은 크기의 4차원 텐서로 생성됩니다.&#x20;
* 스트라이드 크기는 업샘플링 스트라이드 매개변수를 통해 지정할 수 있으며, 스케일링 계수(scale)를 통해 출력 데이터의 값 범위를 조정할 수 있습니다.



### reorg\_cpu

```c
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

// if | w : 2 | h : 2 | c : 4 | stride : 2 |
// [ 1,  2,  3,  4]            [ 1,  5,  2,  6]
// [ 5,  6,  7,  8]     --     [ 9, 13, 10, 14]
// [ 9, 10, 11, 12]     --     [ 3,  7,  4,  8]
// [13, 14, 15, 16]            [11, 15, 12, 16]
```

함수 이름: reorg\_cpu

입력:

* float \*x: 입력 배열 포인터
* int w: 입력 배열의 너비
* int h: 입력 배열의 높이
* int c: 입력 배열의 채널 수
* int batch: 입력 배열의 배치 크기
* int stride: reorg 작업의 stride 값
* int forward: reorg 작업의 방향. forward는 1, backward는 0
* float \*out: 출력 배열 포인터

동작:

* 입력 배열 x를 reorg 작업으로 출력 배열 out으로 변환하는 함수.
* reorg 작업은 입력 배열의 크기를 줄이거나 늘리는 작업을 수행하며, stride 값에 따라 크기가 달라진다.
* forward 값이 1일 경우, 입력 배열 x를 출력 배열 out으로 변환한다.
* forward 값이 0일 경우, 출력 배열 out을 입력 배열 x로 변환한다.
* 입력 배열 x의 각 요소는 출력 배열의 새로운 위치로 이동한다.
* reorg 작업은 다음과 같은 공식으로 수행된다:
  * out\_index = w2 + w \* stride \* (h2 + h \* stride \* (c2 + out\_c \* b))
  * in\_index = i + w \* (j + h \* (k + c \* b))
  * c2 = k % out\_c
  * offset = k / out\_c
  * w2 = i \* stride + offset % stride
  * h2 = j \* stride + offset / stride

설명:

* 입력 배열 x는 4차원 배열로(batch, channel, height, width) 주로 이미지 데이터를 다룰 때 사용된다.
* reorg 작업은 YOLO(Object Detection 알고리즘)에서 사용되는 작업 중 하나로, feature map을 다른 크기로 변환하여 다른 크기의 feature map과 결합하는 작업을 수행한다.
* 이 함수는 CPU 상에서 실행된다.



### flatten

```c
void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){  
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}
```

함수 이름: flatten

입력:

* float \*x: 1차원 배열의 포인터
* int size: 1차원 배열의 크기
* int layers: 2차원 배열의 열의 수
* int batch: 3차원 배열의 배치 크기
* int forward: 1 또는 0 값. 1이면 3차원 배열을 2차원 배열로 펼침. 0이면 2차원 배열을 3차원 배열로 펼침.

동작:

* 3차원 배열을 2차원 배열로 또는 2차원 배열을 3차원 배열로 변환하는 함수
* 3차원 배열의 데이터를 2차원 배열의 형태로 재배치하여, 포인터 swap에 저장
* forward가 1이면 3차원 배열을 2차원 배열로 펼치는 경우이므로, swap의 값을 x로 복사.
* forward가 0이면 2차원 배열을 3차원 배열로 펼치는 경우이므로, swap의 값을 x로 역으로 복사.
* swap 배열을 동적으로 할당하여 사용하므로, 마지막에는 free 함수로 메모리를 해제.

설명:

* 다차원 배열에서 데이터의 순서를 변경하는 함수
* 특히, 딥 러닝에서 입력 데이터를 2차원 배열의 형태로 펼치는 flatten 레이어에서 사용됨.
* 3차원 배열을 2차원 배열로 펼칠 때, 데이터가 가로 방향으로 2차원 배열의 열의 수만큼 연속하게 배치됨.
* 2차원 배열을 3차원 배열로 변환할 때, 데이터가 세로 방향으로 2차원 배열의 열의 수만큼 연속하게 배치됨.



### weighted\_sum\_cpu

```c
void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}
```

함수 이름: weighted\_sum\_cpu

입력:

* float \*a: 첫 번째 입력 배열
* float \*b: 두 번째 입력 배열
* float \*s: 가중치 배열
* int n: 배열의 크기
* float \*c: 결과를 저장할 출력 배열

동작:

* 배열 a, b, s를 입력으로 받아 가중 평균 계산을 통해 c 배열을 계산한다.
* 배열 b가 NULL이 아닌 경우, (1-s\[i])\*b\[i]를 더해준다.

설명:

* 두 개의 배열 a, b와 가중치 배열 s를 이용해 가중 평균 계산을 수행하는 함수이다.
* 결과값은 출력 배열 c에 저장된다.
* 만약 b 배열이 NULL이라면, 두 번째 항은 계산되지 않는다.



### weighted\_delta\_cpu

```c
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}
```

함수 이름: weighted\_delta\_cpu

입력:

* a: float형 배열 포인터
* b: float형 배열 포인터
* s: float형 배열 포인터
* da: float형 배열 포인터
* db: float형 배열 포인터
* ds: float형 배열 포인터
* n: int형 변수
* dc: float형 배열 포인터

동작: 입력으로 받은 배열과 변수들을 사용하여 가중치 계산을 수행합니다.

설명: 이 함수는 뉴럴 네트워크에서 가중치 계산을 수행하는 함수입니다. 입력으로는 두 개의 float형 배열 포인터(a, b), 한 개의 float형 배열 포인터(s), 세 개의 float형 배열 포인터(da, db, ds), int형 변수 n, 그리고 한 개의 float형 배열 포인터 dc가 필요합니다.

이 함수는 입력으로 받은 dc 배열을 사용하여 가중치를 계산합니다. 이때, 가중치 계산은 다음과 같은 방법으로 이루어집니다.

* da\[i] += dc\[i] \* s\[i]: da 배열의 i번째 값에 dc\[i] \* s\[i] 값을 더합니다.
* db\[i] += dc\[i] \* (1-s\[i]): db 배열의 i번째 값에 dc\[i] \* (1-s\[i]) 값을 더합니다.
* ds\[i] += dc\[i] \* (a\[i] - b\[i]): ds 배열의 i번째 값에 dc\[i] \* (a\[i] - b\[i]) 값을 더합니다.

이러한 가중치 계산을 수행하여 결과 값을 da, db, ds 배열에 저장합니다.



### shortcut\_cpu

```c
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}
```

함수 이름: shortcut\_cpu

입력:

* batch: 미니배치 크기
* w1: 이전 레이어의 가로 크기
* h1: 이전 레이어의 세로 크기
* c1: 이전 레이어의 채널 수
* add: 이전 레이어의 출력값
* w2: 현재 레이어의 가로 크기
* h2: 현재 레이어의 세로 크기
* c2: 현재 레이어의 채널 수
* s1: 이전 레이어 출력값의 스케일
* s2: 현재 레이어 출력값의 스케일

동작: shortcut connection(잔여 연결)을 위한 함수로, 이전 레이어의 출력값(add)과 현재 레이어의 출력값(out)을 합하여 최종 출력값을 계산한다.

설명:

* stride와 sample 변수는 서로 반대 개념이며, 가로와 세로 방향의 크기 비율을 나타낸다. 이 값들을 이용해 출력값의 크기를 조정한다.
* 두 레이어의 크기 중에서 더 작은 값(minw, minh, minc)을 이용해 출력값의 인덱스를 계산한다.
* 각 미니배치(b), 채널(k), 세로 방향(j), 가로 방향(i)에 대해 반복문을 수행하며, out\_index와 add\_index를 계산하여 두 값을 가중합하여 최종 출력값을 계산한다.
* 이전 레이어 출력값(add)과 현재 레이어 출력값(out)은 서로 다른 크기를 가질 수 있으므로, stride와 sample 값에 따라 크기를 조정하여 합산한다.
* 출력값(out)에 이전 레이어 출력값의 스케일(s1)과 현재 레이어 출력값의 스케일(s2)을 곱하여 최종 출력값을 계산한다.



### softmax\_x\_ent\_cpu

```c
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}
```

함수 이름: softmax\_x\_ent\_cpu

입력:

* n: 예측값(pred), 실제값(truth), 델타값(delta), 오차값(error)의 개수
* pred: 예측값 배열 포인터
* truth: 실제값 배열 포인터
* delta: 델타값 배열 포인터
* error: 오차값 배열 포인터

동작: softmax와 cross-entropy loss를 계산하는 함수이다.

설명:

* 함수는 예측값(pred), 실제값(truth)을 입력받아서, 델타값(delta)와 오차값(error)을 계산한다.
* softmax 함수는 다중 클래스 분류 문제에서 각 클래스에 대한 예측 확률을 구하기 위해 사용된다. 여기서는 softmax 함수의 결과값이 이미 주어진 것으로 가정한다.
* cross-entropy loss는 예측값과 실제값의 차이를 계산하는 손실 함수 중 하나이다. 예측값과 실제값의 차이를 log 함수를 이용해 계산하며, 실제값이 0인 경우에는 오차를 0으로 계산한다.
* 델타값은 신경망 역전파(backpropagation)에서 사용된다. 역전파에서는 출력층에서의 오차값을 입력층으로 전파시키며, 델타값은 이 과정에서 사용된다.
* 함수는 CPU에서 동작한다.
