# gemm

## GEMM 이란?

참고 자료 : [https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

* `General Matrix to Matrix Multiplication`
* 1979년에 만들어진 BLAS 라이브러리의 일부 입니다.
* 두개의 입력 행렬을 곱해서 출력을 얻는 방법 입니다.

딥러닝에서 대부분의 연산은 `output = input * weight + bias`로 표현이 됩니다. 여기서 `input`, `output`, `weight`를 행렬로 표현해서 GEMM을 사용해 연산할 수 있습니다.

### Fully Connected Layer

`fully connected layer`는 위와 같이 표현할 수 있습니다.

### Convolutional Layer

* `im2col` : 3차원 이미지 배열을 2차원 배열로 변환합니다.

`convolutional layer`는 위와 같이 표현할 수 있습니다. 위 그림의 경우는 `stride`가 `kernel size`와 같은 경우를 의미합니다.

***

## gemm.c

### gemm

gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);

```c
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu(TA,  TB,  M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}
```

함수 이름: gemm&#x20;

입력:

* int TA: 행렬 A의 전치 여부 (0: 전치하지 않음, 1: 전치함)
* int TB: 행렬 B의 전치 여부 (0: 전치하지 않음, 1: 전치함)
* int M: 행렬 C의 행의 수
* int N: 행렬 C의 열의 수
* int K: 행렬 A의 열의 수 (행렬 B의 행의 수와 같아야 함)
* float ALPHA: 스칼라 값
* float \*A: 행렬 A의 포인터
* int lda: 행렬 A의 행 단위 크기
* float \*B: 행렬 B의 포인터
* int ldb: 행렬 B의 행 단위 크기
* float BETA: 스칼라 값
* float \*C: 행렬 C의 포인터
* int ldc: 행렬 C의 행 단위 크기

동작:&#x20;

* 행렬-행렬 곱셈 연산을 수행함.

설명:&#x20;

* 이 함수는 CPU 상에서 행렬-행렬 곱셈 연산을 수행하는 함수이다.&#x20;
* gemm\_cpu 함수를 호출하여 이 연산을 수행한다.&#x20;
* 행렬 A와 행렬 B의 크기와 전치 여부, 스칼라 값 ALPHA와 BETA 등을 입력으로 받고, 연산 결과인 행렬 C를 출력으로 반환한다.



### gemm\_cpu

```c
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
}
```

함수 이름: gemm\_cpu

입력:

* int TA: A 행렬의 전치 여부를 나타내는 플래그
* int TB: B 행렬의 전치 여부를 나타내는 플래그
* int M: C 행렬의 행 수
* int N: C 행렬의 열 수
* int K: A, B 행렬에서 공유하는 차원의 크기
* float ALPHA: A, B 행렬의 곱에 대한 가중치
* float \*A: A 행렬의 포인터
* int lda: A 행렬의 행 당 원소 수
* float \*B: B 행렬의 포인터
* int ldb: B 행렬의 행 당 원소 수
* float BETA: C 행렬에 대한 가중치
* float \*C: C 행렬의 포인터
* int ldc: C 행렬의 행 당 원소 수

동작:&#x20;

* CPU에서 행렬 곱셈 연산을 수행한다.&#x20;
* A, B, C 세 개의 행렬을 인자로 받고, A와 B의 곱에 가중치 ALPHA를 곱한 결과를 C 행렬에 더한다.

설명:&#x20;

* gemm\_cpu 함수는 CPU에서 행렬 곱셈 연산을 수행한다.&#x20;
* 이 함수는 A, B, C 세 개의 포인터와 다양한 인자를 받아서, 행렬 곱셈 연산 결과를 C 행렬에 저장한다.&#x20;
* 함수 내부에서는 TA와 TB 인자를 사용하여 A와 B 행렬이 전치되어 있는지 여부를 확인하고, 이에 따라 gemm\_nn, gemm\_tn, gemm\_nt, gemm\_tt 함수 중 하나를 호출한다.&#x20;
* 이 함수들은 다양한 행렬 곱셈 연산 방법을 구현하고 있다.&#x20;
* 따라서 gemm\_cpu 함수는 이를 이용하여 입력으로 받은 행렬 A, B의 곱에 가중치 ALPHA를 곱한 결과를 C 행렬에 더한다.&#x20;
* 이 때 BETA 인자를 사용하여 기존의 C 행렬 값에 대한 가중치를 조절할 수 있다.



### gemm\_nn

```c
void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
```

함수 이름: gemm\_nn

입력:

* M: 행렬 A의 행의 개수
* N: 행렬 B의 열의 개수
* K: 행렬 A의 열의 개수 또는 행렬 B의 행의 개수
* ALPHA: 행렬 A와 행렬 B의 곱셈 결과에 곱해지는 스칼라 값
* A: 크기 M x K의 행렬 A
* lda: 행렬 A의 leading dimension
* B: 크기 K x N의 행렬 B
* ldb: 행렬 B의 leading dimension
* C: 크기 M x N의 행렬 C

동작:

* 행렬 A와 B를 곱하여 행렬 C를 계산하는 General Matrix Multiply(GEMM) 연산을 수행한다.
* i, k, j 세 개의 for 루프를 사용하여 행렬 C의 각 요소를 계산한다.
* i 루프에서는 행렬 A의 각 행을 순회하며, k 루프에서는 행렬 A의 각 열과 행렬 B의 각 행을 순회하며, j 루프에서는 행렬 B의 각 열을 순회하며 행렬 C의 각 요소를 계산한다.
* OpenMP를 사용하여 병렬 처리한다.

설명:

* General Matrix Multiply(GEMM) 연산은 인공 신경망에서 가장 많이 사용되는 연산 중 하나이다.
* GEMM 연산을 수행하는 방법은 여러 가지가 있으며, 이 함수에서는 A 행렬을 순회하면서 A와 B의 곱을 계산한다.
* OpenMP는 멀티코어 CPU에서 병렬 처리를 수행할 수 있는 라이브러리로, 이를 사용하여 성능을 향상시킨다.



### gemm\_nt

```c
void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}
```

함수 이름: gemm\_nt

입력:

* M: A 행렬의 행 개수
* N: B 행렬의 열 개수
* K: A 행렬의 열 개수 (동시에 B 행렬의 행 개수)
* ALPHA: A와 B 행렬의 곱셈 결과에 곱해질 스칼라 값
* \*A: A 행렬의 포인터
* lda: A 행렬의 행 당 원소 개수
* \*B: B 행렬의 포인터
* ldb: B 행렬의 행 당 원소 개수
* \*C: C 행렬의 포인터
* ldc: C 행렬의 행 당 원소 개수

동작:

* 행렬 A와 B를 곱한 후, C 행렬에 더해주는 연산을 수행한다.
* A와 B 행렬을 곱하기 위해 A는 그대로, B는 전치(transpose)된 형태로 사용된다.
* A의 i번째 행과 B의 j번째 열을 곱한 값을 C의 i번째 행 j번째 열에 누적하여 더해준다.

설명:

* 이 함수는 B 행렬이 전치된 형태로 입력으로 들어올 때 A와 B를 곱한 후 C 행렬에 더해주는 연산을 수행한다.
* 함수 내부에서는 OpenMP를 이용하여 병렬 처리를 수행하며, i, j, k 세 개의 for 루프를 이용하여 행렬의 원소 곱셈 및 덧셈 연산을 수행한다.



### gemm\_tn

```c
void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
```

함수 이름: gemm\_tn

입력:

* int M: 행렬 A의 행의 수
* int N: 행렬 B의 열의 수
* int K: 행렬 A의 열의 수 또는 행렬 B의 행의 수
* float ALPHA: 곱해지는 상수
* float \*A: 행렬 A의 데이터 포인터
* int lda: 행렬 A의 행 간격
* float \*B: 행렬 B의 데이터 포인터
* int ldb: 행렬 B의 행 간격
* float \*C: 출력 행렬 C의 데이터 포인터
* int ldc: 출력 행렬 C의 행 간격

동작:&#x20;

* 행렬 A와 B의 전치행렬인 AT와 BT를 곱하고 ALPHA를 곱한 값을 출력 행렬 C에 더한다.

설명:&#x20;

* 이 함수는 행렬 A와 B의 전치행렬인 AT와 BT를 곱한 결과를 출력하는 함수이다.&#x20;
* 이때 ALPHA를 곱한 값이 출력 행렬 C에 더해진다.&#x20;
* 내부적으로는 OpenMP를 사용하여 병렬 처리를 수행한다.



### gemm\_tt

```c
void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}
```

함수 이름: gemm\_tt

입력:

* int M: 행렬 C의 행 개수
* int N: 행렬 C의 열 개수
* int K: 행렬 A의 열 개수 (행렬 B의 행 개수)
* float ALPHA: 스칼라 값
* float \*A: M x K 크기의 행렬 A
* int lda: 행렬 A의 열 개수
* float \*B: K x N 크기의 행렬 B
* int ldb: 행렬 B의 열 개수
* float \*C: M x N 크기의 행렬 C
* int ldc: 행렬 C의 열 개수

동작:

* 두 개의 행렬 A와 B를 곱한 결과를 행렬 C에 누적한다.
* A와 B는 전치(transpose)되어 있다고 가정한다.
* OpenMP를 사용하여 병렬처리한다.

설명:

* 일반적으로 행렬 곱셈 연산에서는 A x B와 B x A는 다르다. 그러나 gemm\_tt 함수에서는 A와 B 모두 전치된 상태에서 곱셈을 수행하기 때문에 A^T x B^T = (B x A)^T와 같은 결과를 얻는다.
* i, j, k의 순서로 3중 for 루프를 수행하며, 각각 C\[i_ldc+j], A\[i+k_lda], B\[k+j\*ldb]의 값을 참조한다.
* 각각의 C\[i\*ldc+j]의 값을 계산하기 위해 sum 변수를 사용하여 누적 합을 계산한다.
* OpenMP를 사용하여 병렬처리하여 성능을 향상시킨다.

