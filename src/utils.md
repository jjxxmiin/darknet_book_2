# utils

* `rand()` : 0 \~ 32767 사이의 랜덤한 값을 반환합니다.

## what\_time\_is\_it\_now

```c
double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
```

함수 이름: what\_time\_is\_it\_now

입력: 없음

동작:&#x20;

* 현재 시간을 가져와서 double 형태로 반환합니다.

설명:&#x20;

* 함수는 gettimeofday 함수를 사용하여 현재 시간을 가져오고, 이를 double 형태로 변환하여 반환합니다.&#x20;
* tv\_sec은 초를, tv\_usec은 마이크로초를 나타냅니다.



## read\_intlist

```c
int *read_intlist(char *gpu_list, int *ngpus, int d)
{
    int *gpus = 0;
    if(gpu_list){
        int len = strlen(gpu_list);
        *ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++*ngpus;
        }
        gpus = calloc(*ngpus, sizeof(int));
        for(i = 0; i < *ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpus = calloc(1, sizeof(float));
        *gpus = d;
        *ngpus = 1;
    }
    return gpus;
}
```

함수 이름: read\_intlist

입력:

* char \*gpu\_list: GPU 번호 목록을 쉼표로 구분하여 나열한 문자열
* int \*ngpus: GPU 개수를 저장하기 위한 포인터 변수
* int d: GPU 번호 목록이 없을 경우 사용할 기본 GPU 번호

동작:

* 입력된 gpu\_list 문자열을 파싱하여 포함된 GPU 개수를 구한다.
* 구한 GPU 개수만큼 메모리를 동적 할당하여 gpus 배열을 생성한다.
* gpu\_list 문자열에서 ','를 기준으로 GPU 번호를 파싱하여 gpus 배열에 저장한다.
* 만약 gpu\_list가 NULL일 경우에는 d 값을 이용하여 1개의 GPU 번호를 갖는 배열을 생성한다.

설명:&#x20;

* read\_intlist 함수는 char 형식의 문자열인 gpu\_list를 파싱하여 정수형 배열인 gpus를 생성하고 이를 반환한다. gpu\_list는 쉼표로 구분된 GPU 번호 목록을 문자열로 갖는다. 이 함수는 gpu\_list가 NULL일 경우, 1개의 GPU 번호를 갖는 배열을 생성하고 d 값을 이용하여 GPU 번호를 저장한다.&#x20;
* 그렇지 않은 경우, gpu\_list 문자열을 파싱하여 ','로 구분된 GPU 번호를 추출하여 gpus 배열에 저장한다. 이때, gpus 배열의 크기는 gpu\_list에 포함된 GPU의 개수와 동일하게 할당된다. 함수는 gpus 배열을 반환하며, GPU의 개수는 ngpus 포인터 변수를 통해 저장된다.

## read\_map

```c
int *read_map(char *filename)
{
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    while((str=fgetl(file))){
        ++n;
        map = realloc(map, n*sizeof(int));
        map[n-1] = atoi(str);
    }
    return map;
}
```

함수 이름: read\_map

입력:

* char \*filename: 읽어들일 맵 파일의 경로와 이름

동작:

* 입력된 파일을 읽어들인다.
* 파일에서 한 줄씩 읽어들이면서 맵을 구성한다.
* 맵의 크기를 늘리면서 맵을 구성하는 정수형 배열인 map을 동적 할당한다.
* 파일을 모두 읽으면 구성된 map을 반환한다.

설명:&#x20;

* read\_map 함수는 입력으로 받은 파일에서 맵을 구성하는 정수형 배열을 반환한다.&#x20;
* 파일은 filename으로 지정된 경로에 있는 파일이다. 함수는 파일을 열고 파일에서 한 줄씩 읽어들이면서 맵을 구성한다.&#x20;
* 맵은 정수형 배열로 구성되며, 파일에서 읽어들인 문자열을 정수로 변환하여 배열에 저장한다.&#x20;
* 파일에서 한 줄씩 읽을 때마다, 맵의 크기를 늘리면서 배열을 동적으로 할당한다. 함수는 파일을 모두 읽고 맵을 구성한 배열을 반환한다.



## sorta\_shuffle

```c
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections)
{
    size_t i;
    for(i = 0; i < sections; ++i){
        size_t start = n*i/sections;
        size_t end = n*(i+1)/sections;
        size_t num = end-start;
        shuffle(arr+(start*size), num, size);
    }
}
```

함수 이름: read\_map

입력:

* char \*filename: 읽어들일 맵 파일의 경로와 이름

동작:

* 입력된 파일을 읽어들인다.
* 파일에서 한 줄씩 읽어들이면서 맵을 구성한다.
* 맵의 크기를 늘리면서 맵을 구성하는 정수형 배열인 map을 동적 할당한다.
* 파일을 모두 읽으면 구성된 map을 반환한다.

설명:&#x20;

* read\_map 함수는 입력으로 받은 파일에서 맵을 구성하는 정수형 배열을 반환한다.&#x20;
* 파일은 filename으로 지정된 경로에 있는 파일이다. 함수는 파일을 열고 파일에서 한 줄씩 읽어들이면서 맵을 구성한다.&#x20;
* 맵은 정수형 배열로 구성되며, 파일에서 읽어들인 문자열을 정수로 변환하여 배열에 저장한다.&#x20;
* 파일에서 한 줄씩 읽을 때마다, 맵의 크기를 늘리면서 배열을 동적으로 할당한다. 함수는 파일을 모두 읽고 맵을 구성한 배열을 반환한다.



## shuffle

```c
void shuffle(void *arr, size_t n, size_t size)
{
    size_t i;
    void *swp = calloc(1, size);
    for(i = 0; i < n-1; ++i){
        size_t j = i + rand()/(RAND_MAX / (n-i)+1);
        memcpy(swp,          arr+(j*size), size);
        memcpy(arr+(j*size), arr+(i*size), size);
        memcpy(arr+(i*size), swp,          size);
    }
}
```

함수 이름: shuffle

입력:

* void \*arr: 섞을 배열
* size\_t n: 배열의 요소 개수
* size\_t size: 배열 요소의 크기

동작:

* 배열의 요소들을 무작위로 섞는다.

설명:&#x20;

* shuffle 함수는 void 포인터로 전달된 배열을 무작위로 섞는다.&#x20;
* 배열은 n개의 요소를 가지고 있으며, 각 요소의 크기는 size이다. 함수는 무작위로 선택한 요소와 다른 요소의 위치를 교환하여 배열을 섞는다.&#x20;
* 이를 위해 rand 함수를 사용하여 배열에서 선택할 요소의 인덱스를 무작위로 선택하고, 선택한 요소와 다른 요소의 값을 교환한다.&#x20;
* 값을 교환할 때는 memcpy 함수를 사용하여 두 요소의 값을 복사한다. 이 과정을 배열의 모든 요소에 대해 반복하면, 배열의 요소들이 무작위로 섞인다. 함수는 배열을 직접 수정하기 때문에 반환 값이 없다.



## random\_index\_order

```c
int *random_index_order(int min, int max)
{
    int *inds = calloc(max-min, sizeof(int));
    int i;
    for(i = min; i < max; ++i){
        inds[i] = i;
    }
    for(i = min; i < max-1; ++i){
        int swap = inds[i];
        int index = i + rand()%(max-i);
        inds[i] = inds[index];
        inds[index] = swap;
    }
    return inds;
}
```

함수 이름: random\_index\_order

입력:

* int min: 랜덤 인덱스의 최소값
* int max: 랜덤 인덱스의 최대값 (최대값은 포함되지 않음)

동작:

* min에서 max-1 사이의 정수들을 무작위로 섞은 배열을 반환한다.

설명:&#x20;

* random\_index\_order 함수는 min에서 max-1 사이의 정수들을 무작위로 섞은 배열을 반환한다.&#x20;
* 함수는 배열을 만들고, 인덱스 값을 배열에 저장한다. 그 다음, 배열에서 무작위로 선택한 요소와 다른 요소의 값을 교환하여 배열을 무작위로 섞는다.&#x20;
* 이를 위해 rand 함수를 사용하여 배열에서 선택할 요소의 인덱스를 무작위로 선택하고, 선택한 요소와 다른 요소의 값을 교환한다.&#x20;
* 값을 교환할 때는 swap 변수를 사용하여 두 요소의 값을 복사한다. 이 과정을 배열의 모든 요소에 대해 반복하면, 배열의 요소들이 무작위로 섞인다. 함수는 섞인 정수들이 저장된 배열을 반환한다.



## del\_arg

```c
void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}
```

함수 이름: del\_arg

입력:

* int argc: 명령줄 인수의 개수
* char \*\*argv: 명령줄 인수의 배열
* int index: 삭제할 인수의 인덱스

동작:

* argv 배열에서 주어진 인덱스에 해당하는 인수를 삭제한다.

설명:&#x20;

* del\_arg 함수는 주어진 명령줄 인수 배열에서 주어진 인덱스에 해당하는 인수를 삭제한다.&#x20;
* 함수는 배열을 순회하면서 주어진 인덱스 이후의 인수를 한 칸씩 앞으로 이동시키고, 배열의 끝에 null 값을 추가하여 삭제한 인수의 자리를 비운다.&#x20;
* 이를 위해 for 문을 사용하여 인덱스 이후의 인수들을 한 칸씩 앞으로 이동시키고, 마지막에 null 값을 추가하여 배열의 크기를 하나 줄인다.&#x20;
* 함수는 원래의 명령줄 인수 배열을 변경한다. 따라서 이 함수를 사용하면 원래의 명령줄 인수 배열이 수정된다는 것에 주의해야 한다.



## find\_arg

```c
int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}
```

함수 이름: find\_arg

입력:

* argc: 인자 개수
* argv: 인자 리스트
* arg: 찾으려는 인자

동작:

* argv에서 arg와 동일한 값을 찾는다.
* 찾으면 해당 값을 argv에서 삭제(del\_arg 함수 호출)하고 1을 반환한다.
* 못 찾으면 0을 반환한다.

설명:&#x20;

* 인자로 주어진 argv 리스트에서 arg와 동일한 값을 찾아서 해당 값을 삭제하는 함수이다.&#x20;
* argv는 문자열 배열로 구성되어 있으며, argc는 argv 배열의 크기를 나타낸다.&#x20;
* argv에서 arg를 찾으면 해당 값을 삭제(del\_arg 함수 호출)하고 1을 반환한다.&#x20;
* arg를 찾지 못하면 0을 반환한다.



## find\_int\_arg

```c
int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}
```

함수 이름: find\_int\_arg

입력:

* argc: 명령줄 인자의 개수
* argv: 명령줄 인자를 담고 있는 문자열 배열
* arg: 찾으려는 인자의 이름
* def: 기본값으로 사용할 정수값

동작:

* argv 배열에서 arg 인자를 찾고, 그 인자 바로 다음에 오는 정수값을 반환한다. arg 인자가 없는 경우, def 값을 반환한다.

설명:

* 명령줄에서 인자를 찾아 해당하는 정수값을 반환하는 함수이다.
* argv 배열에서 arg 인자를 찾으면, 그 인자 바로 다음에 오는 문자열을 정수값으로 변환하여 def 변수에 저장한다.
* 그리고나서 argv 배열에서 arg와 그 인자 바로 다음에 오는 문자열 두 개를 삭제한다.
* 만약 arg 인자가 argv 배열에 없다면, 함수는 def 값을 반환한다.



## find\_float\_arg

```c
float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atof(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}
```

함수 이름: find\_float\_arg

입력:

* argc: 인자 개수
* argv: 인자 리스트
* arg: 찾을 인자 이름
* def: 찾을 인자가 없을 경우 반환할 기본값

동작:&#x20;

* 주어진 인자 리스트에서 인자 이름이 arg인 경우 해당 인자 다음에 오는 값을 실수형으로 변환하여 반환한다.&#x20;
* 인자를 찾았을 경우 인자 리스트에서 해당 인자와 다음 인자를 삭제한다. 만약 인자 이름이 arg인 인자가 없을 경우 기본값 def를 반환한다.

설명:&#x20;

* 이 함수는 주어진 인자 리스트에서 특정 인자 이름을 찾아 그 값을 실수형으로 반환하는 함수이다.&#x20;
* 입력으로 받은 argc와 argv는 프로그램 실행 시 사용자가 입력한 인자들의 리스트이다.&#x20;
* 이 함수는 리스트에서 arg 이름을 가진 인자를 찾아 그 다음에 오는 값을 실수형으로 변환하여 반환한다.&#x20;
* 인자를 찾았을 경우 해당 인자와 그 다음 인자를 인자 리스트에서 삭제한다. 인자를 찾지 못한 경우 def 값을 반환한다.



## find\_char\_arg

```c
char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}
```

함수 이름: find\_char\_arg

입력:

* argc: 명령줄 인자(argument)의 개수
* argv: 명령줄 인자들의 배열
* arg: 찾으려는 인자의 이름
* def: 찾지 못했을 경우 반환할 기본값

동작:

* argv 배열에서 arg와 동일한 문자열을 찾고, 해당 문자열 다음에 오는 인자 값을 def로 설정한다.
* 인자 값이 없으면 기본값 def를 반환한다.
* 찾은 인자와 그 다음 인자 값은 argv 배열에서 삭제된다.

설명:&#x20;

* 이 함수는 주어진 명령줄 인자에서 특정 인자를 찾아 그 인자 다음에 오는 인자 값을 반환하는 함수이다. 만약 해당 인자가 없으면 기본값을 반환한다.&#x20;
* 예를 들어, find\_char\_arg(argc, argv, "-f", "data.txt")는 argv 배열에서 "-f" 인자가 있는지 찾아, 그 다음에 오는 인자 값을 반환한다.&#x20;
* 만약 "-f" 인자가 없으면 기본값 "data.txt"를 반환한다. 함수는 찾은 인자와 그 다음 인자를 argv 배열에서 삭제한다.



## basecfg

```c
char *basecfg(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
    while((next = strchr(c, '/')))
    {
        c = next+1;
    }
    c = copy_string(c);
    next = strchr(c, '.');
    if (next) *next = 0;
    return c;
}
```

함수 이름: basecfg&#x20;

입력:&#x20;

* 문자열 포인터 (cfgfile)

동작:

1. 입력된 문자열에서 '/'를 찾아서 '/' 다음 문자열을 c에 저장
2. c에서 '.'을 찾아서 '.' 이후 문자열을 삭제하고 반환

설명:&#x20;

* cfgfile은 파일 경로를 포함한 파일 이름과 확장자가 포함된 문자열이다.&#x20;
* 이 함수는 입력된 문자열에서 '/'를 찾아서 파일 이름과 확장자가 제외된 파일 이름만을 반환한다.&#x20;
* 예를 들어, cfgfile이 "/home/user/my\_config.cfg"라면, "my\_config" 문자열이 반환된다.



## alphanum\_to\_int

```c
int alphanum_to_int(char c)
{
    return (c < 58) ? c - 48 : c-87;
}
```

함수 이름: alphanum\_to\_int

입력:&#x20;

* 문자 하나(c)

동작:&#x20;

* 입력받은 문자(c)가 숫자 또는 알파벳(a\~f)인 경우, 해당하는 10진수 값을 반환한다.

설명:&#x20;

* 입력받은 문자(c)가 숫자인 경우, 해당하는 정수로 변환하여 반환한다. 알파벳(a,f)인 경우, 10,15를 나타내는 숫자로 변환하여 반환한다.



## int\_to\_alphanum

```c
char int_to_alphanum(int i)
{
    if (i == 36) return '.';
    return (i < 10) ? i + 48 : i + 87;
}
```

함수 이름: int\_to\_alphanum 입력: 정수형 변수 i

동작: 입력된 i에 따라서 다음과 같은 동작을 수행합니다.

* i가 36과 같으면, 문자 '.'을 반환합니다.
* i가 10보다 작으면, 48을 더한 값을 문자형으로 반환합니다.
* i가 10 이상이면, 87을 더한 값을 문자형으로 반환합니다.

설명:&#x20;

* 이 함수는 정수형 변수를 문자형으로 변환하는 함수입니다.&#x20;
* 반환된 문자형 값은 문자열에서 숫자 대신 사용될 수 있습니다.&#x20;
* 예를 들어, 0부터 35까지의 정수값을 각각 '0'부터 '9', 'a'부터 'z', '.'으로 변환할 수 있습니다.





## pm

```c
void pm(int M, int N, float *A)
{
    int i,j;
    for(i =0 ; i < M; ++i){
        printf("%d ", i+1);
        for(j = 0; j < N; ++j){
            printf("%2.4f, ", A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}
```

함수 이름: pm

입력:

* M: A의 행 개수
* N: A의 열 개수
* A: M x N 크기의 2차원 배열(평면 매트릭스)

동작:&#x20;

* 주어진 2차원 배열 A를 출력하는 함수이다.
* 각 행의 첫 열에는 해당 행의 인덱스(i+1)가 출력된다.
* 각 행의 원소들은 ','와 함께 공백을 두고 출력된다.

설명:&#x20;

* M x N 크기의 2차원 배열 A를 출력하는 함수이다.
* 이중 for 루프를 통해 각 행과 열을 반복하면서 배열 A의 값을 출력한다.
* 각 행의 첫 열에는 해당 행의 인덱스(i+1)가 출력되며, 나머지 열의 값들은 ','와 함께 공백을 두고 출력된다.



## find\_replace

```c
void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = {0};
    char *p;

    sprintf(buffer, "%s", str);
    if(!(p = strstr(buffer, orig))){  // Is 'orig' even in 'str'?
        sprintf(output, "%s", str);
        return;
    }

    *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p+strlen(orig));
}
```

함수 이름: find\_replace

입력:

* char \*str: 대상 문자열
* char \*orig: 치환 대상 문자열
* char \*rep: 치환될 문자열
* char \*output: 치환 결과가 저장될 문자열

동작:

* 대상 문자열(str)에서 치환 대상 문자열(orig)을 찾아 치환 문자열(rep)로 변경하여 결과를 output 문자열에 저장하는 함수이다.
* 대상 문자열에서 치환 대상 문자열이 없으면 그대로 output에 저장한다.

설명:

* buffer 배열에 대상 문자열(str)을 복사한다.
* strstr 함수를 사용하여 buffer에서 치환 대상 문자열(orig)이 있는지 검사한다.
* 치환 대상 문자열이 없으면 그대로 대상 문자열(str)을 output에 복사한다.
* 치환 대상 문자열이 있으면 해당 위치의 문자를 '\0'으로 변경하여 이전까지의 문자열을 buffer에 복사한다.
* sprintf 함수를 사용하여 buffer, 치환 문자열(rep), 치환 대상 문자열(orig) 다음의 문자열을 순서대로 결합하여 output에 저장한다.



## sec

```c
float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}
```

함수 이름: sec

입력:&#x20;

* clock\_t 형식의 clocks 변수

동작:&#x20;

* 입력된 clocks 변수 값을 CLOCKS\_PER\_SEC로 나누어서 초 단위로 반환합니다.

설명:&#x20;

* CLOCKS\_PER\_SEC는 시스템당 초당 클록 틱 수입니다.&#x20;
* sec 함수는 입력된 clocks 값을 CLOCKS\_PER\_SEC로 나누어서 초 단위로 반환합니다.&#x20;
* 이 함수는 주로 실행 시간을 측정하는 데 사용됩니다.



## top\_k

```c
void top_k(float *a, int n, int k, int *index)
{
    int i,j;
    for(j = 0; j < k; ++j) index[j] = -1;
    for(i = 0; i < n; ++i){
        int curr = i;
        for(j = 0; j < k; ++j){
            if((index[j] < 0) || a[curr] > a[index[j]]){
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}
```

함수 이름: top\_k

입력:

* float \*a : n개의 원소를 가진 float 배열 a
* int n : 배열 a의 크기
* int k : 구하고자 하는 상위 k개의 원소 개수
* int \*index : 구해진 k개의 원소의 인덱스를 저장할 int 배열

동작:

* n개의 원소를 가진 float 배열 a에서 가장 큰 값부터 k개의 값을 찾아서 각 원소의 인덱스를 int 배열 index에 저장하는 함수이다.
* 초기에는 index 배열의 각 원소를 -1로 설정한다.
* 배열 a를 순회하면서 현재 인덱스를 curr에 저장한다.
* 상위 k개의 값이 저장된 index 배열에서 현재 원소 a\[curr]가 어떤 값과 비교하여 가장 큰 값인지 비교한다.
* a\[curr]이 더 크다면 curr와 index\[j]를 교환한다.
* 반복문이 끝나면 index 배열에는 a에서 가장 큰 k개의 원소의 인덱스가 저장된다.

설명:

* 이 함수는 배열에서 가장 큰 k개의 값을 찾는 문제에서 많이 사용된다.
* 시간 복잡도는 O(nk)이다. n이 매우 큰 경우, 다른 알고리즘을 사용하는 것이 더 효율적일 수 있다.



## error

```c
void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}
```

함수 이름: error

입력:&#x20;

* const char \*s: 에러 메시지를 포함한 문자열 포인터

동작:&#x20;

* error 함수는 perror 함수를 사용하여 인자로 받은 문자열 포인터 s를 출력하고, assert(0)를 호출하여 프로그램 실행을 중단시키며, exit(-1)을 호출하여 프로그램을 종료시킨다.

설명:&#x20;

* error 함수는 프로그램에서 에러가 발생했을 때 사용되는 함수로, 인자로 받은 에러 메시지를 출력하고 프로그램을 중단시키며 종료시킨다.&#x20;
* assert(0)는 프로그램이 실행 중에 강제로 종료되도록 하는 매크로 함수이며, exit(-1)은 프로그램을 종료하는 함수이다.&#x20;
* 따라서 error 함수를 호출하면 프로그램이 에러가 발생한 지점에서 중단되고 종료된다.



## read\_file

```c
unsigned char *read_file(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    size_t size;

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char *text = calloc(size+1, sizeof(char));
    fread(text, 1, size, fp);
    fclose(fp);
    return text;
}
```

함수 이름: read\_file

입력:&#x20;

* char \*filename: 읽어들일 파일의 이름을 포함한 문자열 포인터

동작:&#x20;

* read\_file 함수는 인자로 받은 파일 이름을 이용하여 파일을 읽어들인 다음, 파일 내용을 unsigned char 타입의 포인터로 반환한다.&#x20;
* 함수 내부에서는 파일 포인터를 이용하여 파일의 크기를 계산하고, 파일 크기만큼의 메모리를 동적으로 할당한 후 파일 내용을 읽어들인다.&#x20;
* 마지막으로 파일 포인터를 닫고, 파일 내용을 담고 있는 포인터를 반환한다.

설명:&#x20;

* read\_file 함수는 파일을 읽어들이기 위한 함수로, 인자로 받은 파일 이름을 이용하여 파일 포인터를 열고, 파일 내용을 읽어들인다.&#x20;
* 이때, 파일 크기를 계산하기 위해 fseek와 ftell 함수를 사용한다. 파일의 크기만큼 동적으로 할당된 메모리는 파일 내용을 저장하기 위한 용도로 사용된다.&#x20;
* 마지막으로 파일 포인터를 닫고, 파일 내용을 담고 있는 포인터를 반환한다.&#x20;
* 이러한 방식으로 read\_file 함수는 파일을 읽어들일 때 효율적이고 안전하게 처리할 수 있다.



## malloc\_error

```c
void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}
```

함수 이름: malloc\_error

입력:&#x20;

* 없음

동작:&#x20;

* malloc\_error 함수는 동적으로 메모리를 할당할 때 오류가 발생한 경우 호출되어, stderr에 "Malloc error" 메시지를 출력하고, exit(-1)을 호출하여 프로그램을 종료시킨다.

설명:&#x20;

* malloc\_error 함수는 프로그램이 동적으로 메모리를 할당할 때 발생하는 오류를 처리하기 위한 함수이다.&#x20;
* 메모리 할당에 실패하면, 프로그램은 종료되기 전에 malloc\_error 함수가 호출되어 stderr에 "Malloc error" 메시지를 출력하고, exit(-1)을 호출하여 프로그램을 종료시킨다.&#x20;
* 이를 통해 프로그램이 동적으로 메모리를 할당할 때 오류가 발생하면 적절한 오류 메시지를 출력하고 프로그램을 안전하게 종료할 수 있다.



## file\_error

```c
void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

```

함수 이름: file\_error

입력:&#x20;

* char \*s: 파일 이름을 포함한 문자열 포인터

동작:&#x20;

* file\_error 함수는 파일을 열지 못한 경우 호출되어, stderr에 "Couldn't open file: "와 파일 이름을 출력하고, exit(0)을 호출하여 프로그램을 종료시킨다.

설명:&#x20;

* file\_error 함수는 파일을 열지 못한 경우를 처리하기 위한 함수이다.&#x20;
* 함수 내부에서는 인자로 받은 파일 이름을 출력하며, stderr에 "Couldn't open file: "와 함께 출력한다.&#x20;
* 이후 exit(0)을 호출하여 프로그램을 종료시킨다.&#x20;
* 이를 통해 프로그램이 파일을 열지 못했을 때 적절한 오류 메시지를 출력하고 프로그램을 안전하게 종료할 수 있다.



## split\_str

```c
list *split_str(char *s, char delim)
{
    size_t i;
    size_t len = strlen(s);
    list *l = make_list();
    list_insert(l, s);
    for(i = 0; i < len; ++i){
        if(s[i] == delim){
            s[i] = '\0';
            list_insert(l, &(s[i+1]));
        }
    }
    return l;
}
```

함수 이름: split\_str

입력:&#x20;

* char \*s: 분리할 문자열 포인터
* char delim - 구분자

동작:&#x20;

* split\_str 함수는 입력 문자열을 구분자를 기준으로 분리하여 링크드 리스트에 저장하고, 해당 리스트의 포인터를 반환한다.

설명:&#x20;

* split\_str 함수는 입력 문자열을 구분자를 기준으로 분리하여 링크드 리스트에 저장하는 함수이다.&#x20;
* 함수 내부에서는 문자열 길이를 계산하고, make\_list 함수를 사용하여 새로운 링크드 리스트를 생성한다.&#x20;
* 이후 list\_insert 함수를 사용하여 분리된 문자열을 리스트에 저장하고, 구분자를 '\0'으로 변경하여 다음 문자열의 시작 위치를 가리키도록 한다.&#x20;
* 분리된 문자열은 리스트에 새로운 노드로 추가되며, 마지막으로 생성된 리스트의 포인터를 반환한다.&#x20;
* 이를 통해 프로그램에서 입력 문자열을 구분자를 기준으로 분리하고, 각각의 문자열을 링크드 리스트에 저장하여 사용할 수 있다.



## strip

```c
void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}
```

함수 이름: strip

입력:&#x20;

* char \*s: 공백 문자를 제거할 문자열 포인터

동작:&#x20;

* strip 함수는 입력 문자열에서 공백 문자를 제거하고, 결과를 원본 문자열에 덮어쓰기 한다.

설명:&#x20;

* strip 함수는 입력 문자열에서 공백 문자를 제거하는 함수이다.&#x20;
* 함수 내부에서는 문자열 길이를 계산하고, offset 변수를 사용하여 제거한 문자의 수를 계산한다.&#x20;
* 이후 문자열을 순회하며, 공백 문자인 경우 offset 값을 증가시키고, 그렇지 않은 경우 해당 문자를 offset 만큼 앞으로 이동시켜 덮어쓰기 한다.&#x20;
* 마지막으로, 문자열 끝에 널 문자를 추가하여 문자열의 끝을 표시한다.&#x20;
* 이를 통해 프로그램에서 입력 문자열에서 공백 문자를 제거할 수 있으며, 원본 문자열에 직접 덮어쓰기 하여 메모리를 효율적으로 관리할 수 있다.



## strip\_char

```c
void strip_char(char *s, char bad)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==bad) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}
```

함수 이름: strip\_char

입력:

* char \*s: 제거할 문자가 포함된 문자열 포인터
* char bad: 제거할 문자

동작:&#x20;

* strip\_char 함수는 입력 문자열에서 지정된 문자를 제거하고, 결과를 원본 문자열에 덮어쓰기 한다.

설명:&#x20;

* strip\_char 함수는 입력 문자열에서 지정된 문자를 제거하는 함수이다.&#x20;
* 함수 내부에서는 문자열 길이를 계산하고, offset 변수를 사용하여 제거한 문자의 수를 계산한다.&#x20;
* 이후 문자열을 순회하며, 지정된 문자인 경우 offset 값을 증가시키고, 그렇지 않은 경우 해당 문자를 offset 만큼 앞으로 이동시켜 덮어쓰기 한다.&#x20;
* 마지막으로, 문자열 끝에 널 문자를 추가하여 문자열의 끝을 표시한다.&#x20;
* 이를 통해 프로그램에서 입력 문자열에서 지정된 문자를 제거할 수 있으며, 원본 문자열에 직접 덮어쓰기 하여 메모리를 효율적으로 관리할 수 있다.



## free\_ptrs

```c
void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}
```

함수 이름: free\_ptrs

입력:

* void \*\*ptrs: 메모리 할당이 된 포인터 배열
* int n: 배열의 크기

동작:&#x20;

* free\_ptrs 함수는 메모리 할당이 된 포인터 배열을 순회하며 할당된 메모리를 해제하고, 배열 자체를 해제한다.

설명:&#x20;

* free\_ptrs 함수는 메모리 할당이 된 포인터 배열을 순회하며 각 포인터가 가리키는 메모리를 해제한다.&#x20;
* 이후, 배열 자체를 해제하여 메모리 누수를 방지한다. 이 함수를 사용하면 여러 포인터가 동적으로 할당된 경우, 이를 일괄적으로 해제할 수 있다.&#x20;
* 이러한 기능을 통해 메모리 관리를 효율적으로 수행할 수 있고, 프로그램의 안정성과 성능을 향상시킬 수 있다.



## fgetl

```c
char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = realloc(line, size*sizeof(char));
            if(!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}
```

함수 이름: fgetl

입력:&#x20;

* 파일 포인터(fp)

동작:&#x20;

* 파일에서 한 줄을 읽어와 문자열 포인터를 반환한다. 파일 끝에 도달하면 NULL을 반환한다.

설명:&#x20;

* 이 함수는 파일에서 한 줄을 읽어와 NULL 종료 문자열로 반환하는 함수이다.&#x20;
* 파일 포인터(fp)를 입력으로 받으며, 파일의 끝에 도달하면 NULL을 반환한다.&#x20;
* 이 함수는 파일에서 읽어온 데이터의 크기를 기준으로 동적으로 메모리를 할당하며, 파일에서 읽은 줄이 지정한 버퍼 크기보다 크면 버퍼 크기를 늘려준다.&#x20;
* 반환된 문자열 포인터는 메모리를 해제해야 한다.



## read\_int

```c
int read_int(int fd)
{
    int n = 0;
    int next = read(fd, &n, sizeof(int));
    if(next <= 0) return -1;
    return n;
}
```

함수 이름: read\_int

입력:&#x20;

* 파일 디스크립터(fd)

동작:&#x20;

* 파일 디스크립터에서 4바이트(int 자료형 크기)를 읽어서 정수형으로 변환하여 반환한다.&#x20;
* 만약 파일 디스크립터에서 읽을 데이터가 없다면 -1을 반환한다.

설명:&#x20;

* 파일 디스크립터에서 정수형 데이터를 읽어올 때 사용되는 함수이다.&#x20;
* 반환값이 -1인 경우는 파일 디스크립터에서 더 이상 읽을 데이터가 없다는 것을 의미한다.



## write\_int

```c
void write_int(int fd, int n)
{
    int next = write(fd, &n, sizeof(int));
    if(next <= 0) error("read failed");
}
```

함수 이름: write\_int

입력:&#x20;

* 파일 디스크립터(fd)
* 정수(n)

동작:&#x20;

* 주어진 파일 디스크립터에 주어진 정수를 sizeof(int)만큼 쓴다.

설명:&#x20;

* 파일 디스크립터(fd)에 주어진 정수(n)을 sizeof(int)만큼 쓰는 함수이다.&#x20;
* 만약 쓰기 동작이 실패하면 error 함수를 호출하여 에러를 출력한다.



## read\_all\_fail

```c
int read_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}
```

함수 이름: read\_all\_fail

입력:&#x20;

* 파일 디스크립터(fd)
* 문자열 버퍼(buffer)
* 읽을 바이트 수(bytes)

동작:&#x20;

* 파일 디스크립터로부터 지정된 바이트 수만큼 읽고, 문자열 버퍼에 저장한다.&#x20;
* 파일 끝을 만나거나 읽기에 실패할 경우 1을 반환하고, 그렇지 않으면 0을 반환한다.

설명:&#x20;

* 입력한 파일 디스크립터에서 문자열 버퍼로 지정된 바이트 수만큼 읽는 함수이다.&#x20;
* 읽은 데이터는 문자열 버퍼에 저장된다.&#x20;
* 만약 파일 끝에 도달하거나 읽기 작업이 실패하면 1을 반환하고, 그렇지 않으면 0을 반환한다.



## write\_all\_fail

```c
int write_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}
```

함수 이름: write\_all\_fail

입력:

* int fd: 파일 디스크립터 (파일에 대한 I/O 연산을 위한 식별자)
* char \*buffer: 쓰여질 데이터가 저장된 문자열 포인터
* size\_t bytes: 쓰여질 데이터의 크기

동작:&#x20;

* 주어진 파일 디스크립터를 사용하여 주어진 크기의 데이터를 주어진 버퍼에서 파일에 쓰는 함수입니다.&#x20;
* 데이터를 파일에 쓰는 중에 오류가 발생하면 1을 반환하고 그렇지 않으면 0을 반환합니다.

설명:&#x20;

* 파일 디스크립터에 대한 write() 시스템 콜을 사용하여 주어진 크기의 데이터를 파일에 쓰는 함수입니다.&#x20;
* write() 시스템 콜은 지정된 파일 디스크립터를 사용하여 데이터를 파일에 쓰고, 쓰여진 바이트 수를 반환합니다.&#x20;
* 함수는 이러한 write() 시스템 콜을 여러 번 호출하여 주어진 데이터의 크기만큼 모두 쓰기를 시도합니다.&#x20;
* 만약 어떤 이유로 인해 write() 시스템 콜이 실패하면 1을 반환하고, 그렇지 않으면 0을 반환합니다.



## read\_all

```c
void read_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) error("read failed");
        n += next;
    }
}
```

함수 이름: read\_all

입력:

* fd: 파일 디스크립터 (file descriptor)
* buffer: 읽은 데이터를 저장할 버퍼
* bytes: 읽을 바이트 수

동작:

* 파일 디스크립터에서 bytes만큼 데이터를 읽어 buffer에 저장한다.
* 데이터를 모두 읽을 때까지 계속해서 read() 시스템 콜을 호출한다.

설명:

* 파일 디스크립터에서 지정한 크기만큼의 데이터를 읽는 함수이다.
* 읽은 데이터는 버퍼에 저장되며, 읽은 바이트 수가 bytes와 같아질 때까지 반복해서 호출된다.
* 읽기 도중 에러가 발생하면 error() 함수를 호출하여 프로그램을 종료시킨다.



## write\_all

```c
void write_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) error("write failed");
        n += next;
    }
}
```

함수 이름: write\_all

입력:

* int fd: 파일 디스크립터 (파일이나 소켓 등을 열 때 반환되는 정수 값)
* char \*buffer: 데이터가 저장된 버퍼의 포인터
* size\_t bytes: 버퍼에 저장된 데이터의 크기

동작:

* 주어진 파일 디스크립터(fd)로부터 데이터를 읽어 buffer에 저장한다.
* bytes 만큼의 데이터를 모두 읽을 때까지 반복해서 읽어온다.
* 모두 읽어온 후 buffer에 저장된 데이터를 fd로 전송한다.

설명:

* 파일이나 소켓 등으로부터 큰 용량의 데이터를 읽거나 쓸 때, 일부분만 한 번에 읽고 쓰는 것보다 전체를 한 번에 처리하는 것이 효율적이다.
* 이러한 목적으로 만들어진 함수이며, 파일이나 소켓 등으로부터 지정된 크기의 데이터를 모두 읽거나 쓸 때까지 반복해서 처리한다.



## copy\_string

```c
char *copy_string(char *s)
{
    char *copy = malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}
```

함수 이름: copy\_string

입력:&#x20;

* s: 문자열을 나타내는 포인터 변수&#x20;

동작:

1. 문자열 s의 길이를 계산하여 그 길이+1만큼 메모리를 할당한다.
2. 할당된 메모리에 s의 내용을 복사한다.
3. 복사된 문자열을 나타내는 포인터 변수 copy를 반환한다.

설명:&#x20;

* 이 함수는 입력으로 받은 문자열 s를 복사하여 새로운 메모리에 저장하고, 복사된 문자열을 나타내는 포인터를 반환하는 함수이다.&#x20;
* 이를 위해 우선 문자열 s의 길이를 계산하여 그 길이+1만큼의 메모리를 할당한다.&#x20;
* 이후 할당된 메모리에 문자열 s의 내용을 복사하여 저장하고, 복사된 문자열을 나타내는 포인터 변수를 반환한다.&#x20;
* 이 함수는 문자열을 복사하기 때문에 입력으로 받은 문자열 s를 변경하지 않고, 새로운 메모리에 저장된 문자열을 반환한다.&#x20;
* 이 함수를 사용하면 문자열을 복사하여 다른 변수에 저장해야 할 때 유용하다.



## parse\_csv\_line

```c
list *parse_csv_line(char *line)
{
    list *l = make_list();
    char *c, *p;
    int in = 0;
    for(c = line, p = line; *c != '\0'; ++c){
        if(*c == '"') in = !in;
        else if(*c == ',' && !in){
            *c = '\0';
            list_insert(l, copy_string(p));
            p = c+1;
        }
    }
    list_insert(l, copy_string(p));
    return l;
}
```

함수 이름: parse\_csv\_line

입력:&#x20;

* line: CSV(comma-separated values) 형식으로 구성된 문자열을 나타내는 포인터 변수

동작:

1. make\_list() 함수를 호출하여 새로운 리스트 l을 생성한다.
2. 문자열 line을 순회하며 각 필드를 구분하여 리스트 l에 추가한다.
3. 리스트 l을 반환한다.

설명:&#x20;

* 이 함수는 CSV 형식의 문자열 line을 파싱하여 각 필드를 나타내는 문자열을 리스트 l에 추가하는 함수이다.&#x20;
* 이를 위해 문자열 line을 순회하면서 새로운 필드가 시작될 때마다 이전 필드를 리스트 l에 추가한다.&#x20;
* 각 필드를 구분하는 구분자로는 쉼표(,)를 사용하며, 각 필드는 큰따옴표(")로 감싸져 있을 수 있다.&#x20;
* 이 함수는 각 필드를 나타내는 문자열을 메모리에 새로 할당하여 리스트 l에 추가한다.&#x20;
* 따라서, 이 함수를 사용한 후에는 리스트 l에 저장된 문자열을 참조하는 모든 포인터 변수를 해제해주어야 한다.



## count\_fields

```c
int count_fields(char *line)
{
    int count = 0;
    int done = 0;
    char *c;
    for(c = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done) ++count;
    }
    return count;
}
```

함수 이름: count\_fields

입력:

* line: char 형식의 포인터

동작:

* count\_fields 함수는 주어진 문자열 line에서 구분자로 구분된 필드의 개수를 반환합니다.
* 함수는 문자열 line에서 구분자로 ','를 사용합니다.

설명:

* 이 함수는 문자열 line에서 구분자 ','를 사용하여 필드를 구분합니다.
* 필드의 개수는 구분자로 구분된 개수와 같으므로, 구분자의 개수를 계산하여 필드의 개수를 반환합니다.
* 함수는 문자열의 끝에 도달할 때까지 문자열을 탐색하며, 구분자가 나타나면 필드의 개수를 증가시킵니다.
* 문자열 line은 null 종료되어 있으므로, 문자열의 끝을 나타내는 null 문자를 만날 때까지 문자열을 탐색합니다.
* 마지막 필드가 구분자로 끝나지 않는 경우, 마지막 필드의 끝을 null 문자로 설정하여 필드의 개수를 증가시킵니다.



## parse\_fields

```c
float *parse_fields(char *line, int n)
{
    float *field = calloc(n, sizeof(float));
    char *c, *p, *end;
    int count = 0;
    int done = 0;
    for(c = line, p = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done){
            *c = '\0';
            field[count] = strtod(p, &end);
            if(p == c) field[count] = nan("");
            if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
            p = c+1;
            ++count;
        }
    }
    return field;
}
```

함수 이름: parse\_fields

입력:

* line: char 형식의 포인터
* n: 정수형 변수

동작:

* parse\_fields 함수는 문자열 line을 파싱하여 n개의 float 값을 갖는 배열을 반환합니다.
* 함수는 문자열 line의 구분자로 ','를 사용합니다.
* 반환되는 배열의 요소는 문자열 line에서 구분자로 구분되는 각 필드의 float 값입니다.

설명:

* 이 함수는 주어진 문자열 line을 구분자 ','를 사용하여 파싱합니다.
* 파싱된 각 필드는 float 형식의 값으로 변환되어 반환됩니다.
* 반환된 배열은 메모리 동적 할당 함수인 calloc를 사용하여 할당됩니다.
* 문자열 line의 길이에 따라 필드의 개수는 다를 수 있지만, 입력된 n의 값에 따라 반환되는 배열의 크기는 항상 n입니다.
* 문자열 line의 파싱 과정에서 에러가 발생한 경우, 해당 필드의 값을 NaN(not a number)으로 설정합니다.
* 예외 처리로, DOS 파일 형식의 경우 각 필드의 끝에 '\r' 문자가 포함될 수 있으므로, 이 경우에는 파싱이 실패하고 필드의 값이 NaN으로 설정됩니다.



## sum\_array

```c
float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}
```

함수 이름: sum\_array

입력:

* a: float 포인터 형식의 배열&#x20;
* n: 정수&#x20;

동작:

* 주어진 배열 a의 첫 번째 요소의 주소를 가리키는 포인터를 받아, 배열 a의 모든 요소를 합한 값을 반환하는 함수입니다.
* 함수 내부에서는 for 루프를 사용하여 배열의 모든 요소를 더합니다.

설명:

* 이 함수는 주어진 배열의 모든 요소를 더하여 그 합을 반환합니다.
* 함수 이름에서 알 수 있듯이, 이 함수는 배열의 합을 구하는 것이 목적이므로, 반환 값의 자료형은 float입니다.
* 배열 a의 크기는 n이며, for 루프를 사용하여 배열의 모든 요소를 더합니다.
* 배열의 각 요소는 float 형식이어야 합니다.
* 이 함수는 입력으로 주어진 배열의 모든 요소의 합을 계산하기 때문에, 배열의 요소들이 어떤 의미를 가지고 있는지는 고려하지 않습니다.



## mean\_array

```c
float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}
```

함수 이름: mean\_array&#x20;

입력:

* float \*a: 평균을 구하고자 하는 배열
* int n: 배열의 크기&#x20;

동작:&#x20;

* 배열의 원소들의 합을 배열의 크기로 나누어 평균값을 계산한다.&#x20;

설명:&#x20;

* 주어진 배열의 평균값을 계산하여 반환하는 함수이다.&#x20;
* 평균값은 배열의 원소들의 합을 배열의 크기로 나누어 계산된다.



## mean\_arrays

```c
void mean_arrays(float **a, int n, int els, float *avg)
{
    int i;
    int j;
    memset(avg, 0, els*sizeof(float));
    for(j = 0; j < n; ++j){
        for(i = 0; i < els; ++i){
            avg[i] += a[j][i];
        }
    }
    for(i = 0; i < els; ++i){
        avg[i] /= n;
    }
}
```

함수 이름: mean\_arrays

입력:

* float \*\*a: 2차원 배열 a
* int n: 배열 a의 행 수
* int els: 배열 a의 열 수
* float \*avg: 평균 값을 저장할 1차원 배열

동작:&#x20;

* 2차원 배열 a의 각 열마다 평균 값을 계산하여, 1차원 배열 avg에 저장하는 함수

설명:&#x20;

* mean\_arrays 함수는 2차원 배열 a의 각 열마다 평균 값을 계산하여 1차원 배열 avg에 저장합니다.&#x20;
* 먼저, 배열 avg를 0으로 초기화하고, 2중 for문을 사용하여 배열 a의 각 열에 대해 합을 계산합니다.&#x20;
* 그리고 배열 a의 행 수 n으로 나누어 각 열의 평균 값을 계산합니다. 계산된 평균 값은 배열 avg에 저장됩니다.



## print\_statistics

```c
void print_statistics(float *a, int n)
{
    float m = mean_array(a, n);
    float v = variance_array(a, n);
    printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}
```

함수 이름: print\_statistics

입력:

* float형 포인터 a : 분석할 배열
* int형 n : 배열의 크기

동작:

* 입력으로 받은 배열 a를 이용하여 MSE, 평균, 분산을 계산하고, 이를 출력한다.

설명:

* print\_statistics 함수는 입력으로 받은 배열 a의 MSE, 평균, 분산을 계산하고 이를 출력하는 함수이다.
* 우선, 입력으로 받은 배열 a를 이용하여 MSE, 평균, 분산을 계산한다. 이를 위해서는 mse\_array, mean\_array, variance\_array 함수를 이용한다.
* 계산된 MSE, 평균, 분산은 printf 함수를 이용하여 출력된다. 출력 형식은 "MSE: %.6f, Mean: %.6f, Variance: %.6f"이다.



## variance\_array

```c
float variance_array(float *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}
```

함수 이름: variance\_array

입력:

* float \*a: 분산을 계산하려는 배열
* int n: 배열 a의 길이

동작:

* 배열 a의 분산을 계산합니다.
* 분산은 각 항목의 평균과의 차이를 제곱한 값의 평균입니다.

설명:

* variance\_array 함수는 주어진 배열 a의 분산을 계산하는 함수입니다.
* 배열 a의 길이 n을 받아와서, 각 항목의 평균값을 계산합니다.
* 그리고 배열 a의 각 항목에서 평균값을 빼고, 그 결과를 제곱한 값을 모두 더합니다.
* 이렇게 구한 값에 배열 a의 길이 n으로 나누면 분산을 구할 수 있습니다.
* 구해진 분산값을 반환합니다.



## constrain\_int

```c
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}
```

함수 이름: constrain\_int

입력:

* a: 제한하려는 정수 값
* min: 허용되는 최소값
* max: 허용되는 최대값

동작:

* a 값을 min 과 max 사이의 값으로 제한하여 반환합니다.

설명:

* 주어진 정수 a가 min 보다 작으면 min 값으로, max 보다 크면 max 값으로 대체합니다.
* 그 외에는 a 값을 그대로 반환합니다.
* 반환 값은 항상 min 과 max 사이의 값이 됩니다.



## constrain

```c
float constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}
```

함수 이름: constrain

입력:

* float min: a 값의 최소값
* float max: a 값의 최대값
* float a: 값이 제한될 대상

동작:

* a 값을 min과 max 범위 안으로 조정한다.
* 만약 a가 min보다 작으면 min으로 조정하고, a가 max보다 크면 max로 조정한다.

설명:

* 이 함수는 값의 범위를 제한하는 기능을 수행한다. 값을 제한하는 것은 입력 데이터의 범위를 제한하거나 출력 값을 제한하는 등의 다양한 용도로 사용할 수 있다.



## dist\_array

```c
float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; i += sub) sum += pow(a[i]-b[i], 2);
    return sqrt(sum);
}
```

함수 이름: dist\_array

입력:

* float형 포인터 a: 첫 번째 배열을 가리키는 포인터
* float형 포인터 b: 두 번째 배열을 가리키는 포인터
* int형 n: 배열의 길이
* int형 sub: 배열 요소의 일부분만 사용할 경우, 사용할 요소의 간격

동작:&#x20;

* 두 개의 배열 a, b를 비교하여 유클리드 거리(Euclidean distance)를 계산합니다.&#x20;
* 이때 sub 매개변수를 사용하여 a와 b의 일부분만 사용할 수 있습니다.

설명:&#x20;

* 유클리드 거리는 공간에서 두 점 사이의 거리를 계산하는 방법 중 하나입니다.&#x20;
* 이 함수는 두 배열 a, b 간의 유클리드 거리를 계산하여 반환합니다.&#x20;
* 만약 sub 매개변수가 1이면 모든 요소를 사용하여 거리를 계산하고, sub가 2이면 첫 번째, 세 번째, 다섯 번째 요소와 같은 일부분만 사용하여 거리를 계산합니다.



## mse\_array

```c
float mse_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i]*a[i];
    return sqrt(sum/n);
}
```

함수 이름: mse\_array

입력:

* float 형식의 1차원 배열 a
* a 배열의 크기 n

동작:

* 입력받은 배열 a의 모든 요소에 대해 a\[i]\*a\[i]의 합을 구함
* 합을 n으로 나눈 뒤, 그 결과에 대해 제곱근(sqrt)을 취한 값을 반환함

설명:

* mse\_array 함수는 입력받은 배열 a의 값들이 0에 가까운지, 큰 값인지, 작은 값인지 등을 알아내기 위해 사용될 수 있음
* 입력받은 배열 a의 평균(mean) 값이 0에 가까울수록, mse\_array 함수의 반환값은 작아짐
* 입력받은 배열 a의 값들이 모두 같다면, mse\_array 함수의 반환값은 0이 됨



## normalize\_array

```c
void normalize_array(float *a, int n)
{
    int i;
    float mu = mean_array(a,n);
    float sigma = sqrt(variance_array(a,n));
    for(i = 0; i < n; ++i){
        a[i] = (a[i] - mu)/sigma;
    }
    mu = mean_array(a,n);
    sigma = sqrt(variance_array(a,n));
}
```

함수 이름: normalize\_array

입력:

* a: float 형태의 1차원 배열 포인터
* n: 배열 a의 크기

동작:

* 입력으로 주어진 1차원 배열 a의 값을 정규화한다.
* 평균값과 분산값을 계산하여 각각 mu, sigma 변수에 저장한다.
* 배열 a의 모든 원소에 대하여 (각 원소 값 - mu) / sigma 값을 다시 할당한다.

설명:

* 정규화는 데이터 전처리(preprocessing)의 일환으로, 데이터의 범위(scale)를 일치시키거나 분포(distribution)를 표준화하여 학습 성능을 향상시키기 위해 사용된다.
* 이 함수는 주어진 배열 a를 정규화하기 위해 먼저 평균값과 분산값을 계산하고, 이를 이용하여 모든 원소 값을 재할당한다.
* 최종적으로 mu, sigma 값을 다시 계산하는 이유는, 정규화 작업 후에도 이 값들이 일치해야하며, 이를 검증하기 위함이다.



## translate\_array

```c
void translate_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] += s;
    }
}
```

함수 이름: translate\_array

입력:

* float \*a: 실수형 배열 포인터
* int n: 배열 a의 크기
* float s: 더할 실수값

동작:

* 주어진 배열 a의 모든 요소에 s를 더한다.

설명:

* translate는 "이동하다"라는 뜻을 가지고 있으며, 이 함수는 배열을 이동시키는 것과 같은 효과를 가진다.&#x20;
* 즉, 주어진 배열의 모든 요소를 s만큼 이동시킨다. 예를 들어, 배열 \[1, 2, 3]에 2를 더하면 \[3, 4, 5]가 된다.

## mag\_array

```c
float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];   
    }
    return sqrt(sum);
}
```

함수 이름: mag\_array 입력:

* float \*a: 부동 소수점 배열
* int n: 배열 a의 요소 수

동작:

* 배열 a의 모든 요소를 제곱한 값들의 합을 계산합니다.
* 합의 제곱근을 계산하여 반환합니다.

설명:

* 이 함수는 벡터의 크기를 계산하기 위해 사용됩니다.
* 입력으로 주어진 배열 a는 각 차원의 값을 나타냅니다.
* 배열 a의 각 요소를 제곱한 값들의 합은 벡터의 크기의 제곱과 같습니다.
* 따라서 합의 제곱근을 계산하여 벡터의 크기를 구할 수 있습니다.
* 이 함수는 벡터의 크기 계산 외에도 다양한 응용 분야에서 사용될 수 있습니다.



## scale\_array

```c
void scale_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] *= s;
    }
}
```

함수 이름: scale\_array

입력:

* float \*a: 조정할 배열의 포인터
* int n: 배열의 크기
* float s: 스케일링할 값

동작:

* 입력으로 받은 배열 a의 모든 요소에 스케일링할 값 s를 곱하여 값을 조정한다.

설명:

* 입력으로 받은 배열 a의 모든 요소에 스케일링할 값 s를 곱하여 값을 조정하는 함수이다.
* 배열의 크기 n은 포인터 a가 가리키는 배열의 크기를 나타낸다.



## sample\_array

```c
int sample_array(float *a, int n)
{
    float sum = sum_array(a, n);
    scale_array(a, n, 1./sum);
    float r = rand_uniform(0, 1);
    int i;
    for(i = 0; i < n; ++i){
        r = r - a[i];
        if (r <= 0) return i;
    }
    return n-1;
}
```

함수 이름: sample\_array

입력:

* a: float형 배열 포인터, 확률 분포 값들이 저장된 배열
* n: int형, 배열 a의 길이

동작:

* 배열 a의 원소들의 합(sum)을 계산한다.
* 배열 a의 모든 원소들을 합(sum)으로 나누어서 배열 a를 확률 분포로 만든다.
* 0부터 1사이의 균등 분포(uniform distribution)에서 난수 r을 생성한다.
* 배열 a를 순회하면서 r에서 각 원소 값을 차감해나가다가 r이 0 이하가 되는 첫번째 인덱스 i를 찾는다.
* 인덱스 i를 반환한다.

설명:&#x20;

* 주어진 확률 분포를 따르는 난수를 샘플링하는 함수이다.&#x20;
* 먼저, 배열 a의 원소들의 합을 계산하고, 모든 원소들을 합으로 나누어서 배열 a를 확률 분포로 만든다.&#x20;
* 그 후, 0부터 1사이의 균등 분포에서 난수 r을 생성하고, 배열 a를 순회하면서 r에서 각 원소 값을 차감해나가다가 r이 0 이하가 되는 첫번째 인덱스 i를 찾는다.&#x20;
* 이렇게 찾은 인덱스 i가 주어진 확률 분포를 따르는 난수이다.



## max\_int\_index

```c
int max_int_index(int *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    int max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}
```

함수 이름: max\_int\_index

입력:

* int형 포인터 a: 정수 배열을 가리키는 포인터
* int형 변수 n: 배열 a의 크기

동작:

* 입력으로 받은 정수 배열 a에서 가장 큰 값을 가진 원소의 인덱스를 찾아서 반환한다.

설명:

* 입력으로 받은 배열 a에서 가장 큰 값을 가진 원소의 인덱스를 찾아 반환한다.
* 배열 a의 크기가 0 이하인 경우 -1을 반환한다.



## max\_index

```c
int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}
```

함수 이름: max\_index

입력:&#x20;

* float 형식의 배열 a와 해당 배열의 길이 n

동작:&#x20;

* 입력으로 받은 배열 a에서 가장 큰 값이 위치한 인덱스를 반환합니다.&#x20;
* 배열이 비어있는 경우 -1을 반환합니다.

설명:&#x20;

* 입력으로 받은 배열 a에서 가장 큰 값이 위치한 인덱스를 찾는 함수입니다.&#x20;
* 함수는 배열 a의 첫 번째 원소를 최대값으로 설정하고 배열의 모든 원소를 탐색하며, 최대값보다 큰 값을 찾으면 해당 원소의 인덱스를 최대값 위치로 변경합니다.&#x20;
* 배열의 모든 원소를 탐색한 후, 최대값이 위치한 인덱스를 반환합니다.&#x20;
* 배열이 비어있는 경우 -1을 반환합니다.



## int\_index

```c
int int_index(int *a, int val, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        if(a[i] == val) return i;
    }
    return -1;
}
```

함수 이름: int\_index\
입력:

* int \*a: 정수 배열 포인터
* int val: 찾으려는 정수 값
* int n: 배열의 길이

동작:&#x20;

* 정수 배열 a에서 val 값을 가지는 요소의 인덱스를 찾는다.
* 배열 a의 처음부터 끝까지 반복하면서 val 값과 일치하는 첫 번째 요소의 인덱스를 반환한다.
* 일치하는 요소가 없을 경우 -1을 반환한다.

설명:&#x20;

* 배열 a에서 특정 값을 찾을 때 사용하는 함수이다.&#x20;
* 반환값으로 해당 값이 존재하는지 여부를 알 수 있고, 요소의 인덱스를 알 수 있으므로 배열의 특정 위치에 접근하여 값을 변경하는 등의 작업에 사용될 수 있다.



## rand\_int

```c
int rand_int(int min, int max)
{
    if (max < min){
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand()%(max - min + 1)) + min;
    return r;
}
```

함수 이름: rand\_int

입력:&#x20;

* 정수형 변수 min, max

동작:&#x20;

* min과 max 사이의 난수를 생성하여 반환

설명:&#x20;

* srand() 함수를 호출하여 시드를 설정하고 rand() 함수를 이용하여 min과 max 사이의 난수를 생성한 후 반환하는 함수입니다.&#x20;
* 만약 max가 min보다 작은 경우, 두 변수의 값을 서로 바꿔줍니다.&#x20;
* 반환된 값은 min 이상 max 이하의 정수입니다.



## rand\_normal

```c
// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    if(haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}
```

함수 이름: rand\_normal&#x20;

입력:&#x20;

* 없음&#x20;

동작:&#x20;

* Box-Muller 변환을 사용하여 평균 0, 표준 편차 1인 정규 분포에서 무작위로 값을 생성하여 반환합니다.&#x20;

설명:&#x20;

* 이 함수는 Box-Muller 변환을 사용하여 표준 정규 분포에서 값을 생성합니다.&#x20;
* 변환에는 2개의 난수가 필요하며, 함수는 이전 호출에서 사용하지 않은 경우 두 번째 값을 저장합니다.&#x20;
* 함수는 이전 호출에서 미사용된 두 번째 값을 가지고 있으면 이를 반환하고, 그렇지 않으면 새로운 2개의 난수를 생성하여 변환을 수행합니다.&#x20;
* 이 함수는 수학, 통계, 물리학, 컴퓨터 그래픽 등에서 널리 사용되는 중요한 분포 중 하나인 정규 분포에서 무작위로 값을 생성하는 데 사용됩니다.



## rand\_size\_t

```c
size_t rand_size_t()
{
    return  ((size_t)(rand()&0xff) << 56) |
        ((size_t)(rand()&0xff) << 48) |
        ((size_t)(rand()&0xff) << 40) |
        ((size_t)(rand()&0xff) << 32) |
        ((size_t)(rand()&0xff) << 24) |
        ((size_t)(rand()&0xff) << 16) |
        ((size_t)(rand()&0xff) << 8) |
        ((size_t)(rand()&0xff) << 0);
}
```

함수 이름: rand\_size\_t

입력:&#x20;

* 없음

동작:&#x20;

* rand() 함수를 사용하여 8바이트 크기의 랜덤한 값을 생성하고, 이를 size\_t 형식으로 반환한다.

설명:&#x20;

* rand() 함수는 0부터 RAND\_MAX까지의 값 중에서 랜덤하게 값을 반환한다.&#x20;
* 이 함수는 8개의 rand() 호출을 사용하여 8바이트 크기의 랜덤한 값을 생성하고, 이를 size\_t 형식으로 변환하여 반환한다.&#x20;
* 이 함수를 사용하면, size\_t 형식의 임의의 값이 필요한 경우 이 함수를 호출하여 사용할 수 있다.



## rand\_uniform

```c
float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}
```

함수 이름: rand\_uniform

입력:

* float min: 반환될 수 있는 값 중에서 가장 작은 값
* float max: 반환될 수 있는 값 중에서 가장 큰 값

동작:&#x20;

* 입력으로 받은 min과 max 사이에서 균일하게 분포하는 난수를 생성한다.

설명:

* rand() 함수를 사용하여 0부터 RAND\_MAX 사이의 임의의 정수를 반환한다.
* 이 값을 (max - min) 범위에서 균일하게 분포하는 값으로 변환하고 min 값을 더하여 min과 max 사이에서 균일하게 분포하는 값을 반환한다.
* max가 min보다 작은 경우, min과 max 값을 서로 바꾸어 계산한다.



## rand\_scale

```c
float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if(rand()%2) return scale;
    return 1./scale;
}
```

함수 이름: rand\_scale

입력:&#x20;

* float s: 스케일 값

동작:&#x20;

* 입력된 스케일 값에 무작위로 생성된 가중치 값을 곱하거나 나누어 반환한다.&#x20;
* rand\_uniform 함수를 사용하여 1과 스케일 값 사이의 무작위 실수값을 생성한 후, 50% 확률로 그 값을 반환하거나 1에서 그 값을 나눈 값을 반환한다.

설명:&#x20;

* 이 함수는 머신러닝 모델의 가중치를 초기화하는 데 사용될 수 있다.&#x20;
* 스케일 값은 보통 0에서 1 사이의 값을 갖는데, 이 함수는 이 값을 무작위로 조정하여 초기 가중치를 다양하게 설정할 수 있게 해준다.&#x20;
* 이렇게 함으로써 모델이 다양한 초기 가중치를 가지게 되어, 더욱 다양한 데이터에 대해 더욱 좋은 결과를 얻을 수 있다.

## one\_hot\_encode

```c
float **one_hot_encode(float *a, int n, int k)
{
    int i;
    float **t = calloc(n, sizeof(float*));
    for(i = 0; i < n; ++i){
        t[i] = calloc(k, sizeof(float));
        int index = (int)a[i];
        t[i][index] = 1;
    }
    return t;
}
```

함수 이름: one\_hot\_encode

입력:

* float \*a: 크기가 n인 1차원 실수 배열
* int n: 배열 a의 원소 개수
* int k: 생성하려는 one-hot 인코딩 벡터의 차원 수

동작:

* 크기가 n x k인 2차원 실수 배열 t를 할당하고, 모든 원소를 0으로 초기화
* 배열 a의 각 원소를 정수형으로 변환하여 인덱스(index)를 구하고, t\[i]\[index]를 1로 설정하여 해당 인덱스에 대한 원-핫 인코딩을 만든다.

설명:

* one-hot 인코딩은 범주형 데이터를 다룰 때 많이 사용하는 기법 중 하나로, 각각의 범주에 대한 이진 플래그(0 또는 1)를 사용하여 벡터화한다.
* 이 함수는 크기가 n인 1차원 실수 배열 a에 대해 one-hot 인코딩을 수행한다.
* 입력 배열 a의 각 원소를 정수형으로 변환한 후, 해당 인덱스에 대한 원-핫 인코딩 벡터를 생성하여 크기가 n x k인 2차원 실수 배열 t에 저장한다.
* 이러한 인코딩은 머신러닝 분류 문제에서 자주 사용되며, 예를 들어 k개의 클래스가 있는 분류 문제에서 각각의 샘플은 k차원의 이진 벡터로 표현된다.



## sum\_array

```c
float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}
```

함수 이름: sum\_array

입력:

* float형 배열 a
* int형 변수 n

동작:

* 배열 a의 n개의 원소들을 모두 더한 값을 반환한다.

설명:

* 주어진 float형 배열 a의 n개의 원소들을 모두 더한 값을 계산하여 반환하는 함수이다.
* for 반복문을 이용하여 배열 a의 원소들을 하나씩 더해가며 sum 변수에 더해주고, 마지막에 sum 값을 반환한다.



## mean\_array

```c
float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}
```

함수 이름: mean\_array

입력:

* a: float형 배열
* n: 배열 a의 원소 개수

동작:&#x20;

* 배열 a의 원소들의 평균 값을 계산하여 반환한다.

설명:&#x20;

* 주어진 배열 a의 모든 원소들의 합을 배열 a의 원소 개수 n으로 나눈 값이 배열 a의 평균값이다.&#x20;
* sum\_array 함수를 이용하여 배열 a의 모든 원소들의 합을 먼저 계산하고, n으로 나눈 값을 반환하는 것으로 평균 값을 계산한다.
