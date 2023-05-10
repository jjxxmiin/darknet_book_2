# go

```c
#include "darknet.h"

#include <assert.h>
#include <math.h>
#include <unistd.h>

int inverted = 1;
int noi = 1;
static const int nind = 10;
int legal_go(float *b, float *ko, int p, int r, int c);
int check_ko(float *x, float *ko);

typedef struct {
    char **data;
    int n;
} moves;
```

## fgetgo

```c
char *fgetgo(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 96;
    char *line = malloc(size*sizeof(char));
    if(size != fread(line, sizeof(char), size, fp)){
        free(line);
        return 0;
    }

    return line;
}
```

함수 이름: fgetgo

입력:

* fp: FILE 포인터. 파일 포인터입니다.

동작:&#x20;

* 이 함수는 파일에서 한 줄을 읽어와서 문자열로 반환합니다.

설명:&#x20;

* 이 함수는 주어진 파일 포인터(fp)에서 한 줄을 읽어와서 문자열로 반환합니다.
* 먼저, 파일의 끝인지 검사하여 파일의 끝에 도달했을 경우 0을 반환합니다.
* 다음으로, 읽은 문자열을 저장할 버퍼인 line을 할당합니다. 버퍼의 초기 크기는 96입니다.
* fread 함수를 사용하여 파일에서 문자를 읽어와서 버퍼(line)에 저장합니다. 읽은 문자의 개수가 할당한 버퍼의 크기(size)와 다를 경우에는 할당한 메모리를 해제하고 0을 반환합니다.
* 마지막으로, 읽은 문자열(line)을 반환합니다.



## load\_go\_moves

```c
moves load_go_moves(char *filename)
{
    moves m;
    m.n = 128;
    m.data = calloc(128, sizeof(char*));
    FILE *fp = fopen(filename, "rb");
    int count = 0;
    char *line = 0;
    while ((line = fgetgo(fp))) {
        if (count >= m.n) {
            m.n *= 2;
            m.data = realloc(m.data, m.n*sizeof(char*));
        }
        m.data[count] = line;
        ++count;
    }
    printf("%d\n", count);
    m.n = count;
    m.data = realloc(m.data, count*sizeof(char*));
    return m;
}
```

함수 이름: load\_go\_moves

입력:

* filename: 문자열. 파일 이름을 나타냅니다.

동작:&#x20;

* 이 함수는 주어진 파일에서 Go 게임의 움직임(moves)을 로드합니다.

설명:&#x20;

* 이 함수는 주어진 파일(filename)에서 Go 게임의 움직임(moves)을 로드합니다.
* 먼저, moves 구조체인 m을 선언하고 초기값을 설정합니다. m.n은 초기에 128로 설정되며, m.data는 128개의 char 포인터를 가리키는 메모리를 할당받습니다.
* 그 다음, 주어진 파일(filename)을 바이너리 모드로 열고 해당 파일의 파일 포인터(fp)를 얻습니다.
* 카운트 변수(count)를 초기화하고, 문자열 포인터(line)을 0으로 초기화합니다.
* 반복문을 사용하여 파일에서 한 줄씩 읽어옵니다.&#x20;
* fgetgo 함수를 사용하여 한 줄을 읽어옵니다.&#x20;
* 읽어온 줄(line)을 가리키는 포인터가 NULL이 아닐 경우에만 반복문이 실행됩니다.
* count가 m.n보다 크거나 같을 경우에는 m.n을 두 배로 증가시키고, m.data의 크기를 재할당하여 메모리를 확장합니다.
* m.data의 count 인덱스에 읽은 줄(line)을 할당합니다. 그리고 count를 증가시킵니다.
* 반복문이 끝나면 count 값을 출력합니다.
* 마지막으로, m.n을 count로 설정하고, m.data의 크기를 count_sizeof(char_)로 재할당합니다. 그리고 m을 반환합니다.



## string\_to\_board

```c
void string_to_board(char *s, float *board)
{
    int i, j;
    memset(board, 0, 2*19*19*sizeof(float));
    int count = 0;
    for(i = 0; i < 91; ++i){
        char c = s[i];
        for(j = 0; j < 4; ++j){
            int me = (c >> (2*j)) & 1;
            int you = (c >> (2*j + 1)) & 1;
            if (me) board[count] = 1;
            else if (you) board[count + 19*19] = 1;
            ++count;
            if(count >= 19*19) break;
        }
    }
}
```

함수 이름: string\_to\_board

입력:

* s: 문자열. Go 게임의 보드 상태를 나타내는 문자열입니다.
* board: 실수(float) 배열. Go 게임 보드를 나타내는 배열입니다.

동작:&#x20;

* 이 함수는 주어진 문자열로부터 Go 게임의 보드 상태를 실수 배열로 변환합니다.

설명:&#x20;

* 이 함수는 주어진 문자열 s를 사용하여 Go 게임의 보드 상태를 실수 배열인 board로 변환합니다.&#x20;
* 먼저, board 배열을 0으로 초기화합니다. memset 함수를 사용하여 board 배열의 모든 요소를 0으로 설정합니다. board 배열은 2차원 Go 게임 보드를 나타내며, 크기는 19x19입니다.
* count 변수를 0으로 초기화합니다.
* 이중 반복문을 사용하여 문자열 s를 순회합니다. 바깥쪽 반복문은 i를 0부터 90까지 증가시키며, 안쪽 반복문은 j를 0부터 3까지 증가시킵니다. 이는 문자열 s의 각 문자를 4비트 단위로 처리하기 위한 반복문입니다.
* 각 문자 c에서 j번째 비트와 j+1번째 비트를 추출하여 me와 you 변수에 저장합니다. 이를 위해 비트 연산자를 사용합니다.
* me가 1인 경우, board\[count]에 1을 할당합니다. 즉, 현재 위치에 흑돌(자신의 돌)이 있다는 의미입니다. you가 1인 경우, board\[count + 19\*19]에 1을 할당합니다. 즉, 현재 위치에 백돌(상대방의 돌)이 있다는 의미입니다.
* count를 증가시킵니다. 이는 보드 상태를 배열로 변환할 때의 인덱스를 나타냅니다.
* count가 19x19보다 크거나 같으면 반복문을 종료합니다. 이는 보드를 모두 처리했음을 의미합니다.



## board\_to\_string

```c
void board_to_string(char *s, float *board)
{
    int i, j;
    memset(s, 0, (19*19/4+1)*sizeof(char));
    int count = 0;
    for(i = 0; i < 91; ++i){
        for(j = 0; j < 4; ++j){
            int me = (board[count] == 1);
            int you = (board[count + 19*19] == 1);
            if (me) s[i] = s[i] | (1<<(2*j));
            if (you) s[i] = s[i] | (1<<(2*j + 1));
            ++count;
            if(count >= 19*19) break;
        }
    }
}
```

함수 이름: board\_to\_string

입력:

* s: 문자열. 변환된 Go 게임 보드 상태가 저장될 문자열입니다.
* board: 실수(float) 배열. Go 게임 보드를 나타내는 배열입니다.

동작:&#x20;

* 이 함수는 주어진 Go 게임 보드를 문자열로 변환합니다.

설명:&#x20;

* 이 함수는 주어진 실수 배열인 board를 사용하여 Go 게임의 보드 상태를 문자열 s로 변환합니다.
* 먼저, 문자열 s를 0으로 초기화합니다. memset 함수를 사용하여 s의 모든 요소를 0으로 설정합니다.&#x20;
* s는 변환된 보드 상태를 저장하기 위한 문자열입니다. 문자열의 크기는 (19\*19/4+1)입니다.&#x20;
* 여기서 4는 문자열에서 한 문자당 필요한 비트 수를 나타내며, +1은 문자열의 종료를 나타내는 null 문자를 위한 공간입니다.
* count 변수를 0으로 초기화합니다.
* 이중 반복문을 사용하여 board 배열을 순회합니다.&#x20;
* 바깥쪽 반복문은 i를 0부터 90까지 증가시키며, 안쪽 반복문은 j를 0부터 3까지 증가시킵니다.&#x20;
* 이는 문자열 s의 각 문자에 4비트씩 할당하기 위한 반복문입니다.
* 각 위치의 me와 you 값을 결정합니다.&#x20;
* board\[count] 값이 1인 경우 me는 true(1)로 설정하고, board\[count + 19\*19] 값이 1인 경우 you는 true(1)로 설정합니다.
* me가 true인 경우, 문자열 s의 i번째 위치에 (1<<(2\*j)) 값을 논리 OR 연산을 통해 할당합니다.&#x20;
* 즉, 해당 비트 위치에 흑돌(자신의 돌)의 정보를 설정합니다.&#x20;
* you가 true인 경우, 문자열 s의 i번째 위치에 (1<<(2\*j + 1)) 값을 논리 OR 연산을 통해 할당합니다.&#x20;
* 즉, 해당 비트 위치에 백돌(상대방의 돌)의 정보를 설정합니다.
* count를 증가시킵니다. 이는 보드 배열을 순회할 때의 인덱스를 나타냅니다.
* count가 19x19보다 크거나 같으면 반복문을 종료합니다. 이는 보드를 모두 처리했음을 의미합니다.



## occupied

```c
static int occupied(float *b, int i)
{
    if (b[i]) return 1;
    if (b[i+19*19]) return -1;
    return 0;
}
```

함수 이름: occupied

입력:

* b: 실수(float) 배열. Go 게임 보드를 나타내는 배열입니다.
* i: 정수. 배열 b에서 확인할 위치(인덱스)입니다.

동작:&#x20;

* 이 함수는 주어진 위치 i가 보드에서 돌이 놓여진 위치인지를 확인합니다.

설명:&#x20;

* 이 함수는 배열 b에서 주어진 위치 i가 보드에서 돌이 놓여진 위치인지를 확인합니다.&#x20;
* 먼저, b\[i] 값이 0이 아닌 경우 (즉, 흑돌이 놓여진 경우) 1을 반환합니다.&#x20;
* b\[i+19\*19] 값이 0이 아닌 경우 (즉, 백돌이 놓여진 경우) -1을 반환합니다.
* 위 두 조건에 모두 해당하지 않는 경우, 즉 해당 위치에 돌이 없는 경우 0을 반환합니다.



## random\_go\_moves

```c
data random_go_moves(moves m, int n)
{
    data d = {0};
    d.X = make_matrix(n, 19*19*3);
    d.y = make_matrix(n, 19*19+2);
    int i, j;
    for(i = 0; i < n; ++i){
        float *board = d.X.vals[i];
        float *label = d.y.vals[i];
        char *b = m.data[rand()%m.n];
        int player = b[0] - '0';
        int result = b[1] - '0';
        int row = b[2];
        int col = b[3];
        string_to_board(b+4, board);
        if(player > 0) for(j = 0; j < 19*19; ++j) board[19*19*2 + j] = 1;
        label[19*19+1] = (player==result);
        if(row >= 19 || col >= 19){
            label[19*19] = 1;
        } else {
            label[col + 19*row] = 1;
            if(occupied(board, col + 19*row)) printf("hey\n");
        }

        int flip = rand()%2;
        int rotate = rand()%4;
        image in = float_to_image(19, 19, 3, board);
        image out = float_to_image(19, 19, 1, label);
        if(flip){
            flip_image(in);
            flip_image(out);
        }
        rotate_image_cw(in, rotate);
        rotate_image_cw(out, rotate);
    }
    return d;
}
```

함수 이름: random\_go\_moves

입력:

* m: moves 구조체. Go 게임의 움직임 데이터를 포함하는 구조체입니다.
* n: 정수. 생성할 데이터의 개수입니다.

동작:&#x20;

* 이 함수는 무작위로 Go 게임 데이터를 생성합니다. 생성된 데이터는 입력 데이터(X)와 레이블 데이터(y)로 구성된 data 구조체로 반환됩니다.

설명:&#x20;

* 이 함수는 moves 구조체 m에서 무작위로 데이터를 선택하여 Go 게임 데이터를 생성합니다.
* 먼저, data 구조체 d를 초기화합니다. d.X는 크기가 (n, 19_19_3)인 행렬로 초기화되고, d.y는 크기가 (n, 19\*19+2)인 행렬로 초기화됩니다.
* 그런 다음, n번 반복하면서 데이터를 생성합니다. 각 반복에서는 다음 작업을 수행합니다:
  * d.X의 i번째 행에는 Go 게임 보드를 나타내는 배열 board가 할당됩니다.
  * d.y의 i번째 행에는 레이블을 나타내는 배열 label이 할당됩니다.
  * moves 구조체 m에서 무작위로 데이터 b를 선택합니다.
  * b의 첫 번째 문자를 플레이어(player)로, 두 번째 문자를 결과(result)로 추출합니다.
  * b의 세 번째 문자를 행(row)으로, 네 번째 문자를 열(col)로 추출합니다.
  * string\_to\_board 함수를 사용하여 b+4의 데이터를 board 배열로 변환합니다.
  * player가 양수인 경우, 흑돌로 채워진 보드에 대응하는 부분에 1을 설정합니다.
  * label 배열의 19\*19+1번째 요소에는 player와 result의 값이 일치하는지 여부를 저장합니다.
  * row가 19보다 크거나 col이 19보다 큰 경우, label 배열의 19\*19번째 요소를 1로 설정합니다.
  * 그렇지 않은 경우, label 배열의 col + 19\*row번째 요소를 1로 설정합니다. 만약 해당 위치에 돌이 이미 있는 경우 "hey"를 출력합니다.
  * flip과 rotate를 무작위로 선택하여 입력 데이터와 레이블 데이터를 뒤집고 회전시킵니다.
* 위의 과정을 모두 수행한 후, 생성된 data 구조체 d를 반환합니다.



## train\_go

```c
void train_go(char *cfgfile, char *weightfile, char *filename, int *gpus, int ngpus, int clear)
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
    network *net = nets[0];
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    char *backup_directory = "/home/pjreddie/backup/";

    char buff[256];
    moves m = load_go_moves(filename);
    //moves m = load_go_moves("games.txt");

    int N = m.n;
    printf("Moves: %d\n", N);
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time=what_time_is_it_now();

        data train = random_go_moves(m, net->batch*net->subdivisions*ngpus);
        printf("Loaded: %lf seconds\n", what_time_is_it_now() - time);
        time=what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 10);
        }
#else
        loss = train_network(net, train);
#endif
        free_data(train);

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory,base, epoch);
            save_weights(net, buff);

        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%10000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%ld.backup",backup_directory,base,get_current_batch(net));
            save_weights(net, buff);
        }
    }
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free(base);
}
```

함수 이름: train\_go

입력:

* cfgfile: char 포인터. 모델 구성 파일 경로.
* weightfile: char 포인터. 가중치 파일 경로.
* filename: char 포인터. 학습 데이터 파일 경로.
* gpus: int 포인터. GPU 번호 배열.
* ngpus: int. 사용할 GPU 수.
* clear: int. 가중치를 초기화할지 여부(1이면 초기화).

동작:

* Go(바둑) 게임을 학습하는 함수이다.
* ngpus 개수의 GPU를 사용하여 Go 게임 데이터를 이용해 모델을 학습한다.
* 학습된 가중치를 지정된 backup\_directory에 저장한다.

설명:

* train\_go 함수는 Go 게임을 학습하는 함수이다.
* 이 함수는 입력으로 모델 구성 파일 경로(cfgfile), 가중치 파일 경로(weightfile), 학습 데이터 파일 경로(filename), GPU 번호 배열(gpus), 사용할 GPU 수(ngpus), 가중치를 초기화할지 여부(clear)를 받는다.
* 함수 내부에서는 입력으로 받은 cfgfile을 이용하여 모델을 구성하고, 가중치를 불러온다.
* 그리고 ngpus 개수만큼의 GPU를 사용하여 Go 게임 데이터를 이용해 모델을 학습한다.
* 학습된 가중치는 지정된 backup\_directory에 저장된다.
* 학습 도중에는 학습 상태를 출력하며, 현재까지 학습된 이미지 수, 손실(loss), 평균 손실(avg\_loss), 학습률 등을 출력한다.
* 또한, 학습 도중 일정 주기마다 가중치를 저장한다.
* 마지막으로, 학습이 완료된 모델은 메모리에서 해제된다.



## propagate\_liberty

```c
static void propagate_liberty(float *board, int *lib, int *visited, int row, int col, int side)
{
    if (row < 0 || row > 18 || col < 0 || col > 18) return;
    int index = row*19 + col;
    if (occupied(board,index) != side) return;
    if (visited[index]) return;
    visited[index] = 1;
    lib[index] += 1;
    propagate_liberty(board, lib, visited, row+1, col, side);
    propagate_liberty(board, lib, visited, row-1, col, side);
    propagate_liberty(board, lib, visited, row, col+1, side);
    propagate_liberty(board, lib, visited, row, col-1, side);
}
```

함수 이름: propagate\_liberty

입력:

* float \*board: 게임판의 상태를 저장한 배열
* int \*lib: 각 돌의 자유도(해당 돌이 몇 개의 자유롭게 놓일 수 있는 곳과 연결되어 있는지)를 저장한 배열
* int \*visited: 해당 위치를 이미 방문했는지 여부를 저장한 배열
* int row: 현재 위치의 행 인덱스
* int col: 현재 위치의 열 인덱스
* int side: 현재 처리 중인 플레이어의 색깔 (1: 흑돌, 2: 백돌)

동작:&#x20;

* 현재 위치의 돌이 속한 그룹의 자유도를 계산하고, 각 돌의 자유도를 갱신하는 재귀 함수입니다.&#x20;
* 현재 위치의 돌과 같은 색깔이 아니거나 이미 방문한 위치라면 처리하지 않고, 그 외의 경우 현재 위치의 돌의 자유도를 1 증가시키고, 상하좌우의 위치에 대해 같은 작업을 반복합니다.

설명:&#x20;

* 바둑에서 돌의 자유도는 해당 돌이 자유롭게 놓일 수 있는 곳과 연결되어 있는 수를 나타냅니다.&#x20;
* propagate\_liberty 함수는 현재 위치의 돌이 속한 그룹의 자유도를 계산하기 위해 사용됩니다.&#x20;
* 이 함수는 재귀적으로 현재 위치와 인접한 위치를 방문하며, 같은 색깔의 돌이 발견될 때마다 해당 돌의 자유도를 1 증가시킵니다.&#x20;
* 이 과정에서 이미 방문한 위치는 다시 처리하지 않기 위해 visited 배열을 사용합니다.&#x20;
* 이 함수를 이용하여 모든 돌에 대해 자유도를 계산하면, 해당 그룹의 자유도를 모두 합산하여 반환할 수 있습니다.



## calculate\_liberties

```c
static int *calculate_liberties(float *board)
{
    int *lib = calloc(19*19, sizeof(int));
    int visited[19*19];
    int i, j;
    for(j = 0; j < 19; ++j){
        for(i = 0; i < 19; ++i){
            memset(visited, 0, 19*19*sizeof(int));
            int index = j*19 + i;
            if(!occupied(board,index)){
                if ((i > 0)  && occupied(board,index - 1)) propagate_liberty(board, lib, visited, j, i-1, occupied(board,index-1));
                if ((i < 18) && occupied(board,index + 1)) propagate_liberty(board, lib, visited, j, i+1, occupied(board,index+1));
                if ((j > 0)  && occupied(board,index - 19)) propagate_liberty(board, lib, visited, j-1, i, occupied(board,index-19));
                if ((j < 18) && occupied(board,index + 19)) propagate_liberty(board, lib, visited, j+1, i, occupied(board,index+19));
            }
        }
    }
    return lib;
}
```

함수 이름: calculate\_liberties

입력:&#x20;

* float 포인터형 board

동작:&#x20;

* 바둑판 상에서 각 돌의 자유도(돌이 살 수 있는 자유한 빈 교차점의 개수)를 계산하고, 이를 int 배열에 저장하여 반환한다.

설명:&#x20;

* 19x19 크기의 바둑판 상에서 각 교차점(Intersection)에 대해, 해당 교차점이 비어 있고, 상/하/좌/우 중 적어도 하나의 교차점에 같은 색의 돌이 있는 경우 해당 교차점의 자유도를 계산한다.&#x20;
* 이를 위해 해당 교차점에서 DFS(깊이 우선 탐색) 방식으로 상/하/좌/우로 이동하며 같은 색의 돌을 찾아나가고, 이러한 돌들의 자유도를 누적하여 계산한다. 계산된 자유도는 int 배열에 저장하여 반환된다.



## print\_board

```c
void print_board(FILE *stream, float *board, int player, int *indexes)
{
    int i,j,n;
    fprintf(stream, "   ");
    for(i = 0; i < 19; ++i){
        fprintf(stream, "%c ", 'A' + i + 1*(i > 7 && noi));
    }
    fprintf(stream, "\n");
    for(j = 0; j < 19; ++j){
        fprintf(stream, "%2d", (inverted) ? 19-j : j+1);
        for(i = 0; i < 19; ++i){
            int index = j*19 + i;
            if(indexes){
                int found = 0;
                for(n = 0; n < nind; ++n){
                    if(index == indexes[n]){
                        found = 1;
                        /*
                           if(n == 0) fprintf(stream, "\uff11");
                           else if(n == 1) fprintf(stream, "\uff12");
                           else if(n == 2) fprintf(stream, "\uff13");
                           else if(n == 3) fprintf(stream, "\uff14");
                           else if(n == 4) fprintf(stream, "\uff15");
                         */
                        fprintf(stream, " %d", n+1);
                    }
                }
                if(found) continue;
            }
            //if(board[index]*-swap > 0) fprintf(stream, "\u25C9 ");
            //else if(board[index]*-swap < 0) fprintf(stream, "\u25EF ");
            if      (occupied(board, index) == player) fprintf(stream, " X");
            else if (occupied(board, index) ==-player) fprintf(stream, " O");
            else fprintf(stream, " .");
        }
        fprintf(stream, "\n");
    }
}
```

함수 이름: print\_board

입력:

* stream: 출력 스트림
* board: 19x19 크기의 바둑판 상태를 나타내는 실수형 배열
* player: 현재 플레이어의 색상을 나타내는 정수 (1 또는 -1)
* indexes: 강조해야 하는 바둑알의 인덱스를 담고 있는 정수형 배열 (강조할 필요가 없는 경우 NULL)

동작:

* 입력된 바둑판 상태와 강조해야 하는 바둑알의 인덱스를 출력 스트림에 출력한다.
* 출력되는 바둑판은 아래와 같은 형식을 갖는다.
  * 첫 번째 행에는 알파벳 A부터 T까지 출력된다.
  * 각 행의 첫 번째 열에는 해당하는 행의 번호가 출력된다.
  * 바둑알은 "X"와 "O"로 출력된다.
  * 강조해야 하는 바둑알은 숫자 "1"부터 "5"까지로 표시된다.

설명:&#x20;

* 이 함수는 입력된 바둑판 상태와 강조해야 하는 바둑알의 인덱스를 출력하는 함수이다.&#x20;
* 바둑판은 19x19 크기의 실수형 배열로 표현되며, 각 원소는 해당 위치의 바둑알이 있는지 여부를 나타낸다.&#x20;
* 강조해야 하는 바둑알의 인덱스가 주어지면 해당 위치에 있는 바둑알이 강조된다.&#x20;
* 출력은 주어진 출력 스트림에 이루어지며, 바둑판은 위에서 설명한 형식으로 출력된다.



## flip\_board

```c
void flip_board(float *board)
{
    int i;
    for(i = 0; i < 19*19; ++i){
        float swap = board[i];
        board[i] = board[i+19*19];
        board[i+19*19] = swap;
        board[i+19*19*2] = 1-board[i+19*19*2];
    }
}
```

함수 이름: flip\_board

입력:&#x20;

* float형 포인터 변수 board (19\*19 크기의 바둑판 상태를 가리키는 포인터)

동작:&#x20;

* 주어진 바둑판 상태를 대칭으로 뒤집음

설명:&#x20;

* flip\_board 함수는 주어진 바둑판 상태를 대칭으로 뒤집는 함수입니다.&#x20;
* 함수 내부에서는 for문을 이용하여 바둑판의 상태를 저장하는 board 배열의 0\~18, 19\~37, 38\~56 인덱스를 각각 19\*19, 19\*19\*2, 19\*19\*3 인덱스와 바꿔주고,&#x20;
* 마지막으로 board 배열의 2번째 차원(열)의 상태를 1에서 빼준 값을 저장합니다.



## predict\_move2

```c
float predict_move2(network *net, float *board, float *move, int multi)
{
    float *output = network_predict(net, board);
    copy_cpu(19*19+1, output, 1, move, 1);
    float result = output[19*19 + 1];
    int i;
    if(multi){
        image bim = float_to_image(19, 19, 3, board);
        for(i = 1; i < 8; ++i){
            rotate_image_cw(bim, i);
            if(i >= 4) flip_image(bim);

            float *output = network_predict(net, board);
            image oim = float_to_image(19, 19, 1, output);
            result += output[19*19 + 1];

            if(i >= 4) flip_image(oim);
            rotate_image_cw(oim, -i);

            axpy_cpu(19*19+1, 1, output, 1, move, 1);

            if(i >= 4) flip_image(bim);
            rotate_image_cw(bim, -i);
        }
        result = result/8;
        scal_cpu(19*19+1, 1./8., move, 1);
    }
    for(i = 0; i < 19*19; ++i){
        if(board[i] || board[i+19*19]) move[i] = 0;
    }
    return result;
}
```

함수 이름: predict\_move2

입력:&#x20;

* network \*net (신경망 모델 포인터)
* float \*board (바둑판 상태)
* float \*move (바둑 수 예측 결과)
* int multi (여러 각도에서 예측할지 여부)

동작:&#x20;

* 입력된 신경망 모델과 바둑판 상태를 이용하여 바둑 수 예측을 수행하고, 예측 결과를 move에 복사한 후 반환합니다.&#x20;
* 여러 각도에서 예측할 경우 multi를 1로 설정하면 됩니다.&#x20;
* 이 경우 입력된 바둑판 상태를 회전 및 대칭 변환한 결과를 이용하여 예측을 수행하고, 여러 결과의 평균값을 최종 결과로 반환합니다.

설명:

* network \*net: 신경망 모델 포인터
* float \*board: 바둑판 상태
* float \*move: 바둑 수 예측 결과
* int multi: 여러 각도에서 예측할지 여부
* float \*output: 신경망 모델의 출력값
* copy\_cpu(): 배열 복사 함수
* image bim, oim: 입력 이미지와 출력 이미지를 나타내는 구조체
* rotate\_image\_cw(): 이미지 회전 함수
* flip\_image(): 이미지 대칭 변환 함수
* axpy\_cpu(): 벡터 연산 함수 (y = a\*x + y)
* scal\_cpu(): 벡터 연산 함수 (y = a\*x)



## remove\_connected

```c
static void remove_connected(float *b, int *lib, int p, int r, int c)
{
    if (r < 0 || r >= 19 || c < 0 || c >= 19) return;
    if (occupied(b, r*19 + c) != p) return;
    if (lib[r*19 + c] != 1) return;
    b[r*19 + c] = 0;
    b[19*19 + r*19 + c] = 0;
    remove_connected(b, lib, p, r+1, c);
    remove_connected(b, lib, p, r-1, c);
    remove_connected(b, lib, p, r, c+1);
    remove_connected(b, lib, p, r, c-1);
}
```

함수 이름: remove\_connected

입력:&#x20;

* float \*b: 오목판의 상태 정보를 담은 1차원 배열
* int \*lib: 각 좌표의 돌이 가지고 있는 자유도 수를 담은 1차원 배열
* int p: 현재 플레이어의 색깔을 나타내는 변수 (+1이면 흑돌, -1이면 백돌)
* int r: 좌표의 행 값 (0부터 18까지)
* int c: 좌표의 열 값 (0부터 18까지)

동작:&#x20;

* 주어진 좌표(r, c)에 있는 돌이 놓여 있는 그룹에서 자유도 수가 1인 돌들을 제거하고, 해당 돌들과 연결된 다른 자유도 수가 1인 돌들도 재귀적으로 제거한다.&#x20;
* (자유도 수가 1이 아닌 돌, 다른 플레이어의 돌, 좌표 범위를 벗어난 돌은 제거하지 않음)

설명:

* 해당 좌표(r, c)에 있는 돌이 현재 플레이어의 돌이 아니거나, 자유도 수가 1이 아닌 경우 제거하지 않는다.
* 자유도 수가 1인 돌일 경우, 해당 돌과 연결된 다른 자유도 수가 1인 돌들도 제거하기 위해 재귀 호출한다.
* 제거할 돌은 상태 배열 b와 반대 색깔 배열 b+19\*19에서 모두 0으로 바꾼다.



## move\_go

```c
void move_go(float *b, int p, int r, int c)
{
    int *l = calculate_liberties(b);
    if(p > 0) b[r*19 + c] = 1;
    else b[19*19 + r*19 + c] = 1;
    remove_connected(b, l, -p, r+1, c);
    remove_connected(b, l, -p, r-1, c);
    remove_connected(b, l, -p, r, c+1);
    remove_connected(b, l, -p, r, c-1);
    free(l);
}
```

함수 이름: move\_go

입력:

* b: float형 361 크기의 배열. 바둑판 상태를 나타냄.
* p: int형. 플레이어의 색을 나타냄. 1이면 흑돌, -1이면 백돌.
* r: int형. 돌을 놓을 행의 인덱스.
* c: int형. 돌을 놓을 열의 인덱스.

동작:

* 주어진 플레이어 색(p)과 위치(r, c)에 대해 돌을 놓는다.
* 상하좌우에 붙어있는 돌들 중, 아무도 다른 돌의 아이디어가 아닌 돌(자유도 1)들을 제거한다.

설명:

* move\_go 함수는 바둑판 상태(b)와 플레이어 색(p) 그리고 돌을 놓을 위치(r, c)를 입력으로 받아서 해당 위치에 돌을 놓고, 그 돌을 둘러싼 돌들 중에 자유도가 1인 돌들을 모두 제거하는 함수이다.
* 바둑판 상태는 float형 배열 b에 361개의 요소로 저장되며, 19x19 크기의 바둑판에서 각 점의 상태를 나타낸다.
* 플레이어 색은 int형 p로 주어지며, 1이면 흑돌, -1이면 백돌을 나타낸다.
* 돌을 놓을 위치는 0부터 시작하는 행과 열의 인덱스 r, c로 주어진다.
* remove\_connected 함수는 주어진 위치(r, c)와 연결되어 있는 돌들 중에, 자유도가 1인 돌들을 제거하는 함수이다.



## compare\_board

```c
int compare_board(float *a, float *b)
{
    if(memcmp(a, b, 19*19*3*sizeof(float)) == 0) return 1;
    return 0;
}
```

함수 이름: compare\_board

입력:

* a (float \*): 비교할 바둑판 상태 배열 포인터
* b (float \*): 비교할 바둑판 상태 배열 포인터

동작:

* a와 b가 같은지 비교하여 같으면 1, 다르면 0을 반환

설명:

* a와 b는 각각 19x19 크기의 바둑판 상태 배열을 가리키는 포인터이다.
* 함수 내부에서는 memcmp 함수를 사용하여 a와 b가 같은지를 비교하고, 같으면 1을 반환하고, 다르면 0을 반환한다.



## mcts\_tree

```c
typedef struct mcts_tree{
    float *board;
    struct mcts_tree **children;
    float *prior;
    int *visit_count;
    float *value;
    float *mean;
    float *prob;
    int total_count;
    float result;
    int done;
    int pass;
} mcts_tree;
```

구조체로 정의된 mcts\_tree는 게임 플레이에 필요한 정보들을 담고 있는 변수들의 모음이다. 이 구조체는 트리 형태의 데이터 구조를 가지고 있으며, 각 노드는 현재 게임 보드 상태, 자식 노드, 해당 노드에서의 확률, 해당 노드를 방문한 횟수, 해당 노드의 평균 가치 등의 정보를 담고 있다.

* float \*board: 현재 게임 보드 상태를 나타내는 1차원 실수형 배열
* struct mcts\_tree \*\*children: 자식 노드를 가리키는 포인터의 배열
* float \*prior: 해당 노드에서의 확률을 나타내는 1차원 실수형 배열
* int \*visit\_count: 해당 노드를 방문한 횟수를 나타내는 1차원 정수형 배열
* float \*value: 해당 노드의 가치를 나타내는 1차원 실수형 배열
* float \*mean: 해당 노드에서의 평균 가치를 나타내는 1차원 실수형 배열
* float \*prob: 해당 노드의 확률을 나타내는 1차원 실수형 배열
* int total\_count: 트리에서 전체 노드 수를 나타내는 정수형 변수
* float result: 게임 종료 후 최종 결과를 나타내는 실수형 변수
* int done: 게임이 끝났는지 여부를 나타내는 정수형 변수
* int pass: 패스를 한 번 이상 했는지 여부를 나타내는 정수형 변수



## free\_mcts

```c
void free_mcts(mcts_tree *root)
{
    if(!root) return;
    int i;
    free(root->board);
    for(i = 0; i < 19*19+1; ++i){
        if(root->children[i]) free_mcts(root->children[i]);
    }
    free(root->children);
    free(root->prior);
    free(root->visit_count);
    free(root->value);
    free(root->mean);
    free(root->prob);
    free(root);
}
```

함수 이름: free\_mcts

입력:&#x20;

* mcts\_tree \*root: 해제할 MCTS 트리의 루트 노드 포인터

동작:&#x20;

* MCTS 트리의 모든 노드와 할당된 동적 메모리를 해제함

설명:&#x20;

* free\_mcts 함수는 MCTS 트리의 루트 노드 포인터를 입력으로 받아서 해당 트리의 모든 노드와 할당된 동적 메모리를 해제합니다.&#x20;
* 이 함수를 사용하여 MCTS 트리의 메모리 누수를 방지할 수 있습니다.



## network\_predict\_rotations

```c
float *network_predict_rotations(network *net, float *next)
{
    int n = net->batch;
    float *in = calloc(19*19*3*n, sizeof(float));
    image im = float_to_image(19, 19, 3, next);
    int i,j;
    int *inds = random_index_order(0, 8);
    for(j = 0; j < n; ++j){
        i = inds[j];
        rotate_image_cw(im, i);
        if(i >= 4) flip_image(im);
        memcpy(in + 19*19*3*j, im.data, 19*19*3*sizeof(float));
        if(i >= 4) flip_image(im);
        rotate_image_cw(im, -i);
    }
    float *pred = network_predict(net, in);
    for(j = 0; j < n; ++j){
        i = inds[j];
        image im = float_to_image(19, 19, 1, pred + j*(19*19 + 2));
        if(i >= 4) flip_image(im);
        rotate_image_cw(im, -i);
        if(j > 0){
            axpy_cpu(19*19+2, 1, im.data, 1, pred, 1);
        }
    }
    free(in);
    free(inds);
    scal_cpu(19*19+2, 1./n, pred, 1);
    return pred;
}
```

함수 이름: network\_predict\_rotations

입력:

* network \*net: 신경망 모델을 나타내는 포인터
* float \*next: 바둑판 상태를 나타내는 1차원 배열 포인터

동작:

* 바둑판 상태를 다양한 회전, 대칭 변환을 적용하여 데이터를 증강하고, 이를 이용해 신경망 모델을 이용하여 다음 수 예측 결과를 반환

설명:

* 입력으로 받은 바둑판 상태(next)를 float\_to\_image 함수를 이용해 이미지 형식으로 변환
* 변환된 이미지에 8가지 방향(0, 90, 180, 270도 회전 및 각각의 대칭 변환)을 적용하여 데이터를 증강하고, 이를 하나의 배치로 묶어서 in 배열에 저장
* in 배열을 신경망 모델(net)에 입력하여 다음 수 예측 값을 구함
* 구한 예측 값들을 다시 각 방향, 대칭 변환에 맞게 회전, 대칭 변환하여 다시 하나의 예측 값으로 합침
* 최종 예측 값을 반환하기 전에 예측 값들을 모두 더하고 배치 크기로 나누어 평균값을 구함



## expand

```c
mcts_tree *expand(float *next, float *ko, network *net)
{
    mcts_tree *root = calloc(1, sizeof(mcts_tree));
    root->board = next;
    root->children = calloc(19*19+1, sizeof(mcts_tree*));
    root->prior = calloc(19*19 + 1, sizeof(float));
    root->prob = calloc(19*19 + 1, sizeof(float));
    root->mean = calloc(19*19 + 1, sizeof(float));
    root->value = calloc(19*19 + 1, sizeof(float));
    root->visit_count = calloc(19*19 + 1, sizeof(int));
    root->total_count = 1;
    int i;
    float *pred = network_predict_rotations(net, next);
    copy_cpu(19*19+1, pred, 1, root->prior, 1);
    float val = 2*pred[19*19 + 1] - 1;
    root->result = val;
    for(i = 0; i < 19*19+1; ++i) {
        root->visit_count[i] = 0;
        root->value[i] = 0;
        root->mean[i] = val;
        if(i < 19*19 && occupied(next, i)){
            root->value[i] = -1;
            root->mean[i] = -1;
            root->prior[i] = 0;
        }
    }
    //print_board(stderr, next, flip?-1:1, 0);
    return root;
}
```

함수 이름: expand&#x20;

입력:

* float 포인터형 변수 next: 복사할 바둑판 상태
* float 포인터형 변수 ko: 현재 코의 위치
* network 구조체형 변수 net: 딥러닝 모델

동작:&#x20;

* 주어진 바둑판(next)을 복사하여 미래의 상황을 대비한다. 그 다음, 딥러닝 모델(net)을 사용하여 next에 대한 확률 분포를 예측한다.&#x20;
* 이 예측 결과를 기반으로, 각 수의 우선순위를 구하고, 이를 prior 배열에 저장한다.&#x20;
* 그리고 해당 노드의 mean, value, visit\_count, total\_count 등의 변수들을 초기화하고, 결과를 저장할 result 변수에도 값을 할당한다.

설명:&#x20;

* 주어진 바둑판을 복사하여 next에 저장한다.&#x20;
* 이 때, next 배열의 크기는 19\*19\*3으로 할당된다.&#x20;
* 그 다음, mcts\_tree 구조체형 변수 root를 생성하고, 초기화를 진행한다. children 배열과 prior, prob, mean, value, visit\_count, total\_count 등의 변수들을 동적 할당하여 초기화한다.
* 확률 분포를 구하기 위해, network\_predict\_rotations 함수를 이용하여 딥러닝 모델(net)을 적용하여 next에 대한 예측 값을 계산한다. 그리고 예측 값(pred) 중 마지막 원소는 패스의 확률을 의미하므로, val에 2\*pred\[19\*19 + 1] - 1의 값을 할당한다.
* 그 다음, prior 배열에는 각 수의 우선순위를 저장한다. 예를 들어, prior\[0]은 바둑판의 왼쪽 위 모서리에 있는 점의 우선순위를 나타낸다.
* value 배열은 해당 노드를 방문했을 때 받을 수 있는 보상을 나타내며, mean은 방문 횟수를 고려하여 평균 보상 값을 나타낸다.
* visit\_count는 해당 노드를 방문한 횟수를 나타내며, total\_count는 해당 노드의 자식 노드들도 포함하여 전체 방문 횟수를 나타낸다.
* 마지막으로, result 변수에 val 값을 할당하여 노드의 결과 값을 저장한다.



## copy\_board

```c
float *copy_board(float *board)
{
    float *next = calloc(19*19*3, sizeof(float));
    copy_cpu(19*19*3, board, 1, next, 1);
    return next;
}
```

함수 이름: copy\_board&#x20;

입력:&#x20;

* float 포인터 변수 board (복사할 바둑판)&#x20;

동작:&#x20;

* 19x19 크기의 바둑판을 동적으로 할당한 후, 입력된 바둑판(board)을 복사하여 할당한 메모리(next)에 저장하고, next를 반환함.&#x20;

설명:&#x20;

* 입력된 바둑판(board)를 동적으로 할당한 메모리(next)에 복사하여 저장하고, next를 반환하는 함수입니다.&#x20;
* 바둑판은 19x19 크기이며, 각 좌표마다 세 가지 채널(R, G, B)을 가지고 있기 때문에 19x19x3 크기의 메모리를 할당합니다.&#x20;
* 함수 내부에서는 copy\_cpu 함수를 사용하여 입력된 바둑판(board)의 값을 복사하여 next에 저장합니다.&#x20;
* 이후 next를 반환합니다.



## select\_mcts

```c
float select_mcts(mcts_tree *root, network *net, float *prev, float cpuct)
{
    if(root->done) return -root->result;
    int i;
    float max = -1000;
    int max_i = 0;
    for(i = 0; i < 19*19+1; ++i){
        root->prob[i] = root->mean[i] + cpuct*root->prior[i] * sqrt(root->total_count) / (1. + root->visit_count[i]);
        if(root->prob[i] > max){
            max = root->prob[i];
            max_i = i;
        }
    }
    float val;
    i = max_i;
    root->visit_count[i]++;
    root->total_count++;
    if (root->children[i]) {
        val = select_mcts(root->children[i], net, root->board, cpuct);
    } else {
        if(max_i < 19*19 && !legal_go(root->board, prev, 1, max_i/19, max_i%19)) {
            root->mean[i]  = -1;
            root->value[i] = -1;
            root->prior[i] = 0;
            --root->total_count;
            return select_mcts(root, net, prev, cpuct);
            //printf("Detected ko\n");
            //getchar();
        } else {
            float *next = copy_board(root->board);
            if (max_i < 19*19) {
                move_go(next, 1, max_i / 19, max_i % 19);
            }
            flip_board(next);
            root->children[i] = expand(next, root->board, net);
            val = -root->children[i]->result;
            if(max_i == 19*19){
                root->children[i]->pass = 1;
                if (root->pass){
                    root->children[i]->done = 1;
                }
            }
        }
    }
    root->value[i] += val;
    root->mean[i] = root->value[i]/root->visit_count[i];
    return -val;
}
```

함수 이름: select\_mcts

입력:

* mcts\_tree \*root: MCTS 트리의 루트 노드
* network \*net: 신경망
* float \*prev: 이전 판의 상태
* float cpuct: Cpuct 하이퍼파라미터 값

동작:&#x20;

* MCTS 알고리즘에서 가장 중요한 함수 중 하나로, 선택 단계를 수행한다.&#x20;
* MCTS 트리를 순회하며 UCB1 알고리즘을 이용하여 exploration과 exploitation을 균형있게 수행하면서 자식 노드 중에 최적의 선택을 찾는다.

설명:

* UCB1 알고리즘에 따라, MCTS 트리의 각 노드에 대해 다음 값을 계산하고, 가장 큰 값을 가진 자식 노드를 선택한다.
  * root->prob\[i] = root->mean\[i] + cpuct\*root->prior\[i] \* sqrt(root->total\_count) / (1. + root->visit\_count\[i]);
* 선택된 자식 노드가 이미 존재하는 경우, 해당 노드로 이동하여 재귀적으로 select\_mcts 함수를 호출한다. 이때, 값은 -val로 반환한다.
* 선택된 자식 노드가 존재하지 않는 경우, 다음 과정을 수행한다.
  * 선택된 위치가 유효한지 확인한다. 만약 유효하지 않으면 해당 노드에 대한 정보를 업데이트하고 select\_mcts 함수를 재귀적으로 호출한다.
  * 유효한 경우, 새로운 판 상태를 만들어 다음 노드를 확장한다.
  * 만약 선택된 위치가 패스인 경우, 해당 노드에 대한 정보를 업데이트하고, 상위 노드의 패스 여부에 따라 노드를 완료 상태로 변경한다.
* 선택된 자식 노드에 대한 값을 업데이트하고 반환한다.



## run\_mcts

```c
mcts_tree *run_mcts(mcts_tree *tree, network *net, float *board, float *ko, int player, int n, float cpuct, float secs)
{
    int i;
    double t = what_time_is_it_now();
    if(player < 0) flip_board(board);
    if(!tree) tree = expand(copy_board(board), ko, net);
    assert(compare_board(tree->board, board));
    for(i = 0; i < n; ++i){
        if (secs > 0 && (what_time_is_it_now() - t) > secs) break;
        int max_i = max_int_index(tree->visit_count, 19*19+1);
        if (tree->visit_count[max_i] >= n) break;
        select_mcts(tree, net, ko, cpuct);
    }
    if(player < 0) flip_board(board);
    //fprintf(stderr, "%f Seconds\n", what_time_is_it_now() - t);
    return tree;
}
```

함수 이름: run\_mcts

입력:

* mcts\_tree \*tree: MCTS 트리의 루트 노드 포인터
* network \*net: 신경망 모델 포인터
* float \*board: 현재 게임 보드 상태를 나타내는 1차원 배열 포인터
* float \*ko: 코(禁)의 위치를 나타내는 1차원 배열 포인터
* int player: 현재 수를 놓는 플레이어
* int n: MCTS 알고리즘 반복 횟수
* float cpuct: MCTS 알고리즘의 탐색-확률 균형 매개변수
* float secs: MCTS 알고리즘 실행 시간 제한(초)

동작:&#x20;

* 주어진 보드와 코 상태를 이용하여 MCTS 알고리즘을 n번 반복하고, 그 결과로 얻은 새로운 MCTS 트리의 루트 노드 포인터를 반환한다.&#x20;
* 이 때 cpuct와 secs 매개변수를 사용하여 MCTS 알고리즘의 동작을 제어하며, 시간 제한에 도달하거나 MCTS 트리에서 방문 횟수가 가장 많은 자식 노드가 n회 이상 방문되면 알고리즘을 종료한다.

설명:

* MCTS: 몬테 카를로 트리 탐색(Monte Carlo tree search) 알고리즘
* MCTS 트리: 게임의 가능한 모든 수를 탐색하기 위한 트리 자료구조
* MCTS 알고리즘: MCTS 트리를 활용하여 게임을 탐색하고, 최적의 수를 결정하는 알고리즘
* 신경망 모델: 딥러닝을 이용하여 게임 상태를 입력으로 받고, 수의 확률 분포와 이길 확률 등의 출력을 내는 모델
* 코(禁): 바둑 게임에서 금수 규칙을 적용하기 위한 규칙. 금수는 한 수를 둔 후, 상대방이 이전에 둔 위치에 돌을 놓는 것이 금지된 상태를 말한다.



## move\_mcts

```c
mcts_tree *move_mcts(mcts_tree *tree, int index)
{
    if(index < 0 || index > 19*19 || !tree || !tree->children[index]) {
        free_mcts(tree);
        tree = 0;
    } else {
        mcts_tree *swap = tree;
        tree = tree->children[index];
        swap->children[index] = 0;
        free_mcts(swap);
    }
    return tree;
}
```

함수 이름: move\_mcts

입력:&#x20;

* mcts\_tree 구조체 포인터인 tree
* 정수형 변수 index

동작:&#x20;

* 입력으로 받은 index를 사용하여 tree 구조체의 자식 중 해당 인덱스를 가진 구조체를 선택하고, 그 구조체로 tree를 갱신한다.&#x20;
* 만약 index가 유효하지 않거나 해당 인덱스를 가진 자식 구조체가 없으면 tree를 해제하고 0으로 초기화한다.

설명:

* mcts\_tree: 몬테카를로 트리를 구현한 구조체
* free\_mcts: 몬테카를로 트리를 해제하는 함수



## move

```c
typedef struct {
    float value;
    float mcts;
    int row;
    int col;
} move;
```

함수 이름: 없음&#x20;

입력: 없음&#x20;

동작:&#x20;

* 게임의 수를 저장하는 구조체 move를 정의한다.&#x20;

설명:

* typedef: 새로운 자료형을 정의하는 예약어
* struct: 구조체를 정의하는 예약어
* move: 구조체의 이름
* value: 해당 수의 예상 가치
* mcts: 해당 수의 MCTS (Monte Carlo Tree Search) 점수
* row: 해당 수의 행 좌표
* col: 해당 수의 열 좌표



## pick\_move

```c
move pick_move(mcts_tree *tree, float temp, int player)
{
    int i;
    float probs[19*19+1] = {0};
    move m = {0};
    double sum = 0;
    /*
    for(i = 0; i < 19*19+1; ++i){
        probs[i] = tree->visit_count[i];
    }
    */
    //softmax(probs, 19*19+1, temp, 1, probs);
    for(i = 0; i < 19*19+1; ++i){
        sum += pow(tree->visit_count[i], 1./temp);
    }
    for(i = 0; i < 19*19+1; ++i){
        probs[i] = pow(tree->visit_count[i], 1./temp) / sum;
    }

    int index = sample_array(probs, 19*19+1);
    m.row = index / 19;
    m.col = index % 19;
    m.value = (tree->result+1.)/2.;
    m.mcts  = (tree->mean[index]+1.)/2.;

    int indexes[nind];
    top_k(probs, 19*19+1, nind, indexes);
    print_board(stderr, tree->board, player, indexes);

    fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", index/19, index%19, tree->result, tree->prior[index], probs[index], tree->mean[index], (tree->children[index])?tree->children[index]->result:0, tree->visit_count[index]);
    int ind = max_index(probs, 19*19+1);
    fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", ind/19, ind%19, tree->result, tree->prior[ind], probs[ind], tree->mean[ind], (tree->children[ind])?tree->children[ind]->result:0, tree->visit_count[ind]);
    ind = max_index(tree->prior, 19*19+1);
    fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", ind/19, ind%19, tree->result, tree->prior[ind], probs[ind], tree->mean[ind], (tree->children[ind])?tree->children[ind]->result:0, tree->visit_count[ind]);
    return m;
}
```

함수 이름: pick\_move&#x20;

입력:&#x20;

* mcts\_tree 구조체 포인터 tree
* float 타입의 temp
* int 타입의 player&#x20;

동작:&#x20;

* MCTS(Monte Carlo Tree Search) 알고리즘을 이용하여 주어진 게임 트리에서 다음 수를 선택하는 함수입니다.&#x20;
* 현재 상태의 트리와 탐색을 위한 온도(temp), 현재 플레이어(player)를 입력으로 받습니다.&#x20;
* 각 수의 확률을 계산하고 softmax 함수를 적용하여 확률 분포를 만듭니다. 그 다음, 이 확률 분포에 따라 다음 수를 샘플링합니다.&#x20;
* 선택된 수와 해당 수의 확률, 평균 가치 등을 출력합니다.

설명:

* move: 게임에서의 수를 나타내는 구조체로, int 타입의 row와 col, float 타입의 value와 mcts로 이루어져 있습니다.
* mcts\_tree: MCTS 알고리즘에서 사용되는 게임 트리를 나타내는 구조체로, 현재 상태의 보드(board), 수의 방문 횟수(visit\_count), 사전 확률(prior), 자식 노드(children), 평균 가치(mean) 등의 정보를 담고 있습니다.
* softmax: 확률 분포를 만들기 위한 함수로, 주어진 배열을 입력으로 받아 softmax 함수를 적용한 결과를 출력합니다.
* sample\_array: 주어진 확률 분포에 따라 샘플링하여 선택된 인덱스를 반환하는 함수입니다.
* top\_k: 주어진 배열에서 가장 큰 k개의 원소의 인덱스를 반환하는 함수입니다.
* print\_board: 현재 상태의 보드를 출력하는 함수입니다.



## valid\_go

```c
void valid_go(char *cfgfile, char *weightfile, int multi, char *filename)
{
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    float *board = calloc(19*19*3, sizeof(float));
    float *move = calloc(19*19+2, sizeof(float));
    // moves m = load_go_moves("/home/pjreddie/backup/go.test");
    moves m = load_go_moves(filename);

    int N = m.n;
    int i,j;
    int correct = 0;
    for (i = 0; i <N; ++i) {
        char *b = m.data[i];
        int player = b[0] - '0';
        //int result = b[1] - '0';
        int row = b[2];
        int col = b[3];
        int truth = col + 19*row;
        string_to_board(b+4, board);
        if(player > 0) for(j = 0; j < 19*19; ++j) board[19*19*2 + j] = 1;
        predict_move2(net, board, move, multi);
        int index = max_index(move, 19*19+1);
        if(index == truth) ++correct;
        printf("%d Accuracy %f\n", i, (float) correct/(i+1));
    }
}
```

함수 이름: valid\_go

입력:

* char \*cfgfile: 구성 파일 경로
* char \*weightfile: 가중치 파일 경로
* int multi: 사용할 GPU 수
* char \*filename: 게임 파일 경로

동작:&#x20;

* 주어진 구성 파일과 가중치 파일을 사용하여 신경망을 로드하고, 주어진 게임 파일에서 수를 읽어 신경망을 사용해 이동 예측을 수행하고 정확도를 출력하는 함수.

설명:

* 먼저 basecfg 함수를 사용하여 구성 파일 경로에서 파일 이름을 가져와 출력한다.
* load\_network 함수를 사용하여 구성 파일과 가중치 파일에서 신경망을 로드하고, set\_batch\_network 함수를 사용하여 배치 크기를 1로 설정한다.
* 이후 게임 파일에서 수를 읽어들이고, 각 수에 대해 다음을 수행한다:
  * 문자열로 된 보드 상태를 실수형 배열로 변환한다.
  * 예측을 위한 출력 배열(move)을 초기화한다.
  * predict\_move2 함수를 사용하여 주어진 보드 상태(board)에 대한 예측을 수행한다.
  * max\_index 함수를 사용하여 가장 확률이 높은 인덱스를 찾고, 이를 실제 정답과 비교하여 정확도를 계산한다.
  * 정확도를 출력한다.



## print\_game

```c
int print_game(float *board, FILE *fp)
{
    int i, j;
    int count = 3;
    fprintf(fp, "komi 6.5\n");
    fprintf(fp, "boardsize 19\n");
    fprintf(fp, "clear_board\n");
    for(j = 0; j < 19; ++j){
        for(i = 0; i < 19; ++i){
            if(occupied(board,j*19 + i) == 1) fprintf(fp, "play black %c%d\n", 'A'+i+(i>=8), 19-j);
            if(occupied(board,j*19 + i) == -1) fprintf(fp, "play white %c%d\n", 'A'+i+(i>=8), 19-j);
            if(occupied(board,j*19 + i)) ++count;
        }
    }
    return count;
}
```

함수 이름: print\_game

입력:&#x20;

* float 포인터형 변수 board
* FILE 포인터형 변수 fp

동작:&#x20;

* 19x19 오목 게임의 현재 상태를 출력한다.&#x20;
* komi와 boardsize 정보를 출력하고, clear\_board 명령어를 통해 초기화한 후, 현재 상태에 맞게 play 명령어를 출력한다.&#x20;
* 이때, board 배열에서 해당 좌표의 돌이 흑돌인지 백돌인지 확인하여, play black 또는 play white 명령어를 출력한다.&#x20;
* 마지막으로, board 배열을 순회하며 돌이 놓여있는 칸의 개수를 세고, 그 값을 반환한다.

설명:

* occupied(board,j\*19 + i): board 배열에서 (i,j) 좌표에 있는 돌의 상태를 반환한다. 흑돌이면 1, 백돌이면 -1, 돌이 없으면 0을 반환한다.
* fprintf(fp, "play color %c%d\n", 'A'+i+(i>=8), 19-j): 오목 서버에게 현재 상태를 전달하기 위해 play 명령어를 출력한다.&#x20;
* color는 흑돌인지 백돌인지에 따라 black 또는 white로 대체되며, %c와 %d에 각각 (i,j) 좌표의 문자와 숫자 값이 들어간다.&#x20;
* 이때, %c에 'A'부터 시작하는 알파벳 중 i번째 문자를 선택하며, 만약 i>=8이면 다음 문자를 선택한다. %d에는 19-j가 들어간다.&#x20;
* 이는 오목 서버에서 y좌표가 1부터 시작하는 반면, board 배열에서 y좌표가 0부터 시작하기 때문에 좌표를 변환하는 과정이다.
* count: board 배열을 순회하며 돌이 놓여있는 칸의 개수를 저장한다. 이 값은 반환되어 게임 진행 중에 몇 수가 놓였는지 알려준다.



## stdin\_ready

```c
int stdin_ready()
{
    fd_set readfds;
    FD_ZERO(&readfds);

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;
    FD_SET(STDIN_FILENO, &readfds);

    if (select(1, &readfds, NULL, NULL, &timeout)){
        return 1;
    }
    return 0;
}
```

함수 이름: stdin\_ready&#x20;

입력: 없음&#x20;

동작:&#x20;

* 표준 입력 (stdin)에 대해 non-blocking으로 확인하고, 입력이 준비되어 있으면 1을 반환하고, 준비되어 있지 않으면 0을 반환합니다.&#x20;

설명:&#x20;

* 이 함수는 표준 입력 (stdin)이 준비되어 있는지 확인합니다.&#x20;
* 이를 위해 select() 시스템 호출을 사용하며, 이 함수는 호출 후 즉시 반환됩니다.&#x20;
* 입력이 준비되어 있으면 1을 반환하고, 준비되어 있지 않으면 0을 반환합니다.&#x20;
* 따라서 이 함수는 입력이 준비될 때까지 블로킹하지 않습니다.



## ponder

```c
mcts_tree *ponder(mcts_tree *tree, network *net, float *b, float *ko, int player, float cpuct)
{
    double t = what_time_is_it_now();
    int count = 0;
    if (tree) count = tree->total_count;
    while(!stdin_ready()){
        if (what_time_is_it_now() - t > 120) break;
        tree = run_mcts(tree, net, b, ko, player, 100000, cpuct, .1);
    }
    fprintf(stderr, "Pondered %d moves...\n", tree->total_count - count);
    return tree;
}
```

함수 이름: ponder&#x20;

입력:

* mcts\_tree \*tree: 현재 상태를 나타내는 MCTS 트리
* network \*net: 신경망 모델
* float \*b: 현재 바둑판 상태
* float \*ko: 코(禁) 상태 (바둑 규칙 중 하나)
* int player: 플레이어 (1: 흑돌, -1: 백돌)
* float cpuct: 탐색에서 탐욕적인 정도를 결정하는 파라미터

동작:&#x20;

* 편집기 등을 통해 새로운 입력이 들어올 때까지 지정된 시간(여기서는 120초) 동안 MCTS 트리를 계속 탐색하고 업데이트한다.&#x20;
* 그리고 이동 수를 출력하며, 마지막으로 업데이트된 MCTS 트리를 반환한다.

설명:

* 함수 ponder는 현재 상태를 나타내는 MCTS 트리를 입력으로 받고, 입력이 들어올 때까지 지정된 시간 동안 계속해서 MCTS 알고리즘을 수행한다.
* 이 함수는 주로 게임 AI에서 사용되며, 입력을 기다리는 동안 컴퓨터가 최선의 수를 고민하는 것을 '고민하다(pondering)'라고 표현한다.
* 입력이 들어오면 새로운 MCTS 트리를 생성하고, 기존의 MCTS 트리에서 계속해서 탐색하며 업데이트한다.
* 마지막으로 업데이트된 MCTS 트리와 이동 수를 출력하며 반환한다.



## engine\_go

```c
void engine_go(char *filename, char *weightfile, int mcts_iters, float secs, float temp, float cpuct, int anon, int resign)
{
    mcts_tree *root = 0;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));
    float *board = calloc(19*19*3, sizeof(float));
    flip_board(board);
    float *one = calloc(19*19*3, sizeof(float));
    float *two = calloc(19*19*3, sizeof(float));
    int ponder_player = 0;
    int passed = 0;
    int move_num = 0;
    int main_time = 0;
    int byo_yomi_time = 0;
    int byo_yomi_stones = 0;
    int black_time_left = 0;
    int black_stones_left = 0;
    int white_time_left = 0;
    int white_stones_left = 0;
    float orig_time = secs;
    int old_ponder = 0;
    while(1){
        if(ponder_player){
            root = ponder(root, net, board, two, ponder_player, cpuct);
        }
        old_ponder = ponder_player;
        ponder_player = 0;
        char buff[256];
        int id = 0;
        int has_id = (scanf("%d", &id) == 1);
        scanf("%s", buff);
        if (feof(stdin)) break;
        fprintf(stderr, "%s\n", buff);
        char ids[256];
        sprintf(ids, "%d", id);
        //fprintf(stderr, "%s\n", buff);
        if (!has_id) ids[0] = 0;
        if (!strcmp(buff, "protocol_version")){
            printf("=%s 2\n\n", ids);
        } else if (!strcmp(buff, "name")){
            if(anon){
                printf("=%s The Fool!\n\n", ids);
            }else{
                printf("=%s DarkGo\n\n", ids);
            }
        } else if (!strcmp(buff, "time_settings")){
            ponder_player = old_ponder;
            scanf("%d %d %d", &main_time, &byo_yomi_time, &byo_yomi_stones);
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "time_left")){
            ponder_player = old_ponder;
            char color[256];
            int time = 0, stones = 0;
            scanf("%s %d %d", color, &time, &stones);
            if (color[0] == 'b' || color[0] == 'B'){
                black_time_left = time;
                black_stones_left = stones;
            } else {
                white_time_left = time;
                white_stones_left = stones;
            }
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "version")){
            if(anon){
                printf("=%s :-DDDD\n\n", ids);
            }else {
                printf("=%s 1.0. Want more DarkGo? You can find me on OGS, unlimited games, no waiting! https://online-go.com/user/view/434218\n\n", ids);
            }
        } else if (!strcmp(buff, "known_command")){
            char comm[256];
            scanf("%s", comm);
            int known = (!strcmp(comm, "protocol_version") ||
                    !strcmp(comm, "name") ||
                    !strcmp(comm, "version") ||
                    !strcmp(comm, "known_command") ||
                    !strcmp(comm, "list_commands") ||
                    !strcmp(comm, "quit") ||
                    !strcmp(comm, "boardsize") ||
                    !strcmp(comm, "clear_board") ||
                    !strcmp(comm, "komi") ||
                    !strcmp(comm, "final_status_list") ||
                    !strcmp(comm, "play") ||
                    !strcmp(comm, "genmove_white") ||
                    !strcmp(comm, "genmove_black") ||
                    !strcmp(comm, "fixed_handicap") ||
                    !strcmp(comm, "genmove"));
            if(known) printf("=%s true\n\n", ids);
            else printf("=%s false\n\n", ids);
        } else if (!strcmp(buff, "list_commands")){
            printf("=%s protocol_version\nshowboard\nname\nversion\nknown_command\nlist_commands\nquit\nboardsize\nclear_board\nkomi\nplay\ngenmove_black\ngenmove_white\ngenmove\nfinal_status_list\nfixed_handicap\n\n", ids);
        } else if (!strcmp(buff, "quit")){
            break;
        } else if (!strcmp(buff, "boardsize")){
            int boardsize = 0;
            scanf("%d", &boardsize);
            //fprintf(stderr, "%d\n", boardsize);
            if(boardsize != 19){
                printf("?%s unacceptable size\n\n", ids);
            } else {
                root = move_mcts(root, -1);
                memset(board, 0, 3*19*19*sizeof(float));
                flip_board(board);
                move_num = 0;
                printf("=%s \n\n", ids);
            }
        } else if (!strcmp(buff, "fixed_handicap")){
            int handicap = 0;
            scanf("%d", &handicap);
            int indexes[] = {72, 288, 300, 60, 180, 174, 186, 66, 294};
            int i;
            for(i = 0; i < handicap; ++i){
                board[indexes[i]] = 1;   
                ++move_num;
            }
            root = move_mcts(root, -1);
        } else if (!strcmp(buff, "clear_board")){
            passed = 0;
            memset(board, 0, 3*19*19*sizeof(float));
            flip_board(board);
            move_num = 0;
            root = move_mcts(root, -1);
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "komi")){
            float komi = 0;
            scanf("%f", &komi);
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "showboard")){
            printf("=%s \n", ids);
            print_board(stdout, board, 1, 0);
            printf("\n");
        } else if (!strcmp(buff, "play") || !strcmp(buff, "black") || !strcmp(buff, "white")){
            ++move_num;
            char color[256];
            if(!strcmp(buff, "play"))
            {
                scanf("%s ", color);
            } else {
                scanf(" ");
                color[0] = buff[0];
            }
            char c;
            int r;
            int count = scanf("%c%d", &c, &r);
            int player = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
            if((c == 'p' || c == 'P') && count < 2) {
                passed = 1;
                printf("=%s \n\n", ids);
                char *line = fgetl(stdin);
                free(line);
                fflush(stdout);
                fflush(stderr);
                root = move_mcts(root, 19*19);
                continue;
            } else {
                passed = 0;
            }
            if(c >= 'A' && c <= 'Z') c = c - 'A';
            if(c >= 'a' && c <= 'z') c = c - 'a';
            if(c >= 8) --c;
            r = 19 - r;
            fprintf(stderr, "move: %d %d\n", r, c);

            float *swap = two;
            two = one;
            one = swap;
            move_go(board, player, r, c);
            copy_cpu(19*19*3, board, 1, one, 1);
            if(root) fprintf(stderr, "Prior: %f\n", root->prior[r*19 + c]);
            if(root) fprintf(stderr, "Mean: %f\n", root->mean[r*19 + c]);
            if(root) fprintf(stderr, "Result: %f\n", root->result);
            root = move_mcts(root, r*19 + c);
            if(root) fprintf(stderr, "Visited: %d\n", root->total_count);
            else fprintf(stderr, "NOT VISITED\n");

            printf("=%s \n\n", ids);
            //print_board(stderr, board, 1, 0);
        } else if (!strcmp(buff, "genmove") || !strcmp(buff, "genmove_black") || !strcmp(buff, "genmove_white")){
            ++move_num;
            int player = 0;
            if(!strcmp(buff, "genmove")){
                char color[256];
                scanf("%s", color);
                player = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
            } else if (!strcmp(buff, "genmove_black")){
                player = 1;
            } else {
                player = -1;
            }
            if(player > 0){
                if(black_time_left <= 30) secs = 2.5;
                else secs = orig_time;
            } else {
                if(white_time_left <= 30) secs = 2.5;
                else secs = orig_time;
            }
            ponder_player = -player;

            //tree = generate_move(net, player, board, multi, .1, two, 1);
            double t = what_time_is_it_now();
            root = run_mcts(root, net, board, two, player, mcts_iters, cpuct, secs);
            fprintf(stderr, "%f Seconds\n", what_time_is_it_now() - t);
            move m = pick_move(root, temp, player);
            root = move_mcts(root, m.row*19 + m.col);


            if(move_num > resign && m.value < .1 && m.mcts < .1){
                printf("=%s resign\n\n", ids);
            } else if(m.row == 19){
                printf("=%s pass\n\n", ids);
                passed = 0;
            } else {
                int row = m.row;
                int col = m.col;

                float *swap = two;
                two = one;
                one = swap;

                move_go(board, player, row, col);
                copy_cpu(19*19*3, board, 1, one, 1);
                row = 19 - row;
                if (col >= 8) ++col;
                printf("=%s %c%d\n\n", ids, 'A' + col, row);
            }

        } else if (!strcmp(buff, "p")){
            //print_board(board, 1, 0);
        } else if (!strcmp(buff, "final_status_list")){
            char type[256];
            scanf("%s", type);
            fprintf(stderr, "final_status\n");
            char *line = fgetl(stdin);
            free(line);
            if(type[0] == 'd' || type[0] == 'D'){
                int i;
                FILE *f = fopen("game.txt", "w");
                int count = print_game(board, f);
                fprintf(f, "%s final_status_list dead\n", ids);
                fclose(f);
                FILE *p = popen("./gnugo --mode gtp < game.txt", "r");
                for(i = 0; i < count; ++i){
                    free(fgetl(p));
                    free(fgetl(p));
                }
                char *l = 0;
                while((l = fgetl(p))){
                    printf("%s\n", l);
                    free(l);
                }
            } else {
                printf("?%s unknown command\n\n", ids);
            }
        } else if (!strcmp(buff, "kgs-genmove_cleanup")){
            char type[256];
            scanf("%s", type);
            fprintf(stderr, "kgs-genmove_cleanup\n");
            char *line = fgetl(stdin);
            free(line);
            int i;
            FILE *f = fopen("game.txt", "w");
            int count = print_game(board, f);
            fprintf(f, "%s kgs-genmove_cleanup %s\n", ids, type);
            fclose(f);
            FILE *p = popen("./gnugo --mode gtp < game.txt", "r");
            for(i = 0; i < count; ++i){
                free(fgetl(p));
                free(fgetl(p));
            }
            char *l = 0;
            while((l = fgetl(p))){
                printf("%s\n", l);
                free(l);
            }
        } else {
            char *line = fgetl(stdin);
            free(line);
            printf("?%s unknown command\n\n", ids);
        }
        fflush(stdout);
        fflush(stderr);
    }
    printf("%d %d %d\n",passed, black_stones_left, white_stones_left);
}
```

함수 이름: engine\_go&#x20;

입력:

* filename (char \*): 저장된 게임 기보 파일 이름
* weightfile (char \*): 저장된 가중치 파일 이름
* mcts\_iters (int): MCTS 알고리즘의 반복 횟수
* secs (float): 시간 제한 (초)
* temp (float): MCTS 알고리즘의 탐색 정도를 제어하는 온도 매개 변수
* cpuct (float): UCB 점수를 계산할 때 사용되는 매개 변수
* anon (int): 게임에서 익명 플레이어를 허용하는지 여부
* resign (int): 포기 기능을 사용하는지 여부

동작:

* 입력된 매개변수를 바탕으로 Go 엔진을 초기화하고, 메인 루프를 시작합니다.
* 루프에서는 소켓으로부터 명령을 읽어 듣고, 해당 명령을 처리합니다.
* 명령어 처리 중 일부 명령은 다른 함수를 호출합니다.
* protocol\_version 명령어는 아무것도 하지 않습니다.
* time\_settings 명령어는 메인 시간, 초읽기 시간, 초읽기 돌 갯수를 읽어들입니다.
* time\_left 명령어는 각 플레이어의 남은 시간을 읽어들입니다.
* version 명령어는 엔진의 버전 정보를 출력합니다.
* known\_command 명령어는 입력된 명령어가 엔진에서 지원되는지 확인합니다.
* list\_commands 명령어는 엔진에서 지원하는 모든 명령어를 출력합니다.
* quit 명령어는 프로그램을 종료합니다.
* boardsize 명령어는 바둑판의 크기를 설정하고, 바둑판을 초기화합니다.
* fixed\_handicap 명령어는 고정 수초를 설정합니다.
* clear\_board 명령어는 바둑판을 초기화합니다.
* komi 명령어는 코미 값을 설정합니다.
* showboard 명령어는 현재 바둑판의 상태를 출력합니다.
* 명령어 처리가 끝난 후에는 move\_mcts와 ponder 함수를 호출하여 다음 수를 계산합니다.
* move\_mcts 함수는 MCTS 알고리즘을 사용하여 최적의 수를 선택하고 바둑판에 적용합니다.
* ponder 함수는 MCTS 알고리즘을 시뮬레이션하지만, 실제로 수를 놓지는 않습니다.
* 디버그 모드에서는 표준 출력 및 오류 스트림에 디버그 정보를 출력합니다.

설명:

* 이 코드는 AI 기반의 바둑 엔진을 구동하는 메인 루프를 구현하는 것으로 보입니다.&#x20;
* 루프는 네트워크를 통해 다른 컴퓨터에서 실행 중인 바둑 프로그램으로부터 명령을 수신하고 해당 명령에 따라 응답합니다.&#x20;
* 루프는 입력에서 명령을 읽은 다음 명령에 따라 다양한 도우미 함수를 호출합니다.
* MCTS 알고리즘을 사용하여 최적의 다음 수를 선택합니다.&#x20;
* 시간 제한에 도달하거나 게임이 종료될 때까지 이 과정을 반복합니다.&#x20;
* anon 매개 변수가 1로 설정되면, 익명 플레이어가 게임에 참여할 수 있습니다.&#x20;
* resign 매개 변수가 1로 설정되면, 엔진은 일정 수준 이상 뒤지면 자동으로 게임을 포기합니다.



## test\_go

```c
void test_go(char *cfg, char *weights, int multi)
{
    int i;
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(time(0));
    float *board = calloc(19*19*3, sizeof(float));
    flip_board(board);
    float *move = calloc(19*19+1, sizeof(float));
    int color = 1;
    while(1){
        float result = predict_move2(net, board, move, multi);
        printf("%.2f%% Win Chance\n", (result+1)/2*100);

        int indexes[nind];
        int row, col;
        top_k(move, 19*19+1, nind, indexes);
        print_board(stderr, board, color, indexes);
        for(i = 0; i < nind; ++i){
            int index = indexes[i];
            row = index / 19;
            col = index % 19;
            if(row == 19){
                printf("%d: Pass, %.2f%%\n", i+1, move[index]*100);
            } else {
                printf("%d: %c %d, %.2f%%\n", i+1, col + 'A' + 1*(col > 7 && noi), (inverted)?19 - row : row+1, move[index]*100);
            }
        }
        //if(color == 1) printf("\u25EF Enter move: ");
        //else printf("\u25C9 Enter move: ");
        if(color == 1) printf("X Enter move: ");
        else printf("O Enter move: ");

        char c;
        char *line = fgetl(stdin);
        int picked = 1;
        int dnum = sscanf(line, "%d", &picked);
        int cnum = sscanf(line, "%c", &c);
        if (strlen(line) == 0 || dnum) {
            --picked;
            if (picked < nind){
                int index = indexes[picked];
                row = index / 19;
                col = index % 19;
                if(row < 19){
                    move_go(board, 1, row, col);
                }
            }
        } else if (cnum){
            if (c <= 'T' && c >= 'A'){
                int num = sscanf(line, "%c %d", &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 2) move_go(board, 1, row, col);
            } else if (c == 'p') {
                // Pass
            } else if(c=='b' || c == 'w'){
                char g;
                int num = sscanf(line, "%c %c %d", &g, &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 3) {
                    int mc = (g == 'b') ? 1 : -1;
                    if (mc == color) {
                        board[row*19 + col] = 1;
                    } else {
                        board[19*19 + row*19 + col] = 1;
                    }
                }
            } else if(c == 'c'){
                char g;
                int num = sscanf(line, "%c %c %d", &g, &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 3) {
                    board[row*19 + col] = 0;
                    board[19*19 + row*19 + col] = 0;
                }
            }
        }
        free(line);
        flip_board(board);
        color = -color;
    }
}
```

함수 이름: test\_go&#x20;

입력:

* char \*cfg: 모델 구성 파일 경로
* char \*weights: 모델 가중치 파일 경로
* int multi: 예측 시 사용할 스레드 수

동작:&#x20;

* 주어진 모델을 사용하여 사용자와 상호작용하면서 가상의 바둑 게임을 진행하고, 사용자로부터 바둑 돌을 입력받아 게임을 진행합니다.&#x20;
* 각 상황에서 모델을 사용하여 예측을 수행하고, 가능한 수를 출력하여 사용자가 선택할 수 있도록 합니다.

설명:&#x20;

* 이 함수는 주어진 모델 파일을 사용하여 바둑 게임을 시뮬레이션하고, 사용자와 상호작용하면서 게임을 진행합니다.&#x20;
* 바둑판은 19 x 19 크기로 가정하며, 모델은 현재 바둑판의 상태를 입력으로 받아서 다음 돌을 어디에 놓을지 예측합니다.
* 게임 시작시, 모델은 초기 바둑판 상태를 입력으로 받습니다.&#x20;
* 게임은 사용자와 모델이 번갈아가며 돌을 놓으면서 진행됩니다.&#x20;
* 각 턴마다, 현재 바둑판 상태를 입력으로 모델을 호출하여 다음 수의 확률 분포를 예측합니다.&#x20;
* 모델은 top-k 알고리즘을 사용하여 확률이 가장 높은 k개의 수를 출력하고, 각 수의 좌표와 확률을 표시합니다.&#x20;
* 사용자는 출력된 수 중에서 선택하여 돌을 놓을 수 있습니다.
* 사용자의 입력은 표준 입력을 통해 받습니다.&#x20;
* 사용자는 수의 좌표를 A1부터 T19까지 알파벳과 숫자로 입력하거나, 놓을 돌의 색깔(b 또는 w)과 좌표를 입력하여 바둑판에 돌을 놓을 수 있습니다.&#x20;
* 또한 'p'를 입력하여 패스할 수도 있으며, 'c'와 좌표를 입력하여 바둑판의 특정 위치에 놓인 돌을 제거할 수도 있습니다.
* 이 함수는 무한 루프를 돌며 게임을 진행하며, 사용자가 'q'를 입력하기 전까지 종료되지 않습니다.



## score\_game

```c
float score_game(float *board)
{
    int i;
    FILE *f = fopen("game.txt", "w");
    int count = print_game(board, f);
    fprintf(f, "final_score\n");
    fclose(f);
    FILE *p = popen("./gnugo --mode gtp < game.txt", "r");
    for(i = 0; i < count; ++i){
        free(fgetl(p));
        free(fgetl(p));
    }
    char *l = 0;
    float score = 0;
    char player = 0;
    while((l = fgetl(p))){
        fprintf(stderr, "%s  \t", l);
        int n = sscanf(l, "= %c+%f", &player, &score);
        free(l);
        if (n == 2) break;
    }
    if(player == 'W') score = -score;
    pclose(p);
    return score;
}
```

함수 이름: score\_game

입력:&#x20;

* float형 포인터 변수 board

동작:

1. "game.txt" 파일을 쓰기 모드로 열고 파일 포인터 f에 저장한다.
2. print\_game 함수를 이용해 board 상태를 파일 f에 출력하고, 출력한 횟수를 count 변수에 저장한다.
3. "final\_score" 문자열을 파일 f에 출력한다.
4. 파일 포인터 f를 닫는다.
5. "game.txt" 파일을 읽기 모드로 열고 popen 함수를 이용해 "gnugo --mode gtp" 명령어를 실행시킨다.
6. count 변수만큼 반복하면서 popen 함수로부터 라인을 읽어와서 free 함수를 이용해 메모리를 해제한다.
7. popen 함수로부터 라인을 읽어온 후, 문자열 l과 score 변수를 초기화하고, player 변수에 따라 score 변수의 부호를 결정한다.
8. popen 함수로부터 읽어온 라인이 "= W+score" 형태이면, score 변수의 부호를 바꾼다.
9. popen 함수로부터 읽어온 라인이 "= B+score" 형태이면, score 변수의 부호를 바꾸지 않는다.
10. popen 함수로부터 읽어온 라인이 "= score" 형태이면, player 변수와 상관없이 score 변수의 값을 저장한다.
11. popen 함수로부터 읽어온 라인이 "= ?" 형태이면, 해당 라인을 무시한다.
12. popen 함수로부터 읽어온 라인이 없을 때까지 7\~11 과정을 반복한다.
13. popen 함수로부터 읽어온 결과를 처리한 후, popen 함수를 닫고 score 값을 반환한다.

설명:&#x20;

* 이 함수는 주어진 바둑판(board) 상태에 대해 gnugo 프로그램을 이용하여 승패를 판단하고, 이를 점수(score)로 반환하는 함수이다.&#x20;
* 함수는 파일을 이용하여 gnugo 프로그램과 통신하며, popen 함수를 이용하여 gnugo 프로그램을 실행시키고 결과를 받아온다.&#x20;
* 결과는 문자열 형태로 받아오며, 문자열을 분석하여 승패를 결정하고 score 값을 반환한다.



## self\_go

```c
void self_go(char *filename, char *weightfile, char *f2, char *w2, int multi)
{
    mcts_tree *tree1 = 0;
    mcts_tree *tree2 = 0;
    network *net = load_network(filename, weightfile, 0);
    //set_batch_network(net, 1);

    network *net2;
    if (f2) {
        net2 = parse_network_cfg(f2);
        if(w2){
            load_weights(net2, w2);
        }
    } else {
        net2 = calloc(1, sizeof(network));
        *net2 = *net;
    }
    srand(time(0));
    char boards[600][93];
    int count = 0;
    //set_batch_network(net, 1);
    //set_batch_network(net2, 1);
    float *board = calloc(19*19*3, sizeof(float));
    flip_board(board);
    float *one = calloc(19*19*3, sizeof(float));
    float *two = calloc(19*19*3, sizeof(float));
    int done = 0;
    int player = 1;
    int p1 = 0;
    int p2 = 0;
    int total = 0;
    float temp = .1;
    int mcts_iters = 500;
    float cpuct = 5;
    while(1){
        if (done){
            tree1 = move_mcts(tree1, -1);
            tree2 = move_mcts(tree2, -1);
            float score = score_game(board);
            if((score > 0) == (total%2==0)) ++p1;
            else ++p2;
            ++total;
            fprintf(stderr, "Total: %d, Player 1: %f, Player 2: %f\n", total, (float)p1/total, (float)p2/total);
            sleep(1);
            /*
               int i = (score > 0)? 0 : 1;
               int j;
               for(; i < count; i += 2){
               for(j = 0; j < 93; ++j){
               printf("%c", boards[i][j]);
               }
               printf("\n");
               }
             */
            memset(board, 0, 3*19*19*sizeof(float));
            flip_board(board);
            player = 1;
            done = 0;
            count = 0;
            fflush(stdout);
            fflush(stderr);
        }
        //print_board(stderr, board, 1, 0);
        //sleep(1);

        if ((total%2==0) == (player==1)){
            //mcts_iters = 4500;   
            cpuct = 5;
        } else {
            //mcts_iters = 500;
            cpuct = 1;
        }
        network *use = ((total%2==0) == (player==1)) ? net : net2;
        mcts_tree *t = ((total%2==0) == (player==1)) ? tree1 : tree2;
        t = run_mcts(t, use, board, two, player, mcts_iters, cpuct, 0);
        move m = pick_move(t, temp, player);
        if(((total%2==0) == (player==1))) tree1 = t;
        else tree2 = t;

        tree1 = move_mcts(tree1, m.row*19 + m.col);
        tree2 = move_mcts(tree2, m.row*19 + m.col);

        if(m.row == 19){
            done = 1;
            continue;
        }
        int row = m.row;
        int col = m.col;

        float *swap = two;
        two = one;
        one = swap;

        if(player < 0) flip_board(board);
        boards[count][0] = row;
        boards[count][1] = col;
        board_to_string(boards[count] + 2, board);
        if(player < 0) flip_board(board);
        ++count;

        move_go(board, player, row, col);
        copy_cpu(19*19*3, board, 1, one, 1);

        player = -player;
    }
}
```

함수 이름: self\_go&#x20;

입력:

* filename (문자열 포인터): 첫 번째 신경망 파일 경로
* weightfile (문자열 포인터): 첫 번째 신경망 가중치 파일 경로
* f2 (문자열 포인터): 두 번째 신경망 설정 파일 경로
* w2 (문자열 포인터): 두 번째 신경망 가중치 파일 경로
* multi (정수): 멀티플레이어 모드 여부

동작:

* AlphaGo와 같은 바둑 AI를 만드는데 사용되는 함수이다.
* 입력으로 받은 두 개의 신경망을 사용하여 바둑 게임을 진행한다.
* self-play 방식으로 학습을 하기 위해 자신과 대결하는 모드인 "멀티플레이어 모드"를 지원한다.
* 첫 번째 신경망이 흑돌을, 두 번째 신경망이 백돌을 대표한다.
* MCTS(Monte Carlo Tree Search) 알고리즘을 사용하여 다음 수를 예측하고 수를 놓는다.
* 게임이 종료되면 게임 결과에 따라 승패를 계산하고, 결과를 출력한다.

설명:

* 이 함수는 AlphaGo와 같은 바둑 AI를 구현하는데 사용되는 함수로, 입력으로 두 개의 신경망을 받는다.
* 첫 번째 신경망은 흑돌을 대표하고, 두 번째 신경망은 백돌을 대표한다.
* 만약 멀티플레이어 모드인 경우, 두 번째 신경망은 사용되지 않는다.
* 게임을 시작하기 위해 초기 게임 상태를 설정하고, MCTS 알고리즘을 이용하여 각 턴에서 다음 수를 예측한다.
* 이후 예측된 수를 놓고 게임을 진행한다.
* 게임이 끝나면 게임 결과에 따라 승패를 계산하고, 결과를 출력한다.



## run\_go

```c
void run_go(int argc, char **argv)
{
    //boards_go();
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
    int clear = find_arg(argc, argv, "-clear");

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *c2 = (argc > 5) ? argv[5] : 0;
    char *w2 = (argc > 6) ? argv[6] : 0;
    int multi = find_arg(argc, argv, "-multi");
    int anon = find_arg(argc, argv, "-anon");
    int iters = find_int_arg(argc, argv, "-iters", 500);
    int resign = find_int_arg(argc, argv, "-resign", 175);
    float cpuct = find_float_arg(argc, argv, "-cpuct", 5);
    float temp = find_float_arg(argc, argv, "-temp", .1);
    float time = find_float_arg(argc, argv, "-time", 0);
    if(0==strcmp(argv[2], "train")) train_go(cfg, weights, c2, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) valid_go(cfg, weights, multi, c2);
    else if(0==strcmp(argv[2], "self")) self_go(cfg, weights, c2, w2, multi);
    else if(0==strcmp(argv[2], "test")) test_go(cfg, weights, multi);
    else if(0==strcmp(argv[2], "engine")) engine_go(cfg, weights, iters, time, temp, cpuct, anon, resign);
}
```

함수 이름: run\_go

입력:

* int argc: 실행 시 전달된 인수(argument)의 개수
* char \*\*argv: 실행 시 전달된 인수의 배열(array)

동작:

* 주어진 인수를 기반으로 강화학습 프로그램을 실행시키는 함수
* 실행 시 전달된 인수에 따라서 다양한 서브루틴 함수를 호출하여 동작한다

설명:

* gpu\_list: GPU 리스트를 포함하는 문자열
* gpus: GPU 리스트를 포함하는 정수형 배열
* gpu: 기본 GPU 인덱스 값
* ngpus: 사용 가능한 GPU 개수
* clear: 기존 가중치를 삭제할 지 여부
* cfg: YOLOv3-608.cfg와 같은 모델 구성 파일의 경로
* weights: 사전 훈련된 가중치 파일의 경로 (선택 사항)
* c2: fine-tuning에 사용할 사전 훈련된 가중치 파일의 경로 (선택 사항)
* w2: fine-tuning에 사용할 사전 훈련된 가중치 파일의 경로 (선택 사항)
* multi: 멀티 GPU를 사용할 지 여부
* anon: 익명 플레이를 할 지 여부
* iters: MCTS 트리에서 탐색할 횟수
* resign: 타임 아웃 전에 착수를 포기할 시점
* cpuct: 탐사 정도를 조절하는 상수
* temp: 탐색 도중에 사용되는 softmax 온도 매개변수
* time: 최대 탐색 시간
* train\_go(): 주어진 경로에서 강화학습 알고리즘을 사용하여 모델을 훈련시키는 서브루틴 함수
* valid\_go(): 주어진 경로에서 강화학습 알고리즘을 사용하여 모델을 검증하는 서브루틴 함수
* self\_go(): 주어진 경로에서 강화학습 알고리즘을 사용하여 모델을 평가하는 서브루틴 함수
* test\_go(): 주어진 경로에서 강화학습 알고리즘을 사용하여 모델을 테스트하는 서브루틴 함수
* engine\_go(): 주어진 경로에서 MCTS 알고리즘을 사용하여 게임 엔진을 실행시키는 서브루틴 함수

