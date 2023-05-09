# lstm\_layer

### LSTM Layer란?

LSTM은 Long Short Term Memory networks의 약자입니다. RNN과 같이 자연어처리, 음성처리 등 Sequential 데이터를 처리하기 위해 많이 사용되는 layer입니다.

기존의 RNN은 학습하면서 점점 과거 정보를 잊어버리는(Gradient Vanishing) 문제가 발생하고 이러한 장기적인 의존성 문제를 해결하기 위해서 설계된 layer입니다.

LSTM의 핵심적인 요소는 `cell state` 입니다. LSTM의 `cell state`는 공장의 컨베이어 벨트와 같으며 이러한 컨베이어 벨트에 `gate`를 이용하여 값을 공급하여 정보를 추가하거나 제거해 갑니다.\
`gate`는 총 3가지로 이루어져 있습니다.

* forget gate는 정보를 얼마나 잊을 것인지에 대해서 연산하는 gate입니다. sigmoid를 통해 0 \~ 1 사이의 값이 출력되는데 1에 가까우면 기억하라는 의미를 포함하고 0에 가까우면 잊으라는 의미를 포함합니다.
* input gate는 새로운 정보를 공급하는 연산을 하는 gate입니다. sigmoid를 통해 어떤 입력값을 업데이트해야 할지 결정하고 tanh는 새로운 입력값을 만듭니다. 두 개의 값을 합쳐서 새로운 값이 기존 값에 영향을 주는 값을 만들어 냅니다.
* output gate는 어떤 출력값을 다음 state에 보내줄지 결정하는 gate입니다. sigmoid를 통해 어떤 값을 출력해야 할지 결정하고 tanh는 업데이트 된 cell state의 영향을 말해줍니다.

결론적으로 순서는 forget gate로 잊어야할 부분을 잊고 input gate로 새로운 값을 추가하며 cell state를 업데이트 하고 output gate를 통해 최종적으로 출력합니다.

***

### increment\_layer

```c
static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;
}
```

함수 이름: increment\_layer

입력:

* layer 포인터 l: 값이 증가될 레이어 객체 포인터
* int steps: 증가할 스텝 수

동작:

* l 객체의 output, delta, x, x\_norm 포인터가 가리키는 값에 steps \* l->outputs \* l->batch를 더해 값을 증가시킴

설명:

* 이 함수는 뉴럴 네트워크에서 역전파 알고리즘을 수행하기 위한 LSTM 레이어에서 사용되는 함수입니다.
* LSTM 레이어에서는 시퀀스의 각 타임스텝에 대해 forward와 backward 패스를 수행해야 합니다. increment\_layer 함수는 backward 패스를 수행할 때 이전 시퀀스 타임스텝에 대한 출력, 델타, 입력 등의 포인터를 증가시키기 위해 사용됩니다.
* l 객체의 output, delta, x, x\_norm 포인터는 모두 이전 타임스텝에 대한 값을 가리키고 있습니다. 따라서 steps \* l->outputs \* l->batch 만큼 값을 더해주면 이전 타임스텝의 값에 대한 포인터를 증가시키는 효과를 얻을 수 있습니다.
* 이 함수는 static 키워드를 가지고 있으므로 같은 소스 파일 내에서만 사용 가능합니다.



### forward\_lstm\_layer

```c
void forward_lstm_layer(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);
    if (state.train) {
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = l.h_cpu;
        forward_connected_layer(wf, s);							
        forward_connected_layer(wi, s);							
        forward_connected_layer(wg, s);							
        forward_connected_layer(wo, s);							

        s.input = state.input;
        forward_connected_layer(uf, s);							
        forward_connected_layer(ui, s);							
        forward_connected_layer(ug, s);							
        forward_connected_layer(uo, s);							

        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);

        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);

        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);

        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);			
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);		

        copy_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.c_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.h_cpu, 1);			
        activate_array(l.h_cpu, l.outputs*l.batch, TANH);		
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.h_cpu, 1);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.cell_cpu, 1);		
        copy_cpu(l.outputs*l.batch, l.h_cpu, 1, l.output, 1);

        state.input += l.inputs*l.batch;
        l.output    += l.outputs*l.batch;
        l.cell_cpu      += l.outputs*l.batch;

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}
```

함수 이름: forward\_lstm\_layer

입력:

* layer l: LSTM 레이어의 구조와 파라미터를 저장하는 구조체
* network state: 입력 데이터와 그 외의 네트워크 정보를 저장하는 구조체

동작:

* LSTM 레이어의 forward propagation을 수행하는 함수
* 현재 레이어의 파라미터와 이전 시점의 출력 값을 이용하여 현재 시점의 출력 값을 계산
* 입력 데이터는 state.input에 저장되어 있으며, l.steps 번 만큼 forward propagation을 반복하여 출력 값을 계산
* 각 연산은 내부적으로 connected layer의 forward 연산을 이용하여 수행됨

설명:

* 입력으로 주어진 LSTM 레이어의 구조와 파라미터를 이용하여 forward propagation을 수행하고, 현재 시점의 출력 값을 계산하여 l.output에 저장함
* 이전 시점의 출력 값과 현재 시점의 입력 데이터를 이용하여 현재 시점의 출력 값을 계산함
* 각 게이트(gate)의 출력 값과 입력 데이터를 이용하여 candidate 값과 forget 값을 계산하고, cell 상태를 업데이트하여 현재 시점의 출력 값을 계산함
* forward propagation 도중에는 backpropagation을 위한 미분 값(delta)들도 계산됨
* state.train이 true인 경우에는 현재 시점의 출력 값에 대한 손실 함수의 미분 값(l.delta)도 계산됨



### backward\_lstm\_layer

```c
void backward_lstm_layer(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input += l.inputs*l.batch*(l.steps - 1);
    if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);

    l.output += l.outputs*l.batch*(l.steps - 1);
    l.cell_cpu += l.outputs*l.batch*(l.steps - 1);
    l.delta += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_cpu(l.outputs*l.batch, l.cell_cpu - l.outputs*l.batch, 1, l.prev_cell_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        if (i != 0) copy_cpu(l.outputs*l.batch, l.output - l.outputs*l.batch, 1, l.prev_state_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.output, 1, l.h_cpu, 1);

        l.dh_cpu = (i == 0) ? 0 : l.delta - l.outputs*l.batch;

        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);			

        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);			

        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);			

        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);			

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);			
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);			
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);		

        copy_cpu(l.outputs*l.batch, l.delta, 1, l.temp3_cpu, 1);		

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);			
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);			

        copy_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.temp2_cpu, 1);			

        gradient_array(l.temp_cpu, l.outputs*l.batch, TANH, l.temp2_cpu);
        axpy_cpu(l.outputs*l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);		

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);			
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);			
        mul_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);		
        gradient_array(l.o_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wo.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;															
        backward_connected_layer(wo, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uo.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(uo, s);									

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);			
        mul_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);				
        gradient_array(l.g_cpu, l.outputs*l.batch, TANH, l.temp_cpu);		
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wg.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;														
        backward_connected_layer(wg, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ug.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(ug, s);																

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);			
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);				
        gradient_array(l.i_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wi.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wi, s);						

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ui.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(ui, s);									

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.f_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wf.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wf, s);						

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uf.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(uf, s);									

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);			
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.temp_cpu, 1);				
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, l.dc_cpu, 1);				

        state.input -= l.inputs*l.batch;
        if (state.delta) state.delta -= l.inputs*l.batch;
        l.output -= l.outputs*l.batch;
        l.cell_cpu -= l.outputs*l.batch;
        l.delta -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}
```

함수 이름: backward\_lstm\_layer

입력:

* l: LSTM 레이어 매개변수를 포함하는 레이어 객체
* state: 현재 네트워크 상태를 포함하는 네트워크 상태 객체

동작:

* 현재 시퀀스에서 역방향 LSTM 레이어의 기울기를 계산한다.
* s라는 새로운 네트워크 객체를 모든 값이 0으로 초기화하고, s의 train 플래그를 state.train 값으로 설정하며, 정수 i를 초기화한다.
* 전방 LSTM 레이어와 역방향 LSTM 레이어의 가중치 행렬(각각 wf, wi, wg, wo, uf, ui, ug, uo) 8개의 레이어 객체를 초기화한다.
* 입력과 델타 포인터를 현재 시퀀스의 마지막 타임 스텝을 가리키도록 업데이트한다.
* 출력, cell\_cpu, delta 포인터를 현재 시퀀스의 마지막 타임 스텝을 가리키도록 업데이트한다.
* 시퀀스를 역순으로 반복하며 각 시간 단계에서 다음을 수행한다:
  * l.cell\_cpu와 l.output의 값을 l.c\_cpu와 l.h\_cpu로 복사한다.
  * l.dh\_cpu 기울기를 계산한다.
  * l.f\_cpu, l.i\_cpu, l.g\_cpu, l.o\_cpu 배열의 값을 업데이트한다.
  * 로지스틱 함수와 하이퍼볼릭 탄젠트 함수를 l.f\_cpu, l.i\_cpu, l.g\_cpu, l.o\_cpu 배열에 적용한다.
  * l.delta를 l.temp3\_cpu에 복사하고, l.c\_cpu에 적용된 하이퍼볼릭 탄젠트 함수의 기울기를 l.temp2\_cpu로 계산한다.
  * l.dc\_cpu를 l.dc\_cpu와 l.temp2\_cpu의 합으로 계산하고, l.temp\_cpu에 l.c\_cpu의 값을 복사한 후 하이퍼볼릭 탄젠트 함수를 적용한다.
  * 결과 배열을 l.temp3\_cpu에 곱하여 l.temp2\_cpu를 얻는다.

설명:&#x20;

* LSTM 레이어의 역방향 함수는 역전파를 통해 기울기를 계산하고 최적화에 활용하는 함수이다.
* 이 함수는 현재 시퀀스에서 역방향 LSTM 레이어의 기울기를 계산하기 위해 사용된다.&#x20;
* 함수는 레이어 객체와 네트워크 상태 객체를 매개변수로 받으며, 시퀀스를 역순으로 반복하면서 각 시간 단계에서 필요한 계산을 수행한다.



### update\_lstm\_layer

```c
void update_lstm_layer(layer l, update_args a)
{
    update_connected_layer(*(l.wf), a);
    update_connected_layer(*(l.wi), a);
    update_connected_layer(*(l.wg), a);
    update_connected_layer(*(l.wo), a);
    update_connected_layer(*(l.uf), a);
    update_connected_layer(*(l.ui), a);
    update_connected_layer(*(l.ug), a);
    update_connected_layer(*(l.uo), a);
}
```

함수 이름: update\_lstm\_layer

입력:

* layer l: LSTM 레이어의 파라미터를 포함한 레이어 객체
* update\_args a: 학습률 등 업데이트 인자를 포함한 구조체

동작:&#x20;

* 이 함수는 LSTM 레이어의 파라미터인 8개의 가중치 행렬에 대해 update\_connected\_layer 함수를 호출하여 업데이트를 수행합니다.&#x20;
* update\_connected\_layer 함수는 입력된 연결층의 가중치를 업데이트하고자 할 때 사용되는 함수이며, a에 담긴 업데이트 인자를 바탕으로 각 가중치에 대한 gradient descent를 수행합니다.

설명:&#x20;

* update\_lstm\_layer 함수는 LSTM 레이어 객체 l과 업데이트 인자를 포함한 구조체 a를 입력받습니다.&#x20;
* 이 함수는 l.wf, l.wi, l.wg, l.wo, l.uf, l.ui, l.ug, l.uo에 각각 접근하여, update\_connected\_layer 함수를 호출하여 가중치를 업데이트합니다.&#x20;
* 이 함수는 gradient descent 알고리즘을 사용하여, 주어진 학습률과 업데이트 인자를 바탕으로 가중치를 업데이트합니다.&#x20;
* 이 과정은 학습을 진행하면서 반복적으로 수행되며, 가중치를 최적화하여 모델의 예측 성능을 향상시킵니다.

