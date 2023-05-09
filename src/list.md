# list

```c
// darknet.h

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

```

연결 리스트(list)의 구조체를 정의하는 코드입니다.

* 구조체 이름: list
* 구조체 멤버:
  * size: 리스트에 저장된 노드의 수
  * front: 리스트의 첫 번째 노드를 가리키는 포인터
  * back: 리스트의 마지막 노드를 가리키는 포인터

연결 리스트는 데이터의 삽입, 삭제, 검색 등의 연산을 빠르게 처리할 수 있는 자료구조입니다. Darknet 라이브러리에서는 이러한 연결 리스트를 다양한 용도로 사용합니다. 예를 들어, 네트워크를 구성하는 레이어들을 연결 리스트로 관리하거나, 학습 데이터를 미니배치로 분할한 후 각각의 미니배치를 연결 리스트로 저장하는 등의 용도로 사용됩니다.



## make\_list

```c
list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}
```

함수 이름: make\_list\
입력:&#x20;

* 없음\


동작:&#x20;

* 빈 리스트를 생성하고, 해당 리스트를 가리키는 포인터를 반환함\


설명:&#x20;

* 리스트 구조체를 동적으로 할당하고, size, front, back 멤버 변수를 초기화하여 빈 리스트를 생성한다.&#x20;
* 그리고 해당 리스트를 가리키는 포인터를 반환한다.



## list\_pop

```c
void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;

    return val;
}
```

함수 이름: list\_pop

입력:&#x20;

* list 구조체 포인터 변수 l

동작:&#x20;

* 리스트 l에서 뒤쪽에 있는 노드를 제거하고 해당 노드에 저장되어 있던 값을 반환한다.

설명:&#x20;

* 리스트 l의 back 포인터가 NULL이면, 즉 리스트에 노드가 없으면 NULL을 반환하고 함수를 종료한다.&#x20;
* 그렇지 않은 경우에는 리스트의 back 포인터가 가리키는 노드를 변수 b에 저장하고, b의 val 필드에 저장되어 있는 값을 변수 val에 저장한다.&#x20;
* 그 다음, 리스트의 back 포인터를 b의 prev 필드가 가리키는 노드로 변경하고, 변경된 back 포인터가 NULL이 아니면 해당 노드의 next 필드를 0으로 설정한다.&#x20;
* 그리고 b 노드를 해제하고 리스트의 size를 1 감소시킨다. 마지막으로 val 변수에 저장되어 있는 값을 반환한다.



## list\_insert

```c
void list_insert(list *l, void *val)
{
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;

	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	++l->size;
}
```

함수 이름: list\_insert

입력:

* list \*l: 삽입할 리스트의 포인터
* void \*val: 리스트에 삽입할 값의 포인터

동작:

* val 포인터를 가지는 새로운 노드를 생성하고 리스트의 뒤쪽에 삽입한다.
* 리스트가 비어있을 경우, 새로운 노드를 리스트의 front로 지정한다.
* 리스트가 비어있지 않을 경우, 새로운 노드를 리스트의 back 다음에 연결한다.
* 리스트의 size를 1 증가시킨다.

설명:&#x20;

* 주어진 리스트 l의 맨 뒤쪽에 val 포인터를 가지는 새로운 노드를 생성하고 삽입하는 함수이다.&#x20;
* 리스트가 비어있을 경우, 새로운 노드는 리스트의 front가 되며, 리스트가 비어있지 않을 경우, 새로운 노드는 리스트의 back 다음에 연결된다.&#x20;
* 새로운 노드가 추가되면 리스트의 size를 1 증가시킨다.



## free\_node

```c
void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}
```

함수 이름: free\_node

입력:&#x20;

* n: node 포인터&#x20;

동작:&#x20;

* n을 시작으로 연결된 노드들을 메모리에서 해제합니다.

설명:&#x20;

* 연결 리스트의 노드들을 메모리에서 해제하는 함수입니다.&#x20;
* 이 함수는 시작 노드의 포인터를 입력으로 받으며, 입력된 노드부터 시작하여 다음 노드를 가리키는 포인터를 계속해서 따라가며 각 노드를 메모리에서 해제합니다.&#x20;
* 다음 노드를 가리키는 포인터가 NULL이 될 때까지 이 과정을 반복합니다.



## free\_list

```c
void free_list(list *l)
{
	free_node(l->front);
	free(l);
}
```

함수 이름: free\_list

입력:&#x20;

* l: list 포인터&#x20;

동작:&#x20;

* l이 가리키는 리스트의 모든 노드를 해제하고, 리스트 자체도 해제함.

설명:&#x20;

* 해당 함수는 동적으로 할당된 list 구조체와 그 안에 있는 모든 node 구조체를 해제하는 함수이다.&#x20;
* l이 가리키는 리스트의 맨 앞 노드인 front부터 시작하여 모든 노드의 val 멤버에 할당된 메모리를 먼저 해제하고, 그 다음에 각 노드의 메모리를 해제한다.&#x20;
* 마지막으로, 리스트 자체를 해제한다.



## free\_list\_contents

```c
void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}
```

함수 이름: free\_list\_contents

입력:

* l: 해제할 list의 포인터

동작:

* list에 있는 모든 노드의 val 멤버를 free() 함수를 사용하여 해제한다.

설명:

* list 자료구조는 node의 포인터와 size 멤버를 가지고 있다.
* 각 node는 val 멤버를 가지고 있다.
* 이 함수는 list의 모든 노드를 탐색하면서 각 노드의 val 멤버를 해제한다.



## list\_to\_array

```c
void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
```

함수 이름: list\_to\_array

입력:&#x20;

* l: list 포인터&#x20;

동작:&#x20;

* 연결 리스트 l의 각 노드의 값을 배열에 저장하고, 해당 배열을 반환한다.

설명:&#x20;

* 함수는 동적으로 할당된 배열을 반환하므로 메모리 누수를 방지하기 위해 반드시 해당 배열을 free 해주어야 한다.&#x20;
* 함수는 먼저 연결 리스트 l의 크기에 해당하는 void 포인터 배열 a를 할당하고, 리스트의 모든 노드를 순회하며 각 노드의 값을 배열 a에 저장한다. 이후 배열 a를 반환한다.

