#include <stdlib.h>
#include <string.h>
#include "list.h"


/*
** 创建双链表
*/
list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;  // 双链表长度
	l->front = 0; // 链表第一个节点赋0
	l->back = 0;  // 链表最后一个节点赋0
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/


/*
** 从双链表尾部删除元素，并返回
*/
void *list_pop(list *l){
    if(!l->back) return 0; // 链表为空时，返回0
    node *b = l->back; // 将链表最后一个节点取出
    void *val = b->val; // 链表最后一个节点的值
    l->back = b->prev;  // 将链表倒数第二个节点赋给链表尾节点
    if(l->back) l->back->next = 0; // 当链表不为空时，将最后一个节点的下一个指针赋0
    free(b);
    --l->size;
    
    return val;
}


/*
**	将val指针插入list结构体l中，这里相当于是用C实现了C++中的list的元素插入功能
**	流程: list中并不直接含有void*类型指针，但list中含有的node指针含有void*类型指针，
**		  因此，将首先创建一个node指针new，而后将val传给new，最后再把new插入list指针l中
**	说明: void*指针可以接收不同数据类型的指针，因此第二个参数具体是什么类型的指针得视情况而定
**	调用: 该函数在众多地方调用，很多从文件中读取信息存入到list变量中，都会调用此函数，
**		  注意此函数类似C++的insert()插入方式；而在option_list.h中的opion_insert()函数，
**		  有点类似C++ map数据结构中的按值插入方式，比如map[key]=value，两个函数操作对象都是list变量，
**		  只是操作方式略有不同。
*/
void list_insert(list *l, void *val)
{
    // 定义一个node指针并动态分配内存
	node *new = malloc(sizeof(node));

    // 将输入的val指针赋值给new中的val元素，注意，这是指针复制，共享地址，二者都是void*类型指针
	new->val = val;
	new->next = 0;


    /* 链表内部更新 */
	// 下面的链表嵌套主要注意一下
	// 如果l的back元素为空指针，说明l到目前为止，还没有存入数据（只是在函数外动态分配了内存，并没有赋有效值），
	// 这样，令l的front为new（此后front将不会再变，除非删除），显然此时new的前面没有node，因此new->prev=0
	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		// 如果已经插入了第一个元素，那么往下嵌套，注意这里操作的是指针，互有影响
		// 新插入的node赋给back的下一个node next，
		// 同时对于新的node new来说，其前一个node为l->back
		// 一定注意要相互更新（链表上的数据位置关系都是相对的）
		l->back->next = new;
		new->prev = l->back;
	}

    /* 链表信息更新 */
	// 更新back的值
	// 不管前面进行了什么操作，每次插入new，都必须更新l的back为当前new，因为链表结构，
	// 新插入的元素永远在最后，此处back就是链表中的最后一个元素，front是链表中的第一个元素，
	// 链表中的第一个元素在第一次插入元素之后将不会再变（除非删除）
	l->back = new;

    // l中存储的元素个数加1
	++l->size;
}

/*
**	释放节点内存，注意节点node是结构体指针，其中含有自引用，需要循环释放内存
**	输入: n	需要释放的node指针，其内存是动态分配的
**	注意: 不管什么程序，堆内存的释放都是一个重要的问题，darknet中的内存释放，值得学习！
**		  node结构体含有自引用，输入的虽然是一个节点，但实则是一个包含众多节点的链表结构，
**		  释放内存时，一定要循环释放，且格外注意内存释放的顺序
*/
void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

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
