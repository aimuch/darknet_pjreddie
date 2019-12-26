#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

/*
**  所有的激活函数类别(枚举类)
*/
typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

/** 
** 网络结构类型(枚举类型), 对应的整型值由CONVOLUTIONAL从0开始往下编号, 共24中网络类型(最后一个对应的整型值为23).
*/
typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK     // 表示未识别的网络层名称
} LAYER_TYPE;


/** 
** 损失函数(枚举类型)
*/
typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH, WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer{
    LAYER_TYPE type;        // 网络层的类型, 枚举类型, 取值比如DROPOUT,CONVOLUTIONAL,MAXPOOL分别表示dropout层, 卷积层, 最大池化层, 可参见LAYER_TYPE枚举类型的定义
    ACTIVATION activation;  // 激活函数类型
    COST_TYPE cost_type;    // 损失函数类型
    void (*forward)   (struct layer, struct network);       // 前向传播函数指针, 在具体初始化每个网络层的时候才会赋相应的函数指针
    void (*backward)  (struct layer, struct network);       // 反向传播函数指针, 在具体初始化每个网络层的时候才会赋相应的函数指针
    void (*update)    (struct layer, update_args);          
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);

    int batch_normalize;    // 是否进行BN, 如果进行BN, 则值为1
    int shortcut;
    int batch;              // 一个batch中含有的图片张数, 等于net.batch, 详细可以参考network.h中的注释, 一般在构建具体网络层时赋值(比如make_maxpool_layer()中)
    int forced;
    int flipped;

    int inputs;             // 一张输入图片所含的元素个数(一般在各网络层构建函数中赋值, 比如make_connected_layer()), 第一层的值等于l.h*l.w*l.c, 
                            // 之后的每一层都是由上一层的输出自动推算得到的(参见parse_network_cfg(), 在构建每一层后, 会更新params.inputs为上一层的l.outputs)
    
    int outputs;            // 该层对应一张输入图片的输出元素个数(一般在各网络层构建函数中赋值, 比如make_connected_layer())
                            // 对于一些网络, 可由输入图片的尺寸及相关参数计算出, 比如卷积层, 可以通过输入尺寸以及跨度、核大小计算出; 
                            // 对于另一些尺寸, 则需要通过网络配置文件指定, 如未指定, 取默认值1, 比如全连接层(见parse_connected()函数)
    
    int nweights;           // 网络参数数量
    int nbiases;            // 偏置数量
    int extra;

    int truths;             // 根据region_layer.c判断, 这个变量表示一张图片含有的真实值的个数, 对于检测模型来说, 一个真实的标签含有5个值, 
                            // 包括类型对应的编号以及定位矩形框用到的w,h,x,y四个参数, 且在darknet中, 固定每张图片最大处理30个矩形框(可查看max_boxes参数), 
                            // 因此, 在region_layer.c的make_region_layer()函数中, 赋值为30*5
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;

    int   * indexes;            // 维度为l.out_h * l.out_w * l.out_c * l.batch, 可知包含整个batch输入图片的输出, 一般在构建具体网络层时动态分配内存(比如make_maxpool_layer()中). 
                                // 目前仅发现其用在在最大池化层中. 该变量存储的是索引值, 并与当前层所有输出元素一一对应, 表示当前层每个输出元素的值是上一层输出中的哪一个元素值(存储的索引值是
                                // 在上一层所有输出元素(包含整个batch)中的索引), 因为对于最大池化层, 每一个输出元素的值实际是上一层输出(也即当前层输入)某个池化区域中的最大元素值, indexes就是记录
                                // 这些局部最大元素值在上一层所有输出元素中的总索引. 记录这些值有什么用吗？当然有, 用于反向传播过程计算上一层敏感度值, 详见backward_maxpool_layer()以及forward_maxpool_layer()函数. 

    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;

    float * rand;                // 这个参数目前只发现用在dropout层, 用于存储一些列的随机数, 这些随机数与dropout层的输入元素一一对应, 维度为l.batch*l.inputs(包含整个batch的), 在make_dropout_layer()函数中用calloc动态分配内存, 
                                // 并在前向传播函数forward_dropout_layer()函数中逐元素赋值. 里面存储的随机数满足0~1均匀分布, 干什么用呢？用于决定该输入元素的去留, 
                                // 我们知道dropout层就完成一个事：按照一定概率舍弃输入神经元(所谓舍弃就是置该输入的值为0), rand中存储的值就是如果小于l.probability, 则舍弃该输入神经元(详见：forward_dropout_layer()). 
                                // 为什么要保留这些随机数呢？和最大池化层中的l.indexes类似, 在反向传播函数backward_dropout_layer()中用来指示计算上一层的敏感度值, 因为dropout舍弃了一些输入, 
                                // 这些输入(dropout层的输入, 上一层的输出)对应的敏感度值可以置为0, 而那些没有舍弃的输入, 才有必要由当前dropout层反向传播过去. 

    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;              // 当前层所有偏置, 对于卷积层, 维度l.n, 每个卷积核有一个偏置; 对于全连接层, 维度等于单张输入图片对应的元素个数即outputs, 
                                // 一般在各网络构建函数中动态分配内存(比如make_connected_layer())
    
    float * bias_updates;        // 当前层所有偏置更新值, 对于卷积层, 维度l.n, 每个卷积核有一个偏置; 对于全连接层, 维度为outputs. 所谓权重系数更新值, 
                                // 就是梯度下降中与步长相乘的那项, 也即误差对偏置的导数, 一般在各网络构建函数中动态分配内存(比如make_connected_layer())

    float * scales;
    float * scale_updates;

    float * weights;             // 当前层所有权重系数(连接当前层和上一层的系数, 但记在当前层上), 对于卷积层, 维度为l.n*l.c*l.size*l.size, 
                                // 即卷积核个数乘以卷积核尺寸再乘以输入通道数(各个通道上的权重系数独立不一样); 
                                // 对于全连接层, 维度为单张图片输入与输出元素个数之积inputs*outputs, 一般在各网络构建函数中动态分配内存(比如make_connected_layer())
    
    float * weight_updates;      // 当前层所有权重系数更新值, 对于卷积层维度为l.n*l.c*l.size*l.size; 对于全连接层, 维度为单张图片输入与输出元素个数之积inputs*outputs, 
                                // 所谓权重系数更新值, 就是梯度下降中与步长相乘的那项, 也即误差对权重的导数, 一般在各网络构建函数中动态分配内存(比如make_connected_layer()

    float * delta;               // 存储每一层的敏感度图: 包含所有输出元素的敏感度值(整个batch所有图片). 所谓敏感度, 即误差函数关于当前层每个加权输入的导数值, 
                                // 关于敏感度图这个名称, 可以参考https://www.zybuluo.com/hanbingtao/note/485480. 
                                // 元素个数为l.batch * l.outputs(其中l.outputs = l.out_h * l.out_w * l.out_c), 
                                // 对于卷积神经网络, 在make_convolutional_layer()动态分配内存, 按行存储, 可视为l.batch行, l.outputs列, 
                                // 即batch中每一张图片, 对应l.delta中的一行, 而这一行, 又可以视作有l.out_c行, l.out_h*l.out_c列, 
                                // 其中每小行对应一张输入图片的一张输出特征图的敏感度. 一般在构建具体网络层时动态分配内存(比如make_maxpool_layer()中). 
    
    float * output;              // 存储该层所有的输出, 维度为l.out_h * l.out_w * l.out_c * l.batch, 可知包含整个batch输入图片的输出, 
                                // 一般在构建具体网络层时动态分配内存(比如make_maxpool_layer()中)
                                // 按行存储：每张图片按行铺排成一大行, 图片间再并成一行
    
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;             // softmax层用到的一个参数, 不过这个参数似乎并不常见, 很多用到softmax层的网络并没用使用这个参数, 目前仅发现darknet9000.cfg中使用了该参数, 
                                    // 如果未用到该参数, 其值为NULL, 如果用到了则会在parse_softmax()中赋值, 目前个人的初步猜测是利用该参数来组织标签数据, 以方便访问

    size_t workspace_size;          // net.workspace的元素个数, 为所有层中最大的l.out_h*l.out_w*l.size*l.size*l.c(在make_convolutional_layer()计算得到workspace_size的大小, 在parse_network_cfg()中动态分配内存, 此值对应未使用gpu时的情况)

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float * binary_input_gpu;
    float * binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;                  // 网络总层数(make_network()时赋值)
    int batch;              // parse_net_options()中赋值: 一个batch含有的图片张数, 下面还有个subdivisions参数, 此处的batch*subdivision才等于网络配置文件中指定的batch值
    size_t *seen;           // 目前已经读入的图片张数(网络已经处理的图片张数)(在make_network()中动态分配内存)
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;          // 存储网络所有的层, 在make_network()中动态分配内存
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;             // 一张输入图片的元素个数, 如果网络配置文件中未指定, 则默认等于net->h * net->w * net->c, 在parse_net_options()中赋值
    int outputs;            // 一张输入图片对应的输出元素个数, 对于一些网络, 可由输入图片的尺寸及相关参数计算出, 比如卷积层, 可以通过输入尺寸以及跨度、核大小计算出; 
                            // 对于另一些尺寸, 则需要通过网络配置文件指定, 如未指定, 取默认值1, 比如全连接层
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;        // 中间变量, 用来暂存某层网络的输入(包含一个batch的输入, 比如某层网络完成前向, 将其输出赋给该变量, 作为下一层的输入, 可以参看network.c中的forward_network()与backward_network()两个函数), 
                        // 当然, 也是网络接受最原始输入数据(即第一层网络接收的输入)的变量(比如在图像检测训练中, 最早在train_detector()->train_network()->get_next_batch()函数中赋值)
    
    float *truth;        // 中间变量, 与上面的input对应, 用来暂存input数据对应的标签数据(真实数据)
    
    float *delta;        // 中间变量, 用来暂存某层网络的敏感度图(反向传播处理当前层时, 用来存储上一层的敏感度图, 因为当前层会计算部分上一层的敏感度图, 可以参看network.c中的backward_network()函数), 
                        // net.delta并没有在创建网络之初就为其动态分配了内存, 而是等到反向传播时, 直接将其等于某一层的l.delta(l.delta是在创建每一层网络之初就动态为其分配了内存), 这才为net.delta分配了内存, 
                        // 如果没有令net.delta=l.delta, 则net.delta是未定义的(没有动态分配内存的)
    
    float *workspace;    // 整个网络的工作空间, 其元素个数为所有层中最大的l.workspace_size = l.out_h*l.out_w*l.size*l.size*l.c
                        // (在make_convolutional_layer()计算得到workspace_size的大小, 在parse_network_cfg()中动态分配内存, 
                        // 此值对应未使用gpu时的情况), 该变量貌似不轻易被释放内存, 目前只发现在network.c的resize_network()函数对其进行了释放. 
                        // net.workspace充当一个临时工作空间的作用, 存储临时所需要的计算参数, 比如每层单张图片重排后的结果
                        // (这些参数马上就会参与卷积运算), 一旦用完, 就会被马上更新(因此该变量的值的更新频率比较大)
    
    int train;          // 标志参数, 网络是否处于训练阶段, 如果是, 则值为1(这个参数一般用于训练与测试有不同操作的情况, 比如dropout层, 对于训练, 才需要进行forward_dropout_layer()函数, 对于测试, 不需要进入到该函数)
    int index;          // 标志参数, 当前网络的活跃层(活跃包括前向和反向, 可参考network.c中forward_network()与backward_network()函数)
    
    float *cost;         // 目标函数值, 该参数不是所有层都有的, 一般在网络最后一层拥有, 用于计算最后的cost, 比如识别模型中的cost_layer层, 检测模型中的region_layer层
    float clip;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif

} network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

/**
** 存储图像数据的结构体(不一定是一张图片的, 可能是多张图片拼接在一起存储)
*/
typedef struct {
    int w;      // 每张图片的高度(行)
    int h;      // 每张图片的宽度(列)
    int c;      // 每张图片的通道数
    float *data; // 数据头, 存储二维的图像数据, 按行存储, 如果有多张, 则每张按行存储, 然后所有图片再并成一大行
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;


typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;


network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;


// 双链表结构体
typedef struct list{
    int size;    // 链表长度
    node *front; // 链表第一个节点
    node *back;  // 链表最后一个节点
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void cuda_set_device(int n);
void cuda_free(float *x_gpu);
float *cuda_make_array(float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
float train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

#ifdef __cplusplus
}
#endif
#endif
