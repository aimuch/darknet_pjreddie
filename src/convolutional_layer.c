#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

/*
** 交换weights和binary_weights
*/
void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

/*
** l.binary_weights根据对应l.weights正负赋所有权重绝对值的均值mean, 正赋mean, 负赋-mean
*/
void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){ // size为组卷积核元素个数
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

/*
** l.binary_input根据对应的net.input正负赋1或-1
*/
void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

/*
**  根据输入图像的高度(h), 两边补0的个数(pad), 卷积核尺寸(size)以及跨度(stride)计算输出的特征图的高度
**  输入: l    卷积层, 包含该卷积层的所有参数, 实际这里没有必要输入整个l, 因为只需要到其中的四个参数而已
**  输出: int类型, 输出图像的高度
**  说明: 这个函数的实现应该可以进一步改善一下, 虽然这个函数只是在最初构建网络时调用一次, 之后就不调用了, 不怎么影响性能, 
**       但输入整个l实在不妥(l比较大, 按值传递复制过程比较冗长), 要么就只输入用到的四个参数, 要么传入l的指针, 
**       并且不需要返回值了, 直接在函数内部为l.out_h赋值
*/
int convolutional_out_height(convolutional_layer l)
{
    // pad是每边补0的个数, 因此乘以2
    // 当stride=1, pad=size/2(整数除法, 会往下取整)时, 输出高度就等于输入高度(same策略);
    // 当stride=1, pad=0时, 为valid策略;
    // 当stride!=1时, 输出高度恒小于输入高度(尺寸一定会缩小)
    // 计算公式推导: 设输出高度为x, 总图像高度为h+2*pad个像素, 输出高度为x, 则共有x-1次卷积核移动, 
    // 共占有原图像(x-1)*stride+size个像素, 可能还剩余res个像素, 且res一定小于stride(否则还可以再移位一次), 
    // 因此有(x-1)*stride+size+res=h+2*pad, => x=(h+2*pad-size)/stride+1-res/stride, 因为res<stride, 
    // 对于整数除法来说, 值为0,于是得到最终的输出高度为x=(h+2*pad-size)/stride+1
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

/*
**  根据输入图像的宽度(w), 两边补0的个数(pad), 卷积核尺寸(size)以及跨度(stride)计算输出的特征图的宽度
**  与上一个函数convolutional_out_height()类似, 不再赘述
*/
int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif


/*
**  输入: batch           每个batch含有的图片数
**       h               图片高度(行数)
**       w               图片宽度(列数)
**       c               输入图片通道数
**       n               卷积核个数
**       size            卷积核尺寸
**       stride          步长
**       padding         四周补0长度
**       activation      激活函数类型
**       batch_normalize 是否进行BN(规范化)
**       binary          是否对权重进行二值化
**       xnor            是否对权重以及输入进行二值化
**       adam            使用
*/
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;

    // convolutional_layer是使用typedef定义的layer的别名
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL; // 层属性: 卷积层

    l.groups = groups;      // 分组卷积组数
    l.h = h;                // 输入图像高度
    l.w = w;                // 输入图像宽度
    l.c = c;                // 输入图像通道数
    l.n = n;                // 卷积核数量(即滤波器个数)
    l.binary = binary;      // 是否对权重进行二值化
    l.xnor = xnor;          // 是否对权重以及输入进行二值化
    l.batch = batch;        // 每个batch含有的图片数
    l.stride = stride;      // 步长
    l.size = size;          // 卷积核尺寸
    l.pad = padding;        // 四周补0长度
    l.batch_normalize = batch_normalize; // 是否进行BN(规范化)

    // 该卷积层总的权重元素(卷积核元素)个数=输入图像通道数/分组卷积组数*卷积核个数*卷积核尺寸*卷积核尺寸
    // (因为一个卷积核要作用在输入图片的所有通道上, 所以说是一个卷积核, 实际含有的卷积核参数个数需要乘以输入图片的通道数)
    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float)); // 开空间, 并用0初始化

    // bias就是Wx+b中的b(上面的weights就是W), 有多少个卷积核, 就有多少个b(与W的个数一一对应, 每个W的元素个数为c/groups*size*size)
    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float)); // 开空间, 并用0初始化

    // 该卷积层总的权重元素个数(权重元素个数=输入数据的通道数/分组卷积组数*卷积核个数*卷积核尺寸*卷积核尺寸, 注意因为每一个卷积核是同时作用于输入数据
    // 的多个通道上的, 因此实际上卷积核是三维的, 包括两个维度的平面尺寸, 以及输入数据通道数这个维度, 每个通道上的卷积核参数都是独立的训练参数)
    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);

    // 初始化权重: 缩放因子*标准正态分布随机数, 缩放因子等于sqrt(2./(size*size*c/groups)), 为什么取这个值呢??
    // 此处初始化权重为正态分布, 而在全连接层make_connected_layer()中初始化权重是均匀分布的. 
    // TODO: 个人感觉, 这里应该加一个if条件语句: if(weightfile), 因为如果导入了预训练权重文件, 就没有必要这样初始化了(事实上在detector.c的train_detector()函数中, 
    // 紧接着parse_network_cfg()函数之后, 就添加了if(weightfile)语句判断是否导入权重系数文件, 如果导入了权重系数文件, 也许这里初始化的值也会覆盖掉, 
    // 总之这里的权重初始化的处理方式还是值得思考的, 也许更好的方式是应该设置专门的函数进行权重的初始化, 同时偏置也是, 不过这里似乎没有考虑偏置的初始化, 在make_connected_layer()中倒是有...)
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i){
        l.weights[i] = scale*rand_normal();
    }

    // 根据该层输入图像的尺寸、卷积核尺寸以及跨度计算输出特征图的宽度和高度
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;    // 输出图像高度
    l.out_w = out_w;    // 输出图像宽度
    l.out_c = n;        // 输出图像通道(等于卷积核个数, 有多少个卷积核, 最终就得到多少张特征图, 每张图是一个通道)
    
    // 关于下面两个参数的说明:
    // 一个mini_batch中有多张图片, 每张图片可能有多个通道(彩色图有三通道), l.inputs是每张输入图片所有通道的总元素个数, 
    // 而每张输入图片会有n个卷积核对其进行卷积操作, 因此一张输入图片会输出n张特征图, 这n张特征图的总元素个数就为l.outputs
    l.outputs = l.out_h * l.out_w * l.out_c;    // 对应每张输入图片的所有输出特征图的总元素个数(每张输入图片会得到n也即l.out_c张特征图)
    l.inputs = l.w * l.h * l.c;                 // mini_batch中每张输入图片的像素元素个数

    // l.output为该层所有的输出(包括mini_batch所有输入图片的输出)
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float)); // 该层所有参数变化量

    // 卷积层三种指针函数, 对应三种计算: 前向, 反向, 更新
    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

/*
** 计算每个卷积核的偏置更新值, 所谓偏置更新值, 就是bias = bias - alpha * bias_update中的bias_update
** 输入:  bias_updates  当前层所有偏置的更新值, 维度为l.n(即当前层卷积核的个数)
**       delta         当前层的敏感度图(即l.delta)
**       batch         一个batch含有的图片张数(即l.batch)
**       n             当前层卷积核个数(即l.n)
**       size          当前层输入特征图尺寸(即l.out_w*l.out_h)
** 原理: 当前层的敏感度图l.delta是误差函数对加权输入的导数, 也就是偏置更新值, 只是其中每l.out_w*l.out_h个元素都对应同一个
**      偏置, 因此需要将其加起来, 得到的和就是误差函数对当前层各偏置的导数(l.delta的维度为l.batch*l.n*l.out_h*l.out_w,
**      可理解成共有l.batch行, 每行有l.n*l.out_h*l.out_w列, 而这一大行又可以理解成有l.n, l.out_h*l.out_w列, 这每一小行就
**      对应同一个卷积核也即同一个偏置)
*/
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;

    // 遍历batch中每张输入图片
    // 注意, 最后的偏置更新值是所有输入图片的总和(多张图片无非就是重复一张图片的操作, 求和即可)
    // 总之: 一个卷积核对应一个偏置更新值, 该偏置更新值等于batch中所有输入图片累积的偏置更新值, 
    // 而每张图片也需要进行偏置更新值求和(因为每个卷积核在每张图片多个位置做了卷积运算, 这都对偏置更新值有贡献)以得到每张图片的总偏置更新值.
    for(b = 0; b < batch; ++b){
        // 求和得一张输入图片的总偏置更新值
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}


/*
** 前向卷积层
*/
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    // l.outputs = l.out_h * l.out_w * l.out_c在make各网络层函数中赋值(比如make_convolutional_layer()), 
    // 对应每张输入图片的所有输出特征图的总元素个数(每张输入图片会得到n也即l.out_c张特征图)
    // 初始化输出l.output全为0.0; 输入l.outputs*l.batch为输出的总元素个数, 其中l.outputs为batch中一个输入
    // 对应的输出的所有元素的个数, l.batch为一个batch输入包含的图片张数; 0表示初始化所有输出为0; 
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    // 是否进行二值化操作(这个操作应该只有第一个卷积层使用吧？因为下面直接对net.input操作, 这个理解是错误的, 因为在forward_network()含中, 
    // 每进行一层都会将net.input = l.output, 即下一层的输入被设置为当前层的输出)
    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights); // l.binary_weights根据对应l.weights正负赋所有权重绝对值的均值mean, 正赋mean, 负赋-mean
        swap_binary(&l); // 将二值化后的binary_weights与weights交换
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input); // l.binary_input根据对应的net.input正负赋1或-1
        net.input = l.binary_input; // 将二值化后的输入赋给中间变量输入
    }

    int m = l.n/l.groups;               // 该层卷积核个数
    int k = l.size*l.size*l.c/l.groups; // 该层每组卷积核参数元素个数
    int n = l.out_w*l.out_h;            // 该层输出特征图的尺寸(元素个数)

    for(i = 0; i < l.batch; ++i){       // 图像数量
        for(j = 0; j < l.groups; ++j){  // 卷积组数
            float *a = l.weights + j*l.nweights/l.groups;   // 每组所有卷积核(也即权重), 元素个数为l.n*l.c*l.size*l.size, 按行存储, 共有l*n行, l.c*l.size*l.size列
            float *b = net.workspace;                       // 对输入图像进行重排之后的图像数据
            
            // 对c进行指针偏移: 移到batch中下一张图片卷积组对应输出的起始位置(每循环一次, 将完成对一张图片的卷积操作, 产生的所有特征图的元素个数总和为n*m)
            float *c = l.output + (i*l.groups + j)*n*m;     // 每组存储一张输入图片(多通道)所有的输出特征图(输入图片是多通道的, 输出图片也是多通道的, 有多少个卷积核就有多少个通道, 每个卷积核得到一张特征图即为一个通道)
            
            // 同样, 输入也进行指针偏移, 移动到下一张图片卷积组元素的起始位置, 以便下一次循环处理(batch中每张图片的元素个数为通道数*高度*宽度, 即l.c*l.h*l.w)
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w; // 每组输入的偏移指针

            if (l.size == 1) { // 1*1的卷积核
                b = im;
            } else {

                // 将分组后的图像im变成按一定存储规则排列的数组b, 以方便、高效地进行矩阵(卷积)计算, 详细查看该函数注释(比较复杂)
                // 注意im包含batch中所有图片的数据, 但是每次循环只处理一张(在本循环开始的时候对im进行了移位), 因此在im2col_cpu仅会对其中一张图片
                // 进行重排, l.c/l.groups为每组单张图片的通道数, l.h为每张图片的高度, l.w为每张图片的宽度, l.size为卷积核尺寸, l.stride为跨度
                // 得到的b为一张图片重排后的结果, 也是按行存储的一维数组(共有l.c*l.size*l.size行, l.out_w*l.out_h列), 
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }

            // GEneral Matrix to Matrix Multiplication
            // 此处在im2col_cpu操作基础上, 利用矩阵乘法 c = alpha*a*b + beta*c 完成对图像卷积的操作
            // 0,0表示不对输入a,b进行转置, 
            // m是输入a,c的行数, 具体含义为每个卷积核的个数, 
            // n是输入b,c的列数, 具体含义为每个输出特征图的元素个数(out_h*out_w), 
            // k是输入a的列数也是b的行数, 具体含义为卷积核元素个数乘以输入图像的通道数(l.size*l.size*l.c), 
            // a,b,c即为三个参与运算的矩阵(用一维数组存储),alpha=beta=1为常系数, 
            // a为所有卷积核集合, 元素个数为l.n*l.c*l.size*l.size, 按行存储, 共有l*n行, l.c*l.size*l.size列, 
            // 即a中每行代表一个可以作用在3通道上的卷积核, 
            // b为一张输入图像经过im2col_cpu重排后的图像数据(共有l.c*l.size*l.size行, l.out_w*l.out_h列), 
            // c为gemm()计算得到的值, 包含一张输入图片得到的所有输出特征图(每个卷积核得到一张特征图), c中一行代表一张特征图, 
            // 各特征图铺排开成一行后, 再将所有特征图并成一大行, 存储在c中, 因此c可视作有l.n行, l.out_h*l.out_w列。
            // 详细查看该函数注释(比较复杂)
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    // 如需要规范化(BN在非线性激活函数处理之前完成)
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

/*
** 卷积神经网络反向传播核心函数
** 主要流程: 1) 调用gradient_array()计算当前层l所有输出元素关于加权输入的导数值(也即激活函数关于输入的导数值), 
**             并乘以上一次调用backward_convolutional_layer()还没计算完的l.delta, 得到当前层最终的敏感度图;
**          2) 如果网络进行了BN, 则调用backward_batchnorm_layer()函数进行规范化处理;
**          3) 如果网络没有进行BN, 则直接调用 backward_bias()计算当前层所有卷积核的偏置更新值;
**          4) 依次调用im2col_cpu(), gemm_nt()函数计算当前层权重系数更新值;
**          5) 如果上一层的delta已经动态分配了内存, 则依次调用gemm_tn(), col2im_cpu()计算上一层的敏感度图(并未完成所有计算, 还差一个步骤);
** 强调: 每次调用本函数会计算完成当前层的敏感度计算, 同时计算当前层的偏置、权重更新值, 除此之外, 还会计算上一层的敏感度图, 但是要注意的是, 
**      并没有完全计算完, 还差一步: 乘上激活函数对加权输入的导数值. 这一步在下一次调用本函数时完成. 
*/
void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups; // 每组卷积核个数

    // 每组一个卷积核元素个数(包括l.c(l.c为该层网络接受的输入图片的通道数)个通道上的卷积核元素个数总数, 比如卷积核尺寸为3*3,
    // 输入图片有3个通道, 因为要同时作用于输入的3个通道上, 所以实际上这个卷积核是一个立体的, 共有3*3*3=27个元素, 这些元素都是要训练的参数)
    int n = l.size*l.size*l.c/l.groups; // 每一组一个卷积核元素个数
    int k = l.out_w*l.out_h;            // 每张输出特征图的元素个数: out_w, out_h是输出特征图的宽高

    // 计算当前层激活函数对加权输入的导数值并乘以l.delta相应元素, 从而彻底完成当前层敏感度图的计算, 得到当前层的敏感度图l.delta. 
    // l.output存储了该层网络的所有输出: 该层网络接受一个batch的输入图片, 其中每张图片经卷积处理后得到的特征图尺寸为: l.out_w,l.out_h, 
    // 该层卷积网络共有l.n个卷积核, 因此一张输入图片共输出l.n张宽高为l.out_w,l.out_h的特征图(l.outputs为一张图所有输出特征图的总元素个数), 
    // 所以所有输入图片也即l.output中的总元素个数为: l.n*l.out_w*l.out_h*l.batch; 
    // l.activation为该卷积层的激活函数类型, l.delta就是gradient_array()函数计算得到的l.output中每一个元素关于激活函数函数输入的导数值, 
    // 注意, 这里直接利用输出值求得激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数关于输入的导数值都可以描述为输出值的函数表达式, 
    // 比如对于Sigmoid激活函数(记作f(x)), 其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x), 那么f(x)'=y*(1-y), 只需要输出值y就可以了, 不需要输入x的值, 
    // (暂时不确定darknet中有没有使用特殊的激活函数, 以致于必须要输入值才能够求出导数值, 在activiation.c文件中, 有几个激活函数暂时没看懂, 也没在网上查到). 
    // l.delta是一个一维数组, 长度为l.batch * l.outputs(其中l.outputs = l.out_h * l.out_w * l.out_c), 在make_convolutional_layer()动态分配内存；
    // 再强调一次: gradient_array()不单单是完成激活函数对输入的求导运算, 还完成计算当前层敏感度图的最后一步: l.delta中每个元素乘以激活函数对输入的导数(注意gradient_arry中使用的是*=运算符). 
    // 每次调用backward_convolutional_laye时, 都会完成当前层敏感度图的计算, 同时会计算上一层的敏感度图, 但对于上一层, 其敏感度图并没有完全计算完成, 还差一步, 
    // 需要等到下一次调用backward_convolutional_layer()时来完成, 诚如col2im_cpu()中注释一样
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        // 计算偏置的更新值: 每个卷积核都有一个偏置, 偏置的更新值也即误差函数对偏置的导数, 这个导数的计算很简单, 实际所有的导数已经求完了, 都存储在l.delta中, 
        // 接下来只需把l.delta中对应同一个卷积核的项加起来就可以(卷积核在图像上逐行逐列跨步移动做卷积, 每个位置处都有一个输出, 共有l.out_w*l.out_h个, 
        // 这些输出都与同一个偏置关联, 因此将l.delta中对应同一个卷积核的项加起来即得误差函数对这个偏置的导数)
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    // 遍历batch中的每张照片, 对于l.delta来说, 每张照片是分开存的, 因此其维度会达到: l.batch*l.n*l.out_w*l.out_h, 
    // 对于l.weights,l.weight_updates以及上面提到的l.bias,l.bias_updates, 是将所有照片对应元素叠加起来
    // (循环的过程就是叠加的过程, 注意gemm()这系列函数含有叠加效果, 不是覆盖输入C的值, 而是叠加到之前的C上), 
    // 因此l.weights与l.weight_updates维度为l.n*l.size*l.size, l.bias与l.bias_updates的维度为l.h, 都与l.batch无关
    for(i = 0; i < l.batch; ++i){       // 每张图片
        for(j = 0; j < l.groups; ++j){  // 每个卷积组
            float *a = l.delta + (i*l.groups + j)*m*k;

            // net.workspace的元素个数为所有层中最大的l.workspace_size(在make_convolutional_layer()计算得到workspace_size的大小, 在parse_network_cfg()中动态分配内存, 此值对应未使用gpu时的情况),
            // net.workspace充当一个临时工作空间的作用, 存储临时所需要的计算参数, 比如每层单张图片重排后的结果(这些参数马上就会参与卷积运算), 一旦用完, 就会被马上更新(因此该变量的值的更新频率比较大)
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

