# Image-processing-experiment
Advanced image processing and analysis' homework

# 图像处理实验1:图像灰度变换
# 要求
1. 利用 OpenCV 读取图像。
具体内容:用打开 OpenCV 打开图像,并在窗口中显示
2. 灰度图像二值化处理
具体内容:设置并调整阈值对图像进行二值化处理。
3. 灰度图像的对数变换
具体内容:设置并调整 r 值对图像进行对数变换。
4. 灰度图像的伽马变换
具体内容:设置并调整γ值对图像进行伽马变换。
5. 彩色图像的补色变换
具体内容:对彩色图像进行补色变换。
# 过程
## 灰度变换
灰度变换的定义域和值域应该相等吧。

伽马变换： $s=cr^\gamma \qquad r\in[0,1]$

对数变换： $s=c\log_{v+1}(1+vr)\qquad r\in[0,1]$
## C++的functional库
整个处理流程大致相同，只有几个变换公式不同，所以把公共部分抽象出来成为一个函数，在每个变换中调用公共处理函数。

之前打算因为几个变换的函数实现中的参数个数不同，所以打算用bind的，后来发现lambda更简洁实用些。

function基本上代替了函数指针。

虽然花费了不少时间，但算是搞懂了function、bind和lambda的区别与用法。

注意：补色被我理解成反色了，代码中的这个实现错误我也懒得改了。。。


# 图像处理实验2:直方图均衡
# 要求
1. 计算灰度图像的归一化直方图。
具体内容:利用 OpenCV 对图像像素进行操作,计算归一化直方图.并在
窗口中以图形的方式显示出来
2. 灰度图像直方图均衡处理
具体内容:通过计算归一化直方图,设计算法实现直方图均衡化处理。
3. 彩色图像直方图均衡处理
具体内容: 在灰度图像直方图均衡处理的基础上实现彩色直方图均衡
处理。

# 过程
## tips
通过使用数值乘以布尔值实现了对不同情况的统一处理。
```plot_poly(hist_image,npdfs[i],Scalar(255*(0==i),255*(1==i),255*(2==i)));```

## 二维vector初始化
大致如下

    vector<vector<float>> name(rows, vector<float>(cols, value));
## 画图
不要使用int类型的bin_width，否则取整时所产生的误差会被放大多倍。

# 图像处理实验3:空域滤波
# 要求
1. 利用均值模板平滑灰度图像。
具体内容:利用 OpenCV 对图像像素进行操作,分别利用 3*3.  5*5 和 9*9
尺寸的均值模板平滑灰度图像
2. 利用高斯模板平滑灰度图像。
具体内容:利用 OpenCV 对图像像素进行操作,分别利用 3*3.  5*5 和 9*9
尺寸的高斯模板平滑灰度图像
3. 利用 Laplacian. Robert. Sobel 模板锐化灰度图像。
具体内容:利用 OpenCV 对图像像素进行操作,分别利用 Laplacian.  Robert. 
Sobel 模板锐化灰度图像
4. 利用高提升滤波算法增强灰度图像。
具体内容:利用 OpenCV 对图像像素进行操作,设计高提升滤波算法增
强图像
5. 利用均值模板平滑彩色图像。
具体内容:利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作,利
用 3*3. 5*5 和 9*9 尺寸的均值模板平滑彩色图像
6. 利用高斯模板平滑彩色图像。
具体内容:利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作,分
别利用 3*3. 5*5 和 9*9 尺寸的高斯模板平滑彩色图像
7. 利用 Laplacian. Robert. Sobel 模板锐化灰度图像。
具体内容:利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作,分
别利用 Laplacian. Robert. Sobel 模板锐化彩色图像

# 过程
我没有严格按照要求做，事实上，我增加了些。
## 向量
可使用vector存储要使用的参数。

在使用vector存储函数时，遇到了些麻烦。gaussian_kernel有3个参数，而vector被我设置成元素为```function<void(Mat&, int)>>```，使用bind将gaussian_kernel转化成符合要求的函数对象（仿函数）即可。

我之前错误的做法是，又另外实现了满足条件的只需要接受Mat&,int两个参数的gaussian_kernel函数，函数重载。但编译器似乎不知道该使用哪一个gaussian_kernel函数了

## opencv的矩阵运算
opencv的矩阵运算支持多通道Mat，所以我之前将彩色图像split后处理再merge的做法反而多余了。

当需要对不同通道做不同处理时，这时就需要split-->process-->merge处理流程了。

## Robert算子
一般见到的Robert算子是2×2的，但一般卷积核大小为奇数，所以可把2*2的卷积核放到3×3的右下方。

## 解耦
在实验1时，我把公共部分抽象成一个公共函数，供各个函数调用。

在此次实验中，我把参数和处理函数（回调函数）分别放到不同的vector。将公共处理部分在主函数的for循环中，主动调用vector中的函数和参数。

# 图像处理实验4:图像去噪
# 要求
1. 均值滤波
具体内容:利用 OpenCV 对灰度图像像素进行操作,分别利用算术均值滤波器. 几何均值滤波器. 谐波和逆谐波均值滤波器进行图像去噪。模板大小为5\*5。(注:请分别为图像添加高斯噪声. 胡椒噪声. 盐噪声和椒盐噪声,并观察滤波效果)
2. 中值滤波
具体内容:利用 OpenCV 对灰度图像像素进行操作,分别利用 5\*5 和 9\*9尺寸的模板对图像进行中值滤波。(注:请分别为图像添加胡椒噪声. 盐噪声和椒盐噪声,并观察滤波效果)
3. 自适应均值滤波。
具体内容:利用 OpenCV 对灰度图像像素进行操作,设计自适应局部降低噪声滤波器去噪算法。模板大小 7\*7 (对比该算法的效果和均值滤波器的效果)
4. 自适应中值滤波
具体内容:利用 OpenCV 对灰度图像像素进行操作,设计自适应中值滤波算法对椒盐图像进行去噪。模板大小 7\*7(对比中值滤波器的效果)
5. 彩色图像均值滤波
具体内容:利用 OpenCV 对彩色图像 RGB 三个通道的像素进行操作,利用算术均值滤波器和几何均值滤波器进行彩色图像去噪。模板大小为 5\*5。

# 过程
## calcHist
使用的是
```

void cv::calcHist	(	const Mat * 	images,
int 	nimages,
const int * 	channels,
InputArray 	mask,
OutputArray 	hist,
int 	dims,
const int * 	histSize,
const float ** 	ranges,
bool 	uniform = true,
bool 	accumulate = false 
)
```
### channels
当参数channels为1时，效果和np.histogram差不多。当channels为2时，效果和np.histogram2d差不多。

注意：当channels>1时，函数不是分别计算各个channel的一维直方图

### ranges
range是个二维float数组。其中的各项（一维数组）指定了直方图各个维度的bin边界（取值范围）。当直方图中bin的边界是均匀分割得到的时候（unifor = true），对于每个维度i，只需要指定该维度的第一个bin（索引0）的下边界（包含在内）$L_0$和该维度的最后一个bin（索引为histSize[i]-1）的上边界$U_{\texttt{histSize}[i]-1}$（排除在外）。在这种情况下，ranges的各项（一维数组）只需要包含2个元素（float）就行了。在另外的情况下（uniform=false），ranges[i]包含histSize[i]+1个元素$L_0, U_0=L_1, U_1=L_2, ..., U_{\texttt{histSize[i]}-2}=L_{\texttt{histSize[i]}-1}, U_{\texttt{histSize[i]}-1}$。图片中没有被包含在$L_0$和$U_{\texttt{histSize[i]}-1}$之间的，不会被统计进直方图中。

## STL和Mat
vector<Mat>是深复制。

高效的使用STL 当对象很大时，建立指针的容器而不是对象的容器


[带你深入理解STL之迭代器和Traits技法](https://blog.csdn.net/terence1212/article/details/52287762)

cppMat中对应的是
```
template<typename _Tp>
class cv::DataType< _Tp >
Template "trait" class for OpenCV primitive data types.

```

## static local var
使用静态局部变量实现带*状态*的函数


## nth_element
STL中用来寻找在全排序中位列第n个元素，实际上并不需要全排序，局部排序即可。应该是借鉴快排的思路了。

## ROI
*locateROI*的函数用来寻找ROI（如果ROI是从Mat中截取出来的）在原Mat中的位置。

*adjustROI*的函数用来调整ROI的边界。

## 结构设计
我不是太清楚这些设计的正确命名
### 调用处理框架
特定函数，首先实现其特定的小函数，然后将特定小函数作为回调函数传递给并调用公共的处理函数(类似于处理框架)。主体是特定函数

### 函数表驱动
将特定函数部分实现后放进函数表里。公共处理部分(框架)主动调用函数表里的函数。主体为公共处理部分

如果将公共部分抽象出来成函数，将特定的小函数和公共处理函数封装成函数，就成了实验一中的特定函数。估计在公共处理部分只是整个程序的一小部分时才会这么做吧，比如说库。opencv的特定滤波器好像就这么实现的

# 图像处理实验5:频域滤波
# 要求

1.  灰度图像的 DFT 和 IDFT。
具体内容:利用 OpenCV 提供的 cvDFT 函数对图像进行 DFT 和 IDFT 变换
2. 利用理想高通和低通滤波器对灰度图像进行频域滤波具体内容:利用 cvDFT 函数实现 DFT ,在频域上利用理想高通和低通滤波器进行滤波,并把滤波过后的图像显示在屏幕上(观察振铃现象),要求截止频率可输入。
3. 利用布特沃斯高通和低通滤波器对灰度图像进行频域滤波。具体内容:利用 cvDFT 函数实现 DFT ,在频域上进行利用布特沃斯高通和低通滤波器进行滤波,并把滤波过后的图像显示在屏幕上(观察振铃现象),要求截止频率和 n 可输入。
# 过程
## fftshift
将频率域的谱中心化。
## distanceTransform
Calculates the distance to the closest zero pixel for each pixel of the source image.

算是个小技巧吧。可以调库计算各个点到特定点的距离。
