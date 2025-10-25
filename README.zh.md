<div align="center">
<p align="center">
  Triton NPU编程案例
</p>
</div>

## 简介
GPU和NPU存在体系结构的一些差异，Ascend Triton样例仓提供了一些代码样例给开发者进行参考学习，帮助开发者快速掌握基于NPU Triton算子的一些优化实践，以及基于GPU优化的Triton算子如何快速迁移到NPU实现优化性能。

## 试用范围
本案例仓的一些参考案例，适用范围在A2/A3等昇腾硬件。

## 案例仓结构

```
-basic：基础优化的
-op_extension: Ascend扩展的一些Op语义，提升性能
-transformer：常用网络模型的一些优化案例
```

## 算子案例
#### 1、分核调整
GPU是SIMT架构，基于GPU优化的Triton算子，分核较多，迁移到NPU或者基于NPU原生开发的Triton算子，需要减少分核数，单核内处理更多的数据量，单核内数据做Tiling，并启用double buffer，才能实现性能最优。
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [AddCustomSample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials/AddCustomSample) | 基于Ascend C的Add自定义Vector算子及调用样例 | 
| [HelloWorldSample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials/HelloWorldSample) | 基于Ascend C的自定义算子调用结构演示样例 |

#### 2、数据类型优化
NPU上部分OP矢量运算不支持特定数据类型，计算时会退化为标量运算，影响性能，在确定不影响精度的情况下，建议使用支持的数据类型，提升性能
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [vector cmp](https://github.com/Ascend/triton-ascend-examples/blob/main/basic/002-vector_cmp.zh.md) | Cmp Op不支持int32数据类型矢量运算，确定不影响精度的情况下，转换为FP32 |
| [vector add](https://github.com/Ascend/triton-ascend-examples/blob/main/basic/001-vector_add.zh.md) | 矢量运算不支持int64数据类型，确定不溢出的情况下，转换为int32 | 

#### 3、离散访存
NPU上，Triton算子批量处理数据，能够实现性能的最优，在批量读取数据时，如果数据存放位置不连续，则会导致数据无法批量加载，退化为标量加载
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [discrete-memory-access](https://github.com/Ascend/triton-ascend-examples/blob/main/basic/004-discrete_memory_access.zh.md) | 小数据量的离散访存，全部读取到UB中，使用Gather语义筛选 |


#### 4、提升并行度
提升指令并行度，有助于大幅提升性能
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [load order](https://github.com/Ascend/triton-ascend-examples/blob/main/basic/005-load_order.zh.md) | 将没有数据依赖的Load提前，MTE2指令并行下发加速 |

#### 5、扩展语义
基于NPU特点，提供一些扩展语义，加速数据处理
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [InsertSlice](https://github.com/Ascend/triton-ascend-examples/blob/main/op_extension/001-insert_slice.zh.md) | 多个数据合并处理，提升数据处理效率 |
| [ExtractSlice](https://github.com/Ascend/triton-ascend-examples/blob/main/op_extension/002-extract_slice.zh.md) | 一次批量读取，截取部分数据处理，提升数据处理效率 |
| [LoadPadding](https://github.com/Ascend/triton-ascend-examples/blob/main/op_extension/003-load_care_padding.md) | 带Mask的数据加载，被Mask抹掉部分如果不需要默认值，则不填写Other，提升MTE2与Vector的并行 |


