本系列的主要目标是，解决transformer的$O(n^2)$时间复杂度和空间复杂度问题。具体实现方法基于底层GPU算子。
## 1.  FlashAttention1
将QKV分块读取，在GPUSRAM中进行计算与读写
将K、V分到4个warps(NVIDIA GPU 并行计算的基本单元)；但Q都可见。
## 2.  FlashAttention2
将K、V对4个warps都可见而把Q分割到4的warps

本任务由南京公司牵头负责，北京配合完成。
