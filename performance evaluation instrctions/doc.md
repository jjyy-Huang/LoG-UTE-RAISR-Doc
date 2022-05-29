# 性能评估文档

## Overview
性评估主要分为三大部分，针对于每个设计 IP 进行单独评估分析，其中包括 Bicubic 上采样模块性能评估、纹理分类模块性能评估以及自适应锐化模块性能评估。

## Bicubic 上采样模块性能评估

### 性能参数
使用 Vivado 与 Synopsys Synplify Premier 进行时序分析。输入 $960\times540$ 图像数据作为基准。

#### 最大频率

| FPGA Device Family        | Analysis Tool                     | Fmax (MHz) |
|---------------------------|-----------------------------------|------------|
| Xilinx Virtex UltraScale+ | Synopsys Synplify Premier 2020.03 | 628.6      |
| Xilinx Kintex UltraScale+ | Synopsys Synplify Premier 2020.03 | 417.2      |
| Xilinx Zynq UltraScale+   | Synopsys Synplify Premier 2020.03 | 394.1      |
| Xilinx Zynq UltraScale+   | Vivado 2021.1                     | 428.4      |
| Xilinx Kintex 7           | Synopsys Synplify Premier 2020.03 | 361.7      |
| Xinlinx Artix 7           | Synopsys Synplify Premier 2020.03 | 192.2      |
| Intel Stratix 10          | Synopsys Synplify Premier 2020.03 | 260.7      |
| Intel Max 10              | Synopsys Synplify Premier 2020.03 | 220.5      |
| Intel Arria V             | Synopsys Synplify Premier 2020.03 | 142.5      |

#### 最大延迟
仅用于评估该IP的路径时延，不考虑受系统延迟或其他限制。假设图像输入宽为 $W$ 位像素。

| 描述 | 时钟周期 |
|---|---|
| Bicubic流水运算从输入到输出 | 13 |
| 第一个像素进入到第一个像素输出 | $10\cdot W+28$ |
| 最后一个像素进入到最后一个像素输出 | $13\cdot W + 31$ |

#### 吞吐量
用于评估不同图像帧大小进入。

| 输入分辨率 | 输出分辨率 | 吞吐量(FPS/MHz) | FPS@150MHz |
|:---:|:---:|:---:|:---:|
| 320 x 240 | 1280 x 960 | 3.23 | 484.7 |
| 480 x 270 | 1920 x 1080 | 1.92 | 287.6 |
| 640 x 360 | 2560 x 1440 | 1.08 | 162.1 |
| 960 x 540 | 3840 x 2160 | 0.48 | 72.1 |

### 资源使用量
Xilinx Zynq UltraScale+ 器件的资源利用结果是在 Vivado 综合器下，使用 DSP48E2 和 XPM 宏进行评估的。其他器件的评估结果是使用 Verilog 自动推断完成的，可能由于每个器件的 DSP 模块的数据宽度不同而导致不同。实时视频 Bicubic IP 是为 Xilinx UltraScale+ 系列器件的 DSP48E2 模块特别优化的。为了最大限度地利用资源，建议使用这些器件进行合成。

<img src="doc.assets/Screenshot from 2022-05-29 16-14-20.png" style="zoom:50%;" />


## 纹理分类模块性能评估

由于该模块处于整合阶段，故未进行详细性能评估。

### 性能参数
使用 Vivado 进行时序分析。

#### 最大频率
未进行评估

#### 最大延迟
仅用于评估该IP的路径时延，不考虑受系统延迟或其他限制。假设图像输入宽为 $W$ 位像素。

| 描述 | 时钟周期 |
|---|---|
| 高斯滤波运算从输入到输出 | 9 |
| 拉普拉斯滤波运算从输入到输出 | 4 |
| 纹理检测器运算从输入到输出 | 1 |
| 第一个像素进入到第一个像素输出 | $8\cdot W+14$ |
| 最后一个像素进入到最后一个像素输出 | $8\cdot W + 14$ |

#### 吞吐量
未进行评估

### 资源利用率
Xilinx Zynq UltraScale+ 器件的资源利用结果是在 Vivado 综合器下，使用 DSP48E2 和 XPM 宏进行评估的。其他器件的评估结果是使用Verilog自动推断完成的，可能由于每个器件的 DSP 模块的数据宽度不同而导致不同。实时视频纹理分类 IP 是为 Xilinx UltraScale+ 系列器件的 DSP48E2 模块特别优化的。为了最大限度地利用资源，建议使用这些器件进行合成。

未进行评估。

## 自适应锐化模块性能评估

### 性能参数
使用 Vivado 进行时序分析。

#### 最大频率
此模块最大频率限制是由于内部设计使用了 URAM288，其最大速率由工艺决定。

| FPGA Device Family           | Analysis Tool                     | Fmax (MHz) |
|------------------------------|-----------------------------------|------------|
| Xilinx Virtex UltraScale+    | Vivado 2021.2                     | 355.1      |
| Xilinx Kintex UltraScale+    | Vivado 2021.2                     | 351.3      |
| Xilinx Zynq UltraScale+      | Vivado 2021.2                     | 344.4      |
| Xilinx Versal AI Core Series | Vivado 2021.2                     | 348.8      |

#### 最大延迟
仅用于评估该IP的路径时延，不考虑受系统延迟或其他限制。假设图像输入宽为 $W$ 位像素。

| 描述 | 时钟周期 |
|---|---|
| 锐化卷积运算从输入到输出 | 10 |
| 第一个像素进入到第一个像素输出 | $3\cdot W+10$ |
| 最后一个像素进入到最后一个像素输出 | $3\cdot W + 10$ |

#### 吞吐量
未进行评估

### 资源利用率
Xilinx Zynq UltraScale+ 器件的资源利用结果是在 Vivado 综合器下，使用 DSP48E2 和 XPM 宏进行评估的。其他器件的评估结果是使用 Verilog 自动推断完成的，可能由于每个器件的 DSP 模块的数据宽度不同而导致不同。实时视频自适应锐化 IP 是为 Xilinx UltraScale+ 系列器件的 DSP48E2 模块特别优化的。为了最大限度地利用资源，建议使用这些器件进行合成。

<img src="doc.assets/Screen Shot 2022-05-29 at 7.37.17 PM.png" style="zoom:50%;" />



## Conclusion