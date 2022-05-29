# 仿真验证环境及说明文档

## 验证工具
基于 System Verilog 进行 TestBench 验证代码编写。

使用由队伍成员邓立唯开源的 [**Bitmap Processing Library & AXI-Stream Video Image VIP**](https://github.com/Aperture-Electronic/SystemVerilog-Bitmap-Library-AXI-Image-VIP) 进行测试样例图片读取写回，简化测试样例生成步骤及测试结果输出对比。

> To verficate a video or a image processing IP, you may need to read a real image into your design, send its data by an interface. Then, get the output from the interface, and convert it to a new image, save or compare it.

由于队伍成员每个人有不同的验证工具使用习惯，在本次项目中验证会使用到：
- Intel ModelSim
- Synopsys VCS & Verdi
- Verilator & GtkWave


## 验证方法及策略
采用 **直接验证** 与 **随机验证** 结合。

直接验证：通过对比测试图片在 C Model 进行超分辨率算法运算结果与 RTL 代码在仿真中输出结果对比。

随机验证：对部分子模块(如高斯滤波器、纹理分类器等)所需的运算数据通过产生随机种子产生随机数据，对比参考模型运算输出结果与待测模块输出结果。


## 验证范围

### Bicubic 上采样模块
- 满足 AXI4-Stream 协议要求，完成视频流数据收发；
- 完成 Biubic 上采样算法结果输出；
- 运算结果与参考模型匹配；
- 结果输出是否超出数据范围；
- 满足时序要求。

### 纹理分类模块
- 满足 AXI4-Stream 协议要求，完成视频流数据接收；
- 完成纹理分类算法结果输出；
- 输出寻址结果与参考模型匹配；
- 结果输出是否超出地址范围；
- 满足时序要求。

### 自适应锐化模块
- 满足 AXI4-Stream 协议要求，完成视频流数据收发；
- 完成自适应锐化算法结果输出；
- 输出像素结果与参考模型匹配；
- 结果输出是否超出数据范围；
- 满足时序要求。

### 注意事项
- 不需进行跨时钟域检查
- 需进行代码覆盖率验证

## 验证环境

### 验证平台
该验证平台包含了测试样例生成器、驱动器、待测单元、其他外设VIP(如DDR)、监视器、记分板、参考模型以及一个全局配置。

<img src="doc.assets/Screen Shot 2022-05-29 at 2.44.29 PM.png" style="zoom:50%;" />

### 工程目录
<table border="1">
    <tr>
        <th class="tg-9wq8" rowspan="24">src</th>
        <th class="tg-9wq8" rowspan="2">bicubic</th>
        <th class="tg-lboi">dsp_simd2x_int9xuint8.sv</th>
    </tr>
    <tr>
        <th class="tg-lboi">sfr.v</th>
    </tr>
    <tr>
        <th class="tg-9wq8" rowspan="16">texture_classifier</th>
        <th class="tg-lboi">gaus_5x5_kernel.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">gaus_para_reg.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">dsp_5x5_uint9xuint8_multiplier.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">dsp_25_sum_adder.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">dsp_round_limit.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">lap_3x3_kernel.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">dsp_3x3_sum_adder.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">data_split.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">uniform_code.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">angle_code.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">adderss_code.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">fifo_3line.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">fifo_5line.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">fifo.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">fifo_mapping.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">fifo_ctl_unit.v</th>
    </tr>
    <tr>
        <th class="tg-9wq8" rowspan="6">adaptive_sharp</th>
        <th class="tg-lboi">dsp_conv_5x5_multiplier.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">dsp_25_sum_adder.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">dsp_round_limit.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">filter_weight_store.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">filter_weight_fetch.v</th>
    </tr>
    <tr>
        <th class="tg-lboi">ram.v</th>
    </tr>
    <tr>
        <td class="tg-lboi" rowspan="3">sim</td>
        <td class="tg-9wq8">ref</td>
        <td class="tg-lboi"></td>
    </tr>
    <tr>
        <td class="tg-9wq8">scb</td>
        <td class="tg-lboi"></td>
    </tr>
    <tr>
        <td class="tg-9wq8">tb</td>
        <td class="tg-lboi"></td>
    </tr>
</table>


### 待验证设计
- .....


## 覆盖率

当前项目已完成设计部分，即将进入验证阶段。

## 验证分析

当前项目已完成设计部分，即将进入验证阶段。