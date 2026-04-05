- **表示方式：**
    - 当前：
    给定 base mesh 的顶点集${\left\{v_i\right\}}^{V_{base}}_{i=1}$ 和从scan mesh上采样的点云${\left\{p_j\right\}}^{P}_{j=1}$，将每个点云点用knn分配到**最近的** base mesh顶点$v_i$，并计算相对于$v_i$的坐标和它在scan上的法线。最终，每个顶点$v_i$对应一个点云集$N(v_i)$。
    - 为什么不能直接把base mesh顶点半径r范围内的点云点编码？
        
        因为base mesh的顶点稀疏程度不固定，容易漏点，而扩大半径则容易导致取到其他面的点（如下左图：当顶点在手背时，因为手部很薄，所以会把手心部分的点云点也算入自己的点云集，位移场就不知道该往手心还是手背偏移了）
        
        ![image.png](attachment:ff1bb692-dd45-4b3d-8410-2ad9f89c6275:image.png)
        
- **编码器：**
    - 原始方法：PointNet风格，对每个$v_i$的局部点云聚类独立提取特征：逐点云点的信息$(p_{j} \oplus n_j)$ → 共享的两层FFN得到每个细分点的特征$\phi_j$ → 对属于$v_i$的所有点云点的特征做**scatter max pooling** → 聚合输出每个base mesh顶点$v_i$的特征向量$f_i$
        - 效果：feature dim=**1024**时效果跟ngf在同一细分rate（即一条边的细分点数，目前rate=**12**，ngf论文中测试时使用16）上差不多（见上次图）
        虽然feature dim远大于ngf，但实际上和sparse flex的model channel参数相同，sf是在获取feature后进行一次窗口注意力（此时feature dim还是1024），之后再线性投影（VAE）到隐变量 latent dim=16
    - 后续改动：
        - 在此基础上增加点云点的**法线信息**（和点的位置信息拼接送入）
        也可以加主曲率，但是计算有点耗费时间就先只加入法线信息了。
        - 之前提到的在encoder部分对顶点feature的集合进行多层窗口self attention的设计跑过似乎没有什么效果，没继续调。这部分其实是从sparse flex里抄来的，后来自己理解了一下，它用这个是因为体素的分辨率高，比方说一个褶皱上可能有上百个体素，共享特征可能让它能够学的更快？但是放我们这里可能用处就不大了。
        - **attention pooling替换scatter max：**
        Max Pooling 只取每个维度的最大值，等价于一个"硬注意力"，可能丢失了空间分布模式的不同，对于分辨率高的稀疏体素够用，但base mesh这种“稀疏”程度可能不够？所以引入attention加权聚合信息，同时多头注意力可以关注不同的几何特性。
            
            具体做法：引入H（H=8）个注意力头，计算同一cluster每个点在该注意力头上的分数（ln+softmax），然后对每个头有独立的value投影；拼接各头的feature和普通maxpooling得到的feature，再加一层linear投影回目标维度：
            
            1. 计算注意力分数并在cluster内softmax:
            $s_j=Linear(\phi_j) \in R^H$
            $\alpha^{(h)}_{j}=\frac{exp(s^{(h)_j})}{\sum_{k \in N(v_i)}{exp(s^{h}_k)}} ,h=1,...,H$
            2. 每个头独立value投影加权聚合：
            $val^{(h)}_j=W^{(h)}_V \phi_j$
            $f^{(h)}_i=\sum_{j\in N(v_i)} \alpha^{h}_j \cdot val^{(h)}_j$
            3. 保留max pooling（可选）和其他几个聚合结果拼接，加一层投影linear回到目标维度（我目前设置的是每个head128维）：
            $f_i=[f^{(1)}_i \oplus … \oplus f^{(H)}_i \oplus f^{(max)}_i]$
        
        下图为不同head上热力图可视化的注意力分数（不含max pooling），看起来不同注意头有关注到不同的变化。
        
        ![image.png](attachment:7bef18fc-5942-457b-854a-add6631f3a62:image.png)
        
        ![image.png](attachment:1f2e2acc-82ec-487b-adae-eca8bff5e1ce:image.png)
        
        - 加一个**VAE层**和对应的KL散度loss（解码器同理），压缩到**latent dim=40**
- **解码器：**
    - 和之前一样先融合feature再位移：
        1. 每个面片的三个顶点作为锚点$(p_{anchor_1},p_{anchor_2},p_{anchor_3})$，在该面片上的细分点lin，$p_{lin}$为其世界坐标系下坐标。
        2. 分别对每个的特征，拼接相对位置的位置编码，通过MLP_F预测该锚点在这个细分点贡献的特征。
        3. 最终聚合三个锚点在$p_{lin}$上的特征,与细分点的法向量（由三个锚点的法向量插值得到）的位置编码拼接再用一个MLP预测沿法线$\vec{n}_{lin}$的位移。
        $d=G[\sum_{i=1}^3F(f_i\oplus PE(p_{lin}-p_{anchor_i}))\oplus PE(\vec{n}_{lin})]$
        4. 共享边或者顶点上的细分点偏移值则参考Neural Progressive Mesh对从不同面上得到的偏移量取平均值。
            
            ![image.png](attachment:9cdba78f-d34a-441d-b2e1-5bf31b48871f:image.png)
            
- **训练：**
    - **新增：**参考[sparse flex](https://openaccess.thecvf.com/content/ICCV2025/supplemental/He_SparseFlex_High-Resolution_and_ICCV_2025_supplemental.pdf)，共四项组成render loss：
        
        $\mathcal{L}_{render}=\lambda_d\mathcal{L}_{d}+\lambda_n\mathcal{L}_{n}+\lambda_{ss}\mathcal{L}_{ss}+\lambda_{pl}\mathcal{L}_{lp}$
        
        L_d,L_n为**depth map**和normal map上和gt的L1 loss，L_ss和L_lp为normal map上的SSIM和LPIPS loss，参数（还没调过，直接参考的sparse flex）为：$\lambda_d=10.0,\lambda_n=4.0,\lambda_{ss}=\lambda_{lp}=0.5$
        
    - 其他可用策略：
        
        渐进式训练：训练过程中逐步增加细分rate
        
    - 实验结果：
    ngf我觉得调一调参并解决疑问1是能完全超过，但是sparse flex的分辨率太高了，感觉最多只能接近它的效果？
    （其他类似工作没开源的有[NESI](https://www.cs.ubc.ca/labs/imager/tr/2025/nesi/)、[neural progressive meshes](http://arxiv.org/abs/2308.05741)）
    但我们比sf好的两个优点就是压缩量（它的参数量约为600k左右）和恢复之后的mesh的面数（它恢复的fine mesh面数基本上都超过了1000k，我们的在rate=16时依然和THuman差不多数量级）
        
        base mesh使用从scan简化的网格，subdivision rate=12，attention pooling版本为8头注意力/每头128维拼接；max pooling版本为1024维，最终由VAE压缩到40维：
        |     base   |    GT     |  sparseflex  |  ngf(rate12)   |  ours(attn)  |  ours(max)   |
        
        ![图片2.png](attachment:14674566-5d6a-4ff8-b7e2-9ab428288033:图片2.png)
        
        ![图片4.png](attachment:a6a89471-a52c-4078-8810-994fdd01f20c:图片4.png)
        
        ![图片3.png](attachment:ec56e0ef-2563-471b-948a-a21b8233ace9:图片3.png)
        
    
    - 定量：
    指标：
        - chamfer distance: 随机采样50k个点
        - F-score（0.01、0.005）
        通过计算距离阈值 r 内的点对应关系的精度和召回率来重建精度。具体来说，我们报告形状归一化为 [−1, 1] 的 F 分数 (0.01) 和 F 分数 (0.005)。来自[Tanks and temples: benchmarking large-scale scene reconstruction](https://dl.acm.org/doi/10.1145/3072959.3073599)
        - normal render loss：50个随机视角
        （注：下列ours都是subdivision rate=12）