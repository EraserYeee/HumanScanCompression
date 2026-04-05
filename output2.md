<aside>
💡

去掉VAE和法线方向限制后，3DShape2VecSet的迁移尝试效果不佳，但+SAPIENS2的loss后顶点变化有一定改善，但由于变化不均效果依然不佳。

</aside>

- 数据重新处理：
    - 只使用THuman2.0，且取消归一化
- 之前做法的微调：
    - 去掉VAE：高频和突出处mesh严重不匹配的问题依然没有解决
    - 使用xyz位移：发现配合SAPIENS2的法线梯度loss有一点点改善突出处mesh不匹配的问题，但是没有完全解决（见后文SAPIENS2loss部分）
    - 使用线性插值聚合feature，但对于重心坐标为(u,v_1,w_1)和(u,v_2,w_2)的位置不同的不同细分点，它获得的u对应的feature是一样的；对于两个不同大小的三角形，在它们中重心坐标相同的细分点实际上和feature锚点的距离不同却也会得到相同的feature，因此训练不出任何效果。（后文均未应用这个改动）
        
        $d=G[\sum_{i=1}^3F(f_i\oplus PE(p_{lin}-p_{anchor_i}))] 
        \rightarrow d=G[\sum_{i=1}^3 \lambda_i f_i]$
        
    - 去除LPIPSloss、之前的注意力模式改为单头注意力（稍微比多头差一点但是差别不太大）；数据加载部分加速了一点配合并行可以大半天训一次。
    - EdgeRunner生成的base mesh并没有比QEM简化质量更好
    - 效果：之前的两个问题依然没有解决，定性的效果和上一次区别不大，定量也仅有微小提升。
        
        
        | （下列都未使用VAE） | CD (*10e5)↓ | Normal L1↓ | F1 (0.01)↑ | F1 (0.001)↑ |
        | --- | --- | --- | --- | --- |
        | ngf (rate=12) | 4.323 | 0.01718 | 98.859 | 2.539 |
        | ours（attentive pooling） | 3.674 | 0.01412 | 95.002 | 4.010 |
        | ours（cross attention） | 4.009 | 0.01561 | 94.951 | 3.895 |
        | ours（attentive pooling+normal gradient loss） | 3.190 | 0.01380 | 98.027 | 4.786 |
- 3DShape2VecSet的迁移尝试：
    - 做法1：照抄原做法，QK保留绝对坐标输入，用额外训练的PointEmbed编码QK，只用于计算attn score，并保留聚类作为注意力 mask。因为没有旋转所以这样看起来似乎没有问题。但需要PointEmbed处理绝对位置emb，而且cluster内所有的scan点和锚点的绝对位置很接近，所以失败了：
        
        输入:
        local_points: (B, P, 6)    ← [相对偏移, 法线]（世界坐标系）
        scan_points:  (B, P, 3)    ← 绝对坐标，（世界坐标系 ，新增传入）
        base_verts:   (B, V, 3)    ← 绝对坐标（世界坐标系）
        cluster_idx:  (B, P)
        
        Step 1: PointNet
        point_feats = **PointNet**(local_points)          → (B*P, D)   [局部几何特征, 用作 V]
        
        Step 2: 获取QKV
        vertex_embeds = **PointEmbed**(base_verts)        → (B**V, D)*   [**绝对位置**嵌入, 用作 Q]
        Q = Wq(norm_q(vertex_embeds))
        
        **scan_embeds   = **PointEmbed**(scan_points)       → (B*P, D)   [**绝对位置**嵌入, 用作 K]
        K = Wk(norm_k(scan_embeds))
        
        V = Wv(norm_v(point_feats))         → (B*P, D)   [**局部**几何特征, 用作 V]
        **同一坐标系、同一 PointEmbed的Q、K语义一致**
        
        Step 2: Cluster-Restricted Cross-Attention
        q_per_point = Q[cluster_idx]
        score = (q_per_point · K) / √d_head           → 绝对位置·绝对位置
        attn_out = scatter_add(scatter_softmax(score, cluster_idx)* V, cluster_idx) → (B*V, D)
        
        Step 4: 残差 + FFN
        output = Wout(attn_out)
        output = output + ReLU_FFN(norm_ffn(output))
        
        输出: (B, V, D)
        
    - 做法2：QK也使用局部坐标系 → 即所有顶点的Q完全相同（原点），其实还是退回了我上次的做法；效果和我的做法没什么差别。
- 引入SAPIENS2的法线梯度loss：
    - 做法：$L = (1 - \hat{N}\cdot N) + ||\hat{N} - N||^2 + ||∇\hat{N} - ∇N||^2$
    - 出现问题：直接算的时候，当base和gt渲染出的轮廓差别较大的时候，如果没有别的视角，容易在导致mesh不断延长（常出现在脚底这样比较少视角训练到的位置，如第二行图），比如在mesh凸出，gt为背景的像素上，该loss中前两项梯度为0，对凸出区域没有”推回去“的力。因此，目前比较简单的只在AND区域计算该loss 。
    |          GT normal        |   GT ∇N（xy平面上） |           ours normal       |             ours ∇N         |
        
        ![image.png](attachment:ca0434dd-ed5d-4c6a-8a0b-247abb95f567:image.png)
        
        ![38419f92b4c1ab05eba1c088be56eff6.png](attachment:9f903c4d-8d89-453c-a9b3-b9b978dd36d7:38419f92b4c1ab05eba1c088be56eff6.png)
        
    - 实验结果：
    相比没有使用该loss，凸出处（即base mesh和gt相差较大的地方）的细分点偏移幅度更大，但是出现了细分点偏移不均的情况（如下图），因此结果依然形变不到位。我认为这并不是这个loss的问题，而是模型可能并没有学到对应的“分布”。
    如果按照法线方向偏移可能是解决的方法之一，但沿法线位移时凸出处偏移的效果又不佳。
    |              GT               |           base mesh            |        ours         |
        
        ![image.png](attachment:0ad1789f-5f08-433c-af71-f5110793b021:image.png)
        
        ![image.png](attachment:98dbab6a-a695-4ccd-9863-7f49363b44c7:image.png)
        

考虑的改进方向：

- 在采样点云时，参考Dora做一个重要性采样进行叠加（下图(a)，编码器也可以参照其做法修改应用？）；
    
    ![image.png](attachment:b8292634-d6cc-47de-b828-c683eb12fb3c:image.png)
    
- 将当前方式中的聚类中心base mesh顶点$v_i$改为面片$face_i$的重心$c_i$。对于点云点j,需要编码的信息从$(p_{j} \oplus n_j)$，改为沿面法线方向投影到面片的重心坐标插值$c_i^j$ +法线方向的偏移量$d_j$+点云点法向量和面法线的差:  $(c_i^j\oplus  d_j \oplus n_j - n_i)$；这种显式的”对应“（点云点 - base 面上的重心坐标）应该能改善偏移不均的问题？而且能够改善面上高频细节难以还原的问题，因为无需额外的MLP聚合三个锚点的feature。
- 感觉让细分点一次性位移到正确的位置真的有点难做，我觉得要么是有提供很好的对应关系（例如上一点的重心坐标与点云点对应）、要么是$d=G[\sum_{i=1}^3F(f_i\oplus PE(p_{lin}-p_{anchor_i}))]$ 这里的F………

![image.png](attachment:3e3b7df5-dbc2-46e7-860d-fed3f106ace6:image.png)