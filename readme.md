## 项目概述  

### 尝试 HDBSCAN

密度聚类，难以控制聚合程度（最终至少有两组的要求）。此外，噪声点过多，而我希望所有人的动向都被考虑到，单人不成组的独占一组。

代码在 `pedestrian_group_clustering_agnes.ipynb`，未完成。

### 切换至 AGNES

AGNES 始于单人组，不断树形合并的算法流程更加适合此任务。因此考虑在单帧上执行聚类，然后在时间维度上做跨帧匹配以保持组 id 的一致性。核心目标：如果新生成的 group 能匹配到已有的历史 group 则复用历史 id；匹配不到则分配新的 group id。匹配主要参考 `group_members` 在上一时刻所属的旧 `group_id`.
  
代码主要在 `pedestrian_group_clustering_agnes.ipynb` 中，数据集为 `students003.txt`，输出目录为 `output/`。
  
## 总体思路  

- 每帧对当前可见的行人做特征化和单帧聚类，特征提取了当前位置、当前速度（和上一时刻位置作差得到）  、历史位置（`window` 可调）。
- 对每个新生成的簇（`new_group_ids`），通过簇内成员在历史帧中归属的簇“投票”众数来决定它对应哪个历史 group。  
  
## 关键实现细节 

- 预处理：  
  - 计算速度特征 `vx`, `vy`，以及位置历史特征（按 `pos_history_window` 的偏移）。  
  - 对位置、速度、历史特征分别做标准化，速度和历史特征乘以权重 `v_alpha` 与 `history_alpha`。  
- 单帧聚类：  
  - 使用 `AgglomerativeClustering`（distance\_threshold 或最小簇数 `min_clusters` 的 fallback）。  
  - 返回每个观测点的 `local_cluster` 标签。  
- 跨帧匹配（`process_all_frames`）的核心流程：  
  1. 为每个行人维护 `past_affiliation`：`pid -> deque(maxlen=vote_history_window)`（仅保存最近的 group ids，若需要基于时间窗则保存 `(time, group_id)`）。  
  2. 每帧得到 `clustered_frame` 后，汇总当前簇成员的历史 group 投票集合 `past_group_scope`。  
  3. 只为有投票的簇构造投票行（`vote_dict_rows`），同时记录该行对应的新簇索引（`voter_rows`）。
  4. 构造矩阵 `matching_matrix`（行：有投票的新簇，列：历史 group id）。调用 `scipy` 的 `linear_sum_assignment` 获得最大匹配。
  5. 过滤匹配结果中票数为 0 的配对（视为未匹配），将匹配映射回原簇索引并分配历史 id；未匹配簇分配新的全局 group id。  
  6. 更新每个成员的 `past_affiliation` 并记录输出行（`time,id,group_id`）。  
  
## 已发现的坑与解决方法  

- 原始方案中，新 `group_id` 由新 `group_members` 所属的旧 `group_id` 众数投票得来。但这种方案的问题是：原来在同一组的行人，如果后来被分到不同的组，有可能导致两个新组的 `group_id` 都继承自同一个旧组，导致错误。
  解决方案：使用二分图最大权匹配（匈牙利算法），确保一个旧组只能分配给一个新组
- 坑：匹配矩阵的行索引与簇列表索引不对齐，导致把匹配结果应用到错误的簇上。    
解决：只为有投票的簇构造矩阵行，同时维护从矩阵行索引 -> 簇索引的映射数组（如 `voter_rows`），匹配结果回映射时使用该映射。  
- 坑：匈牙利算法会强制匹配行列，若票数全为 0 会产生无意义匹配。    
解决：在接受每个配对前检查 `matching_matrix[r, c] > 0`，否则视为未匹配。  
- 坑：`deque(maxlen=...)` 只按条数保留历史，无法按时间窗口过期（若帧率不恒定或需要精确时间窗口）。    
解决：若需要按时间窗口投票，应在 `deque` 中保存 `(time, group_id)`，并在每帧开始前剔除过期条目（基于当前 `t` 与 `vote_history_window`）。  
- 坑：当没有历史 group（首帧）或 `matching_rows` 为空时要显式处理，避免构造空矩阵或索引错误。    
解决：加判断 `if len(matching_rows)>0 and len(past_group_scope)>0:`
  
## 主要函数说明  
- `cluster_single_frame(frame, cfg)`    
输入单帧数据，提取标准化特征并运行 `AgglomerativeClustering`，返回带 `local_cluster` 的 DataFrame。  
- `process_all_frames(df, cfg)`    
按时间遍历数据：每帧聚类、收集投票、建立匹配矩阵、运行匈牙利算法、筛选正票匹配、分配 group id、更新 `past_affiliation`、输出结果表。  
  
## 参数说明（`ClusterConfig`）  

- `v_alpha`: 速度特征权重  
- `history_alpha`: 历史位置特征权重  
- `min_clusters`: 最少簇数（fallback）  
- `distance_threshold`: 层次聚类距离阈值  
- `pos_history_window`: 用于生成历史位置特征的时间跨度  
- `vote_history_window`: 投票历史长度（若按时间窗口请改为保存 `(time, id)`）  
- `linkage`: 聚类 linkage 方法  

### 参数选择

经过大量实验和超参数调整，发现 `v_alpha` 不可以调得太大，否则会导致距离较远的同向行走的行人被归为一组；也不可以设置得过小，否则两组相向而行的行人在互相穿过时会被归为一组。

此外 `history_alpha` 和 `pos_history_window` 也不可以设置得过大，否则会冲淡 `x,y,vx,vy` 这几个关键指标，导致混乱；并且，`history` 中维护的==应当是绝对位置==而不是两时刻位置差，否则会导致更多无关点被聚合。

`linkage` 选用了 `complete` 而非 `ward`，允许组的成员个数更加分散（可以有大组也可以有小组），同时避免长条形延伸的组。


## 结果与可视化  

- 借助 matplotlib 实现逐帧聚类结果可视化
- 使用 ffmpeg 聚合成视频

### 具体结果

聚类结果见 `group_labels.csv`，可视化见 `output/`.

其中 `v2.mp4` 使用的参数如下：

```py
v_alpha: float = 0.5  
history_alpha: float = 0.3  
min_clusters: int = 2  
distance_threshold: float = 0.7  
pos_history_window: int = 10  
vote_history_window: int = 5
linkage: str = "ward"
```

`v4.mp4` 使用的参数如下：

```py
v_alpha: float = 1  
history_alpha: float = 0.1  
min_clusters: int = 2  
distance_threshold: float = 1.4  
pos_history_window: int = 3  
vote_history_window: int = 6  
linkage: str = "complete"
```
  
## 可改进方向  

- 将 `past_affiliation` 的存储从简单 `group_id` 扩展为 `(time, group_id)`，并在每帧剔除过期条目以实现真正的时间窗投票。  
- 在匹配中加入空间距离或其他连贯性置信度作为第二因子，结合投票数构建复合打分（比如加权票数 - (lambda)距离）。  
- 处理簇的分裂与合并事件：监测一个历史 group 被多个新簇匹配或多个历史 group 匯合为一个新簇，记录 split/merge 事件并用更复杂的策略处理 id 继承。  
- 使用 Kalman Filter / 轨迹匹配提升跨帧连接鲁棒性，尤其在遮挡或检测丢失场景中。
- 使用原生时空聚类，将前后两帧中的行人一同聚类，自动给出聚类在时间方向的延伸方式。
