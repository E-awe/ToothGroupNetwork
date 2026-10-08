# CASE0001：TGNet → 官方 TADPM 单病例输入

已完成 **真实官方 DentalDataset / DataLoader 读取测试**，上颌13颗、下颌14颗进入32个固定槽位。未执行模型 forward、DDIM、GPU推理、训练或排牙。病例编号 CASE0001 在 Dataset 中使用整数索引 `1`，文件名为 `1_<slot>.obj/.ply`。没有覆盖原始文件或历史上颌缓存。

## 下载后在 Windows 做读取检查

将本包解压到 `D:\CLK\TADPM\TADPM_official\`，得到 `CASE0001_TADPM_input\case_1` 和同级脚本。

在 PowerShell 运行这一条命令；**只读取，不运行网络，也不要求 checkpoint**：

```powershell
& "D:\python\python.exe" "D:\CLK\TADPM\TADPM_official\CASE0001_TADPM_input\read_case.py" --repo "D:\CLK\TADPM\TADPM_official" --case "D:\CLK\TADPM\TADPM_official\CASE0001_TADPM_input\case_1"
```

成功应输出 `PASS: official Dataset/DataLoader, 27/32 slots (upper13 + lower14)`。每次生成新的 `case_1/reports/input_read_<随机ID>.json`，记录本机解释器、软件、GPU信息和全部检查结果；不会覆盖以前报告。云端已经完成的报告为 `case_1/reports/cloud_input_read.json`，它属于 Linux CPU 环境，**不代表 Windows CUDA 已经测试**。

读取脚本使用 TADPM 仓库里的实际 Dataset 和 `utils.builder.dataset_builder`，不是自制替代 Dataset。已验证的官方兼容版本为 E-awe/ToothGroupNetwork 的 `tadpm-windows-smoke` 分支，commit `3523d8e1db6a9136e3224165d68f5f1a6a478599`，基于 lcshhh/TADPM `92967a68191846ab38dcdb2ae7914e19ecd0d396`。

本包 `compat/tadpm-windows-smoke.patch` 是**之前已完成的同一兼容补丁**，供缺少该补丁的官方项目使用；本次没有改变网络。若本机已经应用，不要重复应用。若仍出现 pytorch3d / chamfer 导入错误，先核对本机与上述兼容版本：补丁提供纯 PyTorch rotation6d、可选 Chamfer 导入、设备兼容和原先随机权重 smoke 入口。它保留正式 checkpoint 严格加载与网络参数结构。

依赖核查：已有 PyTorch 2.5.1+cu121 应配 torchvision 0.20.1；读取还需要 open3d、numpy、trimesh、scipy、easydict、PyYAML、timm、einops、matplotlib。`requirements-input-read.txt` 不安装或替换 torch/torchvision。预处理结果已经在包中，不需要本机重新安装 pyfqmr、MAPS、Manifold 或编译 Chamfer/pointops。仅为了读取，不需要 CUDA 扩展。

## 32槽位、独立复核候选与 mask

| 槽位范围 | FDI顺序 |
|---|---|
| 上颌0–15 | 18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28 |
| 下颌16–31 | 48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38 |

mask=1 共27个，mask=0 为 **slot 0、3、15、16、31**，对应没有输入牙网格的 **FDI18、15、28、48、38**。FDI15 是用户指定的缺牙复核候选；其余空槽仅记录“本次输入未表示该牙位”，不能据此确诊缺牙。所有编号均为三维人工复核候选，**未经临床记录最终确认**，无牙位顺移。

上颌C1–C13保留17,16,14,13,12,11,21,22,23,24,25,26,27。

下颌C1–C14候选为37,36,35,34,33,32,31,41,42,43,44,45,46,47。C10保留43；**C11原始预测43→独立候选44（slot20），C12原始预测44→独立候选45（slot19）**。其余不变。原始下颌C10/C11都是43，不能唯一占据固定槽位；这里使用此前用户批准的独立候选表完成接入，不修改原始预测 JSON。

`case_1/tadpm_slot_manifest.csv` 同时保存原始编号、候选编号、component、slot和mask。完整独立复核表及原始预测 CSV/JSON 保存在 `original_reference/`。这些原始文件均逐个核对SHA256，保持字节不变。

## 几何、复用和归一化

使用上传 CASE0001 中的共同 **source coordinates** 网格；没有把上下颌各自旋转后的 TGNet model coordinates 混为一组。原始 source→TGNet 变换保留于 `original_reference/source_geometry.json`。本次没有添加任何旋转。

官方 `get_mesh.py` 先合并上下颌全模型，然后 `(vertices - mesh.centroid) / 40.0`，最后提取单牙。这里同样以含牙龈的上下颌全模型面积加权表面 centroid 统一居中、除以40；牙龈仅参与全局中心计算，**不进入任何牙槽位**。归一化矩阵、逆矩阵、实际中心及上颌缓存的平移量均写入 `reports/preparation.json`。未声明 OBJ 单位为mm；输入必须和权重训练数据使用相容的尺度/方向，正式推理时需向权重提供者核实。

历史上颌 `case_0` 与 CASE0001 source 上颌三角面完全相同，顶点差异小于1e-8。全部13颗的单牙网格、256面 QEM 网格、16384面 MAPS 层级和512个采样点被复用到副本；只按新上下颌公共中心进行统一平移。未重新对上颌采样或重网格。每颗原始 STL 与对应CASE0001 component也经过顶点/三角面集合核对。

下颌14颗按官方流程生成：单牙归一化 → 顶点 FPS512 → QEM256面 → MAPS depth3（16384面）。随机种子按slot固定。全部27颗读取时均验证封闭性、绕向一致、有限法向、非退化面、面序和采样索引。每颗PLY有 **513行**：前512行为FPS点，第513行为面积加权单牙表面 centroid；沿用 `/40` 坐标，没有再次归一化。

`source_components/` 保存27颗未经归一化的原坐标单牙OBJ副本；`source_vertex_indices/` 给出各component对应原source网格顶点索引；`background/` 单独保存上下颌牙龈。`original_reference/*_source_coords.ply` 保留上传的全模型原坐标文件。

## 目录与真实 after 的区别

```text
case_1/
  case_ids.txt                      # 整数1
  dataset_config.json               # 可移植相对路径，train=False、bs=1
  tadpm_slot_manifest.csv            # 32槽位，原始编号与独立候选
  mask.npy                          # (32,) 审计文件
  single_before/1_<slot>.obj         # 原细节归一化单牙
  simplify_before/1_<slot>.obj       # 256面QEM
  remesh_before/1_<slot>.obj         # 16384面，Dataset实际读取
  pcd_before512/1_<slot>.ply         # 512点+centroid
  pcd_after512_PLACEHOLDER/          # before的字节副本，只为Dataset读取
  param_PLACEHOLDER/1.npy            # (32,3,3) 全部identity，只为读取
  sampling_indices/                 # FPS对应single_before顶点索引
  source_components/                # 原坐标单牙副本
  source_vertex_indices/            # component→原source顶点
  background/                       # 原坐标牙龈副本，独立保存
  original_reference/               # 原预测、复核表、坐标与来源记录
  reports/                          # 预处理、模型入口审计、实际读取检查
```

**本病例没有真实after数据。** 官方 Dataset 只有在 OBJ、before PLY、after PLY 同时存在时才令mask=1，而且总会加载 `.npy`。因此占位 after 和 identity 参数只解决读取格式，不代表预测或治疗效果。官方 Dataset 不读取审计 `mask.npy`，实际mask来自这些文件；读取测试已核对两者一致。缺牙槽位没有OBJ/PLY，几何、点云、centroid、Fs为0；`.npy`仍保留32个identity，故缺牙 `gt_params` 为 `[0,0,0,1,0,0,0,1,0]`，不是全9个0。

官方 `register.py` 对中心化 before/after 顶点计算Kabsch旋转；失败则ICP，最终保存 `R.T`，总shape为 `(32,3,3)`，不是4×4位姿。平移不在该npy中：Dataset的gt_params前3项取真实after centroid，后6项取旋转矩阵前两行。没有真实after时不能生成真实注册参数、训练目标或ADD评价。

## 实际 Dataset tensor shape

下表为 batch_size=1（单样本去掉最前面的1）：

| 返回项 | batch shape |
|---|---|
| index | (1,) |
| feats | (1,32,10,256,64) |
| center | (1,32,256,64,3) |
| cordinates（官方拼写） | (1,32,256,64,9) |
| faces | (1,32,256,64,3) |
| Fs | (1,32) |
| before_points、after_points | (1,32,512,3) |
| centroid、after_centroid | (1,32,3) |
| gt_params | (1,32,9) |
| masks | (1,32) |

原 Dataset 大多数浮点结果为float64（faces也在浮点容器中），masks为int32；runner会转float32。每颗网格256个patch，每patch64个面，面序不是随意排列：读取检查验证官方重排与文件面序一致。mask并没有作为模型attention mask传入网络；空槽按官方机制保留零输入，不能压缩为27槽。

## 是否可以只有 before 做正式推理

从源码看 **可以**：`TADPM.forward` 的condition只由治疗前mesh、点云、centroid构成；eval模式的 `diffusion.forward` 不使用 `gt_6dof`，直接调用 `ddim_sample(features)`。after主要用于Dataset接口、训练目标和测试指标。

但 `main.py --test --ckpts ...` 默认是带after的评测入口，会计算ADD，不能把复制before当真实after使用。为后续保留 `predict_before_only.py`：不改网络/condition/DDIM，只绕过假after的指标计算，严格加载完整权重并保存原始32×9输出参数；**默认只检查checkpoint，必须显式传 `--run` 才会运行GPU**。本次没有执行这个预测入口，不保证未经测试的本机GPU运行已经成功。

可先在 Windows 只检查候选权重（请将示例文件名改成真实已训练的完整TADPM权重路径）：

```powershell
& "D:\python\python.exe" "D:\CLK\TADPM\TADPM_official\CASE0001_TADPM_input\predict_before_only.py" --repo "D:\CLK\TADPM\TADPM_official" --case "D:\CLK\TADPM\TADPM_official\CASE0001_TADPM_input\case_1" --checkpoint "D:\CLK\TADPM\TADPM_official\checkpoints\TADPM_trained.pth"
```

它要求官方 `base_model` 状态字典全部628个tensor键及shape匹配（181,705,301参数），在meta设备核对结构，不运行forward。键匹配只能证明兼容性，**不能证明这些参数训练过**，还需权重提供者的训练来源记录。整网权重包含Mesh encoder、全局encoder、transformer、diffusion/U-ViT；仅MeshMAE预训练权重不够，TGNet的h5和随机smoke的pt也不够。

## 本次结论与剩余条件

1. 单病例32槽位输入及云端官方读取检查已完成，可以开始本机输入读取；无需重部署TGNet、重做上颌13颗或运行长时间GPU推理。
2. 云端/上传包没有完整训练TADPM checkpoint；官方README没有提供其下载链接。**正式排牙的当前模型阻塞项是完整、训练完成且来源可核实的TADPM权重。** 未访问用户Windows磁盘，不能断言本机也没有权重。
3. Windows读取尚待运行上面的单条命令；GPU checkpoint-backed预测尚未执行。后续得到权重后，可使用单独的before-only入口，不需要编造真实after；没有真实after就不能报告ADD。
4. 独立FDI候选仍需临床记录最终确认；真实输入单位、坐标方向和训练约定还需跟权重来源核实，不能把结构/读取通过解释为有临床意义的排牙。

复现预处理脚本 `prepare_case32.py` 只在需要重建新副本时使用，要求原CASE0001、已保存的历史上颌case_0及官方仓库。它拒绝覆盖输出目录。本次交付中已含完整27颗结果，正常接续工作只运行读取检查。

