# CASE0001：真实治疗前几何的 TADPM Dataset 接入

本包是 **before-only 输入读取包**。CASE0001 上颌13颗、下颌14颗已经通过实际 `DentalDataset` 和官方 `utils.builder.dataset_builder` 的 DataLoader 加载验证。没有运行模型、DDIM或旧的Windows CUDA smoke test；无需Checkpoint。

与之前的接入包区别：**27颗牙都直接从本次CASE0001上传的source几何重新生成FPS和MAPS，不复用case_0几何/采样/重网格，不建立after文件、不建立GT旋转文件、不返回after或gt_params字段。** 以前的ZIP保持原样；当前阶段使用本包。

## 在已经完成 smoke 的 Windows 项目中读取

把 `CASE0001_before_only.zip` 解压到 **已有 `ToothGroupNetwork-tadpm-windows-smoke` 项目根目录**（包含 `main.py`、`datasets`、`config` 的目录）。应得到 `CASE0001_before_only\case_1`。不用重新部署，不用重跑任何smoke cell或网络推理。

在该项目根目录的PowerShell执行一条命令：

```powershell
& "D:\python\python.exe" ".\CASE0001_before_only\verify_case.py" --apply-dataset-patch
```

这条命令先添加 `DentalDataset` 的可选before-only模式，随后只执行输入读取和几何检查。Dataset原文件保存到项目 `.case_input_backups\DentalDataset_<唯一ID>.py`。补丁已应用则跳过；不需要Git，也不修改模型文件。补丁上下文不匹配时会停止且不写入源码，避免覆盖本机其他修改。

成功输出：`PASS CASE0001: upper13 + lower14 real teeth, no after/GT.` 报告写入 `case_1/reports/input_read_<唯一ID>.json`。每次使用新文件名，不覆盖旧报告。包内 `cloud_before_only_read.json` 是Linux CPU读取结果；用户之前的Windows CUDA `SMOKE_OK`继续作为既有网络兼容记录，本次没有重新执行Windows GPU测试。

如果包放在项目外，可显式指定：

```powershell
& "D:\python\python.exe" "路径\CASE0001_before_only\verify_case.py" --repo "已有smoke项目根目录" --case "路径\CASE0001_before_only\case_1" --apply-dataset-patch
```

补丁仅改 `datasets/DentalDataset.py`：新增 `config.before_only=True` 时的读取分支；使用官方原 `load_mesh_shape` 生成特征。默认不设置此标志，原paired训练/评测12项tuple和数学计算保持原样。before-only返回9个治疗前字段组成的dict；不能把它直接交给仍按paired tuple解包的 `main.py --test`，本阶段使用上述专用读取入口。

## 已检查的 smoke 来源与预处理规则

检查了 `tests/test_windows_smoke_compat.py`、`datasets/DentalDataset.py`、`utils/smoke.py`、现有case_0副本、`config/TADPM.yaml` 和官方预处理源码。详情在 `docs/smoke_input_audit.json`。

`test_windows_smoke_compat.py` 是兼容性单元测试，覆盖rotation6d、可选扩展导入、DDIM与官方数学等价性；它本身不是CASE0001数据转换器。其中DDIM测试使用随机context。本次没有运行这些旧测试。

原Windows smoke 的main入口通过 `configure_smoke_dataset` 将case root映射到以下文件，覆盖配置中 `*` 路径，并设 `train=False`、`bs=1`、`npoint=512`：

| 原paired smoke文件 | 作用/来源 |
|---|---|
| `case_0/case_ids.txt` | 整数病例ID，现有副本为0 |
| `case_0/remesh_before/0_<slot>.obj` | 历史上颌单牙，经QEM256+MAPS depth3成为16384个有序面 |
| `case_0/pcd_before512/0_<slot>.ply` | 历史单牙顶点FPS512，加第513行面积加权表面centroid |
| `case_0/pcd_after512_PLACEHOLDER/0_<slot>.ply` | 历史before副本，原smoke只为满足Dataset接口 |
| `case_0/param_PLACEHOLDER/0.npy` | (32,3,3) identity，原smoke只为满足Dataset接口 |

这些历史输入的上游来源是 `original_coordinates/initial_upper.obj` 和 `component_0~component_13.stl`；历史case_0只处理上颌13颗，居中使用上颌全模型surface centroid，除以40，下颌全空。这些历史文件仅用于核对格式，**没有复制到本包或用于当前CASE0001转换**。原配置 `config/TADPM.yaml` 的全局batch为24、占位路径、test.train=True，属于训练/评测模板；当前读取使用本包独立配置，不修改原config，也不采用这些默认值。

## CASE0001 的真实来源与坐标

真实输入来自用户上传CASE0001中的：

- `upper_predicted_source_coords.ply`、`lower_predicted_source_coords.ply`：治疗前全模型原source坐标网格。
- `upper_vertex_components.npy`、`lower_vertex_components.npy`：与原顶点序对应的component编号。
- 原TGNet `predictions/CASE0001_upper.json`、`CASE0001_lower.json` 和 `component_fdi_candidates.csv`：原始预测记录，保留字节不变。
- 已批准的独立 `reviewed_component_fdi.csv`：只用于选择32槽位，不修改原预测。

转换前核对source mesh与原TGNet模型坐标OBJ的面序和坐标变换一致；没有混合上下颌各自旋转后的TGNet坐标系。沿用共同source坐标框架，先合并上下颌全模型，再按官方 `(v - merged_mesh.centroid) / 40.0` 处理，最后分牙。中心为 `[-0.9359576121403641,29.513620472768093,0.5265130647159325]`，为面积加权表面centroid。归一化矩阵及逆矩阵保存于CASE0001准备报告；这是真实坐标变换记录，不是治疗GT。

牙龈参与官方全模型中心计算，但不进入网络牙齿槽位；原坐标牙龈独立保存在 `background/CASE0001/`。原坐标27颗单牙保存于 `source_components/CASE0001/`，原顶点索引保存在 `source_vertex_indices/CASE0001/`；原全模型及原预测在 `original_reference/CASE0001/`。全程没有改写上传原件。

每颗牙：真实component → 全局中心 `/40` 单牙OBJ → 真实单牙顶点FPS512 + centroid → QEM256 → 官方MAPS depth3（16384面）。随机种子固定并逐牙记录。已经封闭且绕向一致，所以未运行Manifold修复。所有改动是新副本；点云没有再次归一化。

## 牙位索引与缺牙

| 固定slot范围 | FDI顺序 |
|---|---|
| 上颌0–15 | 18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28 |
| 下颌16–31 | 48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38 |

上颌C1–C13保留17,16,14,13,12,11,21,22,23,24,25,26,27。下颌C1–C14使用独立三维复核候选37,36,35,34,33,32,31,41,42,43,44,45,46,47；C10保留43（slot21），C11原43→候选44（slot20），C12原44→候选45（slot19）。详细32行对照在 `slot_manifest.csv`。

mask=1有27个；mask=0为 **slot0、3、15、16、31**，对应输入未表示FDI18、15、28、48、38。FDI15是用户指定缺牙复核候选，其他空槽不作临床缺牙诊断。全部编号仍是三维复核候选，未经临床最终确认。

before-only模式仅在OBJ+before PLY同时存在时启用槽位；两者都不存在时槽位为0，相关mesh特征、点云、centroid、Fs全零；仅缺其中一个文件会明确报错，不会静默当缺牙。没有缺牙的OBJ/PLY假文件，没有identity旋转，没有GT占位向量。32槽不压缩、不顺移。`masks/1.npy`是审计mask，实际Dataset仍根据真实文件生成mask；检查脚本核对两者一致。

## Tensor shape（batch=1）

| 字段 | shape | 来源 |
|---|---|---|
| index | (1,) | `case_ids.txt`，本病例整数1 |
| feats | (1,32,10,256,64) | remesh面area1、排序face_angles3、排序curvs3、normal3 |
| center | (1,32,256,64,3) | 每个三角面3顶点均值，经官方patch重排 |
| cordinates | (1,32,256,64,9) | 每面的3个顶点XYZ，官方拼写保留 |
| faces | (1,32,256,64,3) | 保存的有序面顶点索引，经官方patch重排 |
| Fs | (1,32) | 存在牙为16384，空槽为0 |
| before_points | (1,32,512,3) | before PLY前512行 |
| centroid | (1,32,3) | before PLY第513行 |
| masks | (1,32) | OBJ+PLY存在判定 |

单样本去掉最前面的1。保持原Dataset dtype：index经DataLoader为int64，masks为int32，其余为float64（faces也在原浮点容器中）；正式runner原本会转换为float32。本阶段不进入runner。after_points、after_centroid、gt_params **没有返回**，不以None/全零/identity假装存在。

## 批量复用转换代码

读取已完成的数据不需要重新预处理。要处理后续同格式TGNet导出，使用 `convert_cases.py`；它没有 `--history` 或旧case路径参数，拒绝覆盖已存在输出目录。

```powershell
& "D:\python\python.exe" ".\CASE0001_before_only\convert_cases.py" --source "D:\CLK\TGNet\outputs\CASE0002" --reviewed-csv "D:\CLK\TGNet\outputs\CASE0002\reviewed_component_fdi.csv" --repo "." --out ".\new_cases\CASE0002_before_only" --dataset-id 2
```

后续病例需要与本次相同的TGNet导出布局及与组件对应的复核CSV；不会把当前FDI纠正硬套到别的病例。转换需要已有numpy/open3d/trimesh/scipy、pyfqmr和官方MAPS的普通CPU依赖；**本次读取只使用已有smoke环境，不需要安装预处理依赖或编译CUDA扩展**。

批量manifest为JSON列表，例如（路径相对manifest所在目录）：

```json
[
  {"source":"CASE0001","reviewed_csv":"CASE0001/reviewed_component_fdi.csv","case_name":"CASE0001","dataset_id":1},
  {"source":"CASE0002","reviewed_csv":"CASE0002/reviewed_component_fdi.csv","case_name":"CASE0002","dataset_id":2}
]
```

```powershell
& "D:\python\python.exe" ".\CASE0001_before_only\convert_cases.py" --batch-manifest ".\cases.json" --repo "." --out ".\new_cases\batch_before_only"
& "D:\python\python.exe" ".\CASE0001_before_only\verify_case.py" --repo "." --case ".\new_cases\batch_before_only"
```

各病例使用唯一整数ID，共享官方before文件夹，独立保存原件、mask和来源报告。检查脚本逐病例核对mask、张量、法向/绕向、FPS索引、归一化、patch面序以及原始预测SHA256；拒绝读case_0/after/GT目录，并拒绝导入网络及runner。报告文件名唯一。

## 本次验证记录

- 实际 `DentalDataset(before_only=True)` → 官方DataLoader：CASE0001读取PASS。
- 全27颗来源、几何及采样检查通过；上下颌13+14；没有after/GT字段或文件读取。
- 新增4项Dataset边界测试通过：无after/旋转的缺槽读取、不完整文件报错、拒绝训练/after配置、官方mesh特征和paired __getitem__逻辑保持不变。
- 补丁安装器经过CRLF、原文件备份和重复应用验证；Python3.10语法检查通过。
- 没有部署、训练、创建模型权重、导入TADPM网络、模型forward、DDIM或GPU推理。本阶段成功标准已经达到；正式排牙需要权重属于下一阶段。
