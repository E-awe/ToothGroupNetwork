本分支基于官方 lcshhh/TADPM 提交 92967a68191846ab38dcdb2ae7914e19ecd0d396。
发布仓库为已连接的 E-awe/ToothGroupNetwork，分支 tadpm-windows-smoke；此分支目录结构是官方 TADPM。

目标：Windows / PyTorch 2.5.1+cu121 / RTX 4060 Laptop 上运行随机权重完整 forward。
没有 TADPM checkpoint；输出没有临床意义，也不是治疗后真值。

准备：
- 使用本分支完整源码，或把 compat/tadpm-windows-smoke.patch 应用于上述官方提交。
- 有本地手工修改时，先运行 git apply --check；不要强制覆盖手工文件。
- 保留现有 torch 2.5.1+cu121。建议 Python 3.10–3.12（必须由 D:\python\python.exe 实际支持这些包）。
- 现有 tadpm_official_upper_input.zip 直接解压到本分支根目录，使 case_0/remesh_before 等目录存在。不重新生成病例。
- 不安装官方 requirements.txt 的冲突重复项，不编译 pointops 或 chamfer。

缺少依赖时，在 PowerShell 安装：
```powershell
& "D:\python\python.exe" -m pip install "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
& "D:\python\python.exe" -m pip install -r requirements-windows-smoke.txt
```
torchvision 应与 torch 同为 cu121 构建。PyTorch3D、chamfer、xformers、comet_ml、wandb 不属于此 test 路径的必需依赖。

在 TADPM 根目录执行唯一 smoke inference 命令：
```powershell
& "D:\python\python.exe" main.py --config config/TADPM.yaml --save_root ./smoke_runs --exp_name case0_smoke --test --num_workers 0
```
默认设备为 CUDA，默认输入为 ./case_0。输入在其他目录时添加 --case_root "D:\完整路径\case_0"。
无 --ckpts 自动进入 smoke，test Dataset 的 train=False、batch=1，不做随机数据增强或 drop_last。
模型结构、dim=768、depth=12、condition 构造、U-ViT、100步DDIM/eta=1全部保持官方逻辑。
模型不会利用 mask 跳过缺牙槽位；32槽位完整 forward，mask 随输出保存。

成功标志：
```text
SMOKE_OK: DentalDataset -> TADPM -> Diffusion/UViT -> DDIM (100 steps)
```
输出：smoke_runs/test_case0_smoke/smoke_report.json 和 smoke_batch_000.pt。
pt 中 outputs 为 (1,32,9)，前三维为模型输出的归一化中心，后六维为行约定的 rotation6d；masks 为 (1,32)。
输出全部32个槽位，缺牙/下颌槽位由 masks 判断有效性。这不是 FDI segmentation JSON。
after PLY 与旋转参数是现有输入包的加载占位，smoke 不计算 ADD、不报告治疗效果。
有 --ckpts 时仍使用原 builder.load_model_from_ckpt 的 strict=True 加载与原测试指标路径；正式训练路径保留。
正式 checkpoint 推理请配置真实数据路径；smoke 的 case_0 自动路径覆盖仅在无 checkpoint 时生效。

兼容修改：
- utils/__init__.py：明确本地包，解决 site-packages/utils 冲突。
- utils/rotation6d.py：与 PyTorch3D 同约定的首两行展开及 Gram-Schmidt，保留 batch/dtype/device/autograd。
- datasets/DentalDataset.py、runners/test_tadpm.py、runners/train_tadpm.py：替换 rotation6d import。
- 三个 runner（含 pretrain）：移除没有调用过的 pytorch3d.loss.chamfer_distance import。
- test/train runner：移除未使用的 torchvision.transforms import；timm 仍需要匹配的 torchvision。
- models/MeshMAE.py：捕获 Chamfer ImportError/OSError；仅实例化 Mesh_mae 才报错，Mesh_encoder 保持可用。
- models/MeshMAE.py：encoder 零填充跟随输入设备，参数和特征构造不变。
- models/diffusion.py：设备转移跟随输入；schedule 注册为 persistent=False buffers，不增加 checkpoint keys。DDIM公式不变。
- main.py：增加可选 --case_root 和 --device，明确导入 Path；默认仍 CUDA。
- utils/config.py：用 shutil.copy2 替代 Windows 不可用的 shell cp。
- config/TADPM.yaml：对原来的 * 占位路径加引号，让 YAML 能解析；正式训练参数不变。
- utils/smoke.py：现有病例路径解析、32槽位输出/有限值检查及 forward 调用次数报告。
- requirements-windows-smoke.txt、tests/test_windows_smoke_compat.py：明确依赖和可执行回归检查。

云端已实际验证（见 compat/validation）：
- Linux，Python3.12.14，torch2.5.1+cpu；没有执行 Windows/CUDA 推理。
- 静态编译、原生 main/train/test/pretrain imports（主动禁止 PyTorch3D/chamfer/xformers）通过。
- 4项 unittest：旋转行约定/批量/精度、梯度、无扩展导入及 Mesh_mae 明确报错、官方100步DDIM一致性与state_dict keys。
- 现有13牙病例经原生 DentalDataset/DataLoader、完整TADPM网络和100步DDIM，outputs=(1,32,9)，全有限。
- 参考DDIM测试只在参考代码中将CUDA设备转移到CPU，没有替换U-ViT、condition或DDIM数学公式。
- 没有真实训练 checkpoint，因此没有验证真实 checkpoint 预测质量或 Windows GPU 执行。

复核测试命令（可选）：
```text
python -m unittest discover -s tests -p test_windows_smoke_compat.py -v
```
DDIM参考测试需要 Git 历史中的官方基线提交，因此完整Git克隆才能运行这一项。
