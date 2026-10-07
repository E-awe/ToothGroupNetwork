本目录保存此前已实际生成的 tadpm_official_upper_input.zip 原始文件；未重新生成数据。

ZIP 大小：8,625,260 字节。

在 GitHub 打开 ZIP 文件后，选择 Download raw file（下载原始文件），保存到 Windows 本地 D:\CLK\TADPM\TADPM_official\。
SHA256SUMS 用于校验下载文件。

ZIP 包含 case_0/、prepare_case.py、官方 Dataset 读取验证器、组件-FDI-slot 映射、报告和预处理依赖文件。
基于官方 lcshhh/TADPM 提交 92967a68191846ab38dcdb2ae7914e19ecd0d396。
仅处理上颌13颗牙：FDI18、15、28缺失，对应slot0、3、15保持为空，下颌slot16–31全部为空。
保留原始坐标副本，牙龈单独保存，不作为牙齿输入。

重要：目前没有真实治疗后扫描。pcd_after512_PLACEHOLDER/ 和 param_PLACEHOLDER/ 为原始 Dataset 加载检查的占位数据，不是治疗后真值或模型预测，禁止用于训练或疗效评估。
已执行官方 Dataset 读取和特征构建逻辑，隔离注册器依赖并使用等价纯 Torch 旋转6D转换；未执行完整 TADPM 模型预测。
归一化为完整上颌表面中心居中并除以40，上颌单独处理的适配说明、矩阵及逆矩阵保存在报告中。
ZIP 内 dataset_config.json 包含生成时绝对路径，迁移后运行 configure_case.py --case <本地case_0目录> 更新路径。
