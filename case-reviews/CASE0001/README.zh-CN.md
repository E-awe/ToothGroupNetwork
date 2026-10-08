CASE0001 三维图人工复核候选结果

重要：所有牙位编号仅为三维图复核候选，尚未经过临床记录最终确认。不得将此文件称为临床最终牙位或标注真值。

已采用的规则：
- 上颌C1–C13保持原CSV中的编号，沿用用户当前采用的患者左右方向，不采用mirror列重新编号。
- 上颌缺牙位置单独记录为候选FDI15。缺牙记录没有component_id、顶点或网格，不生成虚构牙齿。
- 下颌C10保持43；C11原预测43改为复核候选44；C12原预测44改为复核候选45。其余牙位不变。
- 上颌C14、下颌C15为牙龈/基底，保持0。

主要文件：
reviewed_component_fdi.csv：29个真实组件记录（上颌14、下颌15），加1个上颌FDI15缺牙候选记录。
original_vs_reviewed_comparison.csv：保留相同的完整原始/候选对照字段。
changed_components.csv：仅列下颌C11和C12两项改变及原因。
upper_reviewed_colored.ply、lower_reviewed_colored.ply：原始source坐标的整颌网格，按复核组件候选FDI统一着色，牙龈为灰色。
upper_reviewed_model_coords.ply、lower_reviewed_model_coords.ply：沿用原输入manifest中已有的model坐标变换，便于对照已采用的三维图方向。
*_original_component_candidates.ply：同一几何/色表下原始组件候选着色结果。
*_original_vertex_prediction.ply：按原始TGNet逐顶点labels着色的独立副本，不是覆盖后的JSON。
*_numbered_comparison.png：原始组件候选/复核候选的编号图，俯视与斜视各一组；不通过镜像换牙号。
*_reviewed_comparison.html：可离线打开的交互三维对照图；完整网格，左右两图可各自旋转检查。
original_reference/：原始候选CSV、两个原始TGNet预测JSON及方向记录的原样副本。
*_reviewed_component_candidates.npz：保留component_id、original_component_fdi、reviewed_component_fdi、raw_tgnet_vertex_fdi四个数组，与原顶点顺序对应。
review_provenance.json：变更、原因、方向矩阵及逆矩阵、输入SHA256前后对照和几何核查。
SHA256SUMS：交付文件校验。
build_review.py：可复现的CPU复核生成脚本；输出目录必须不存在，避免覆盖旧数据。

CSV字段：
original_prediction_fdi是原CSV的candidate_fdi（组件内TGNet主导前景编号），不是临床编号。
original_arch_proposal_fdi保留原CSV的牙弓顺序提案，不将其冒充原预测。
reviewed_candidate_fdi为本次用户指示的三维图复核候选。
original_candidate_fraction、original_foreground_label_purity、original_foreground_fraction及original_flags只描述原始模型结果，不是修正编号的置信度。
record_type=missing_tooth、tooth_present=False表示FDI15缺牙候选；不绑定到任何现存组件。
clinical_confirmed=False、review_status=3d_review_candidate_pending_clinical_confirmation明确临床记录确认仍待完成。
下游读取时先按record_type/component及role/tooth过滤；不要把缺牙记录当成一个组件。

关键区别：复核后的PLY/NPZ是“组件级候选映射”的展示衍生数据；组件内所有顶点按一个候选着色，因此不会保留原预测在牙齿上分出的背景斑块。原始JSON中的labels、instances完全不变。尚未生成修正TGNet segmentation JSON。
输入ZIP不含模型权重或单牙STL；全部提供的原始文件均保持字节不变，未修改任何权重或STL，也未运行模型重推理。
仅记录用户指定的缺牙候选15，不将未列出的其他牙位自动认定为临床缺失。

复现（NumPy、Trimesh、Matplotlib、Plotly）：
python build_review.py --source /path/to/original/CASE0001 --out /path/to/new/reviewed

临床记录最终确认后，应另建最终确认版本，保留本次候选与原始预测的可追溯关系。
