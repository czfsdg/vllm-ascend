# 年中述职展示材料

本目录包含一套不超过 5 页的年中述职展示材料，围绕“实时视频推理”“投机加速 DFlash / DCut”“版本交付”“协作贡献与改进方向”提炼。

## 文件

- `index.html`：网页展示版，浏览器直接打开即可演示，按整屏滚动切换页面。
- `generate_pptx.py`：本地生成 PPT 的脚本，运行后会生成 `midyear_review_presentation.pptx`。
- `midyear_review_presentation.pptx`：本地生成的 PPT 展示版，共 5 页；该文件已被 `.gitignore` 忽略，不会进入 GitHub PR。

## 页面结构

1. 年中述职总览：关键成果快照。
2. 实时视频推理核心突破：SP、Timestep PP、压缩策略、系统构建。
3. 实时视频推理交付与影响力：客户验收、技术文章、外部合作。
4. 投机加速 DFlash 训练 & DCut 适配：机制探索、验证、环境支撑、版本交付。
5. 协作贡献、他人帮助与改进方向：团队协作、获得支持、Agent 工具提升计划。

## 生成 PPT

```bash
python reports/midyear_review/generate_pptx.py
```

生成后的 `reports/midyear_review/midyear_review_presentation.pptx` 可直接从本地工作区拷贝或下载使用，但不会被提交到 GitHub。
