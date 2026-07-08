#!/usr/bin/env python3
"""Generate the local midyear review PPTX deck.

The generated PPTX is intentionally ignored by git because GitHub PRs in this
repository should not include binary PowerPoint files. Run this script locally
when you need to refresh the PPT deliverable.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

OUTPUT_FILE = Path(__file__).with_name("midyear_review_presentation.pptx")
SLIDE_WIDTH = 12_192_000
SLIDE_HEIGHT = 6_858_000

SLIDES = [
    (
        "年中述职",
        [
            "实时视频推理 × 投机加速",
            "客户验收通过｜305/306 需求零打回｜2 篇技术文章",
            "SP 方案 10.14 → 16.4 FPS，加速 61.7%",
            "Timestep PP 最佳验证 24.06 FPS",
        ],
    ),
    (
        "实时视频推理：核心突破",
        [
            "SP 扩展至 15 卡：突破 8 卡 Ulysses 限制，通信计算并行 + RoPE 预计算",
            "Timestep PP：降低 Layer PP 流水线气泡，Rolling Forcing 18.12 FPS，新推理方式 20.2 FPS",
            "压缩策略：空间分辨率压缩在精度可接受前提下加速 40.49%",
            "系统构建：支持 T2V、I2V、实时视频编辑，多分辨率与时长配置",
        ],
    ),
    (
        "实时视频推理：交付与影响力",
        [
            "客户交付：整理推理代码并提供技术支持，提交魔芯科技且通过验收",
            "成果沉淀：2 篇技术文章发布于稼先社区和 2012 网站，多次上首页",
            "外部影响：累计 1300+ 阅读，引发技术讨论，多支海内外团队咨询合作",
            "价值闭环：技术突破 → 工程落地 → 客户成功 → 影响力扩散",
        ],
    ),
    (
        "投机加速：DFlash 训练 & DCut 适配",
        [
            "单向 DFlash：将双向建模改造为单向建模，接收率基本一致，具备研究意义",
            "NPU / GPU 验证：覆盖训练、推理、性能多维度测试",
            "环境与数据：修复 GPU 机器、重装驱动和 Toolkit、准备训练数据集",
            "DCut：Qwen3 多数据集验证，低接收率下 19.1 out_tokens/s；Qwen3.5 开发中",
            "版本交付：305 两项、306 两项需求完成交付，无打回",
        ],
    ),
    (
        "协作贡献与改进方向",
        [
            "贡献他人：协助定位 DFlash 接收率下降，提出损失函数与接收率目标不匹配方向",
            "保障进度：修复 ffmpeg 安装兼容性、新版镜像显存管理 bug，减少团队阻塞",
            "获得帮助：汶轩、鹏哥、宁哥在方案设计、方向把关、vLLM NPU 入门和文章修改上提供支持",
            "待改进：提升 Agent 工具使用能力，将其用于开发、排障、测试、文档和信息检索",
        ],
    ),
]


def build_content_types(slide_count: int) -> str:
    slide_overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for index in range(1, slide_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/ppt/presentation.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        f"{slide_overrides}</Types>"
    )


def build_root_relationships() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="ppt/presentation.xml"/>'
        "</Relationships>"
    )


def build_presentation(slide_count: int) -> str:
    slide_ids = "".join(f'<p:sldId id="{255 + index}" r:id="rId{index}"/>' for index in range(1, slide_count + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        f"<p:sldIdLst>{slide_ids}</p:sldIdLst>"
        f'<p:sldSz cx="{SLIDE_WIDTH}" cy="{SLIDE_HEIGHT}" type="wide"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    )


def build_presentation_relationships(slide_count: int) -> str:
    relationships = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        f'Target="slides/slide{index}.xml"/>'
        for index in range(1, slide_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )


def build_text_box(
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    font_size: int = 3200,
    *,
    bold: bool = False,
    color: str = "EEF6FF",
) -> str:
    bold_tag = "<a:b/>" if bold else ""
    shape_id = abs(hash((text, x, y, width, height))) % 100_000
    return (
        "<p:sp><p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="Text"/><p:cNvSpPr txBox="1"/><p:nvPr/>'
        "</p:nvSpPr><p:spPr>"
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln>'
        '</p:spPr><p:txBody><a:bodyPr wrap="square"/><a:lstStyle/><a:p><a:r>'
        f'<a:rPr lang="zh-CN" sz="{font_size}" dirty="0">{bold_tag}'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
        f'<a:t>{escape(text)}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="{font_size}"/>'
        "</a:p></p:txBody></p:sp>"
    )


def build_slide(title: str, bullets: list[str], index: int) -> str:
    shapes = [
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Background"/><p:cNvSpPr/><p:nvPr/>'
        "</p:nvSpPr><p:spPr>"
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_WIDTH}" cy="{SLIDE_HEIGHT}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '<a:solidFill><a:srgbClr val="08111F"/></a:solidFill><a:ln><a:noFill/></a:ln>'
        "</p:spPr></p:sp>",
        build_text_box(title, 650_000, 430_000, 10_800_000, 720_000, 4200, bold=True, color="43D9FF"),
    ]
    y_position = 1_450_000
    for bullet in bullets:
        shapes.append(
            build_text_box(
                f"• {bullet}",
                900_000,
                y_position,
                10_300_000,
                720_000,
                2300,
            )
        )
        y_position += 850_000
    shapes.append(
        build_text_box(f"{index:02d} / 05", 10_600_000, 6_250_000, 1_000_000, 300_000, 1200, bold=True, color="ABC0D8")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr>"
        '<a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
        f"</p:grpSpPr>{''.join(shapes)}</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    )


def generate_pptx() -> None:
    with ZipFile(OUTPUT_FILE, "w", ZIP_DEFLATED) as pptx:
        pptx.writestr("[Content_Types].xml", build_content_types(len(SLIDES)))
        pptx.writestr("_rels/.rels", build_root_relationships())
        pptx.writestr("ppt/presentation.xml", build_presentation(len(SLIDES)))
        pptx.writestr("ppt/_rels/presentation.xml.rels", build_presentation_relationships(len(SLIDES)))
        for index, (title, bullets) in enumerate(SLIDES, 1):
            pptx.writestr(f"ppt/slides/slide{index}.xml", build_slide(title, bullets, index))
            pptx.writestr(
                f"ppt/slides/_rels/slide{index}.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>',
            )


def main() -> None:
    generate_pptx()
    print(f"Generated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
