# V-Guard 作品报告 LaTeX 工程

本目录用于维护 `V-Guard_文献修改版.docx` 转换得到的 XeLaTeX 作品报告工程。正文、标题层级、引用和参考文献以 `V-Guard_文献修改版.docx` 为准；其他 Word/PDF 文件仅作为格式和规范参考。

## 文件结构

```text
report/
  main.tex                     主入口文件
  setup/preamble.tex           宏包、字体、标题、页眉页脚、图表样式
  chapters/                    正文各章节与参考文献
  figures/                     从 Word 提取并整理后的图片
  scripts/build_latex_project.py  从中间稿重建工程的辅助脚本
  latexmkrc                    latexmk 配置，使用 XeLaTeX
  Makefile                     简单编译入口
  V-Guard-report.pdf           当前交付版 PDF
  V-Guard_文献修改版.docx       正文与参考文献标准源文件
```

`build/`、`tmp/` 和 LaTeX 辅助文件为本地生成产物，不纳入 Git。

## 编译环境

推荐环境：

- Windows 10/11
- MiKTeX 25.x 或 TeX Live 2025/2026
- `latexmk`
- XeLaTeX
- 中文字体：宋体 `SimSun`、黑体 `SimHei`
- 英文字体：`Times New Roman`

可选检查工具：

- Poppler：`pdfinfo`、`pdftoppm`、`pdffonts`
- Python 3：用于脚本检查 PDF 页数、引用和文本抽取

本工程不使用 pdfLaTeX。中文排版基于 `ctex`，正文为小四号、1.5 倍行距，正文页脚居中显示页码。

## 编译命令

在 `report/` 目录下执行：

```powershell
latexmk -xelatex main.tex
```

或使用：

```powershell
make
```

编译输出默认位于：

```text
build/tex/main.pdf
```

如需更新交付版 PDF，可在编译通过后执行：

```powershell
Copy-Item build\tex\main.pdf V-Guard-report.pdf -Force
```

## 清理命令

```powershell
latexmk -C main.tex
Remove-Item -Recurse -Force build,tmp -ErrorAction SilentlyContinue
```

如果 PDF 正在被 WPS、浏览器或系统预览器占用，先关闭对应 PDF 预览窗口后再清理。

## 维护注意事项

- 不要用旧版 Word 或旧 PDF 覆盖 `V-Guard_文献修改版.docx` 的正文内容。
- 参考文献应保持 60 条，并按正文首次出现顺序维护。
- 图题在图下，表题在表上。
- 当前表格列统一采用水平、垂直居中排版。
- 图 3-5 已改为后续图片顺延编号；主动保护扰动生成步骤作为算法编号显示。
- 提交前至少运行一次 `latexmk -xelatex main.tex`，确认 PDF 可生成。
