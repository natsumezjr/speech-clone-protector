from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "build" / "raw_modified.tex"
MEDIA_DIR = ROOT / "build" / "media_modified" / "media"


def clean_heading(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^\d+(?:\.\d+){0,2}\s*", "", title)
    title_overrides = {
        "端到端数据流转控制": "后端数据流转控制",
        "核心算法的资源约束与底层优化": "算法的计算优化",
        "攻防验证沙箱与多维效能度量": "防御效果自动化评估",
        "异步调度架构与高可用状态展示": "异步调度架构与状态同步",
        "前后端接口协议与数据适配设计": "前后端接口与数据适配",
    }
    return title_overrides.get(title, title)


def strip_caption_punct(text: str) -> str:
    return re.sub(r"[。．.]$", "", text.strip())


def figure_block(path: str, chap: str, num: str, caption: str) -> str:
    caption = strip_caption_punct(caption)
    if chap == "3" and num == "5":
        return r"""
\begin{algorithm}[H]
\caption{V-Guard 主动保护扰动生成算法}
\label{alg:vguard}
\small
\begin{algorithmic}[1]
    \Inputs 原始音频 $x$，最大迭代步数 $N_{iter}$，保护模式 $m \in \{\mathrm{targeted},\mathrm{untargeted}\}$。
    \Parameters 最大扰动幅度 $\epsilon$，损失权重 $\lambda_{sem}, \lambda_{id}, \lambda_{psy}, \lambda_{2}$，更新步长 $\eta$。
    \Output 受保护音频 $x'$。

    \State $\delta \gets \mathrm{Uniform}(-\epsilon, \epsilon)$；
    \State $x' \gets \mathrm{Clamp}(x + \delta, -1.0, 1.0)$；
    \State $Z_{x}^{sem} \gets \mathcal{F}_{sem}(x)$，$Z_{x}^{id} \gets \mathcal{F}_{id}(x)$；
    \State $\Theta_{x}, P_{x} \gets \mathrm{Masker}(x)$；
    \If{$m = \mathrm{targeted}$}
        \State $x_{t} \gets \mathrm{SelectTargetAudio}(x)$；
        \State $Z_{t}^{id} \gets \mathcal{F}_{id}(x_{t})$；
    \EndIf

    \For{$i \gets 1$ \textbf{to} $N_{iter}$}
        \State $Z_{x'}^{sem} \gets \mathcal{F}_{sem}(x')$，$Z_{x'}^{id} \gets \mathcal{F}_{id}(x')$；
        \If{$m = \mathrm{targeted}$}
            \State $\mathcal{L}_{id} \gets -\mathrm{Sim}_{id}\!\left(Z_{t}^{id}, Z_{x'}^{id}\right)$；
        \Else
            \State $\mathcal{L}_{id} \gets \mathrm{Sim}_{id}\!\left(Z_{x}^{id}, Z_{x'}^{id}\right)$；
        \EndIf
        \State $\mathcal{L}_{sem} \gets \mathrm{Sim}_{sem}\!\left(Z_{x}^{sem}, Z_{x'}^{sem}\right)$；
        \State $\mathcal{L}_{psy} \gets \mathrm{PsyLoss}\!\left(\delta, \Theta_{x}, P_{x}\right)$；
        \State $\mathcal{L}_{2} \gets \lVert \delta \rVert_{2}$；
        \State $\mathcal{L} \gets \lambda_{sem}\mathcal{L}_{sem}+\lambda_{id}\mathcal{L}_{id}+\lambda_{psy}\mathcal{L}_{psy}+\lambda_{2}\mathcal{L}_{2}$；
        \State $\delta \gets \mathrm{Clamp}\!\left(\delta-\eta\cdot\mathrm{sign}\left(\nabla_{\delta}\mathcal{L}\right), -\epsilon, \epsilon\right)$；
        \State $x' \gets \mathrm{Clamp}(x+\delta, -1.0, 1.0)$；
    \EndFor
    \State \Return $x'$；
\end{algorithmic}
\end{algorithm}
"""
    return rf"""
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=\textwidth,height=0.72\textheight,keepaspectratio]{{{path}}}
  \caption{{{caption}}}
  \label{{fig:{chap}-{num}}}
\end{{figure}}
"""


def table_caption(match: re.Match[str]) -> str:
    chap, num, caption = match.group(1), match.group(2), strip_caption_punct(match.group(3))
    return rf"\captionof{{table}}{{{caption}}}\label{{tab:{chap}-{num}}}"


def fix_table_48(text: str) -> str:
    """Rewrite the ASR shift examples table with fixed-width wrapping columns."""
    pattern = re.compile(
        r"\\captionof\{table\}\{保护后 ASR 转写偏移示例\}\\label\{tab:4-8\}\s*"
        r"\{\\def\\LTcaptype\{table\}.*?\\end\{longtable\}\s*\}",
        re.S,
    )
    replacement = r"""\begin{table}[htbp]
\centering
\caption{保护后 ASR 转写偏移示例}
\label{tab:asr-shift-examples}
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}
  >{\centering\arraybackslash}m{0.13\textwidth}
  >{\centering\arraybackslash}m{0.18\textwidth}
  >{\centering\arraybackslash}m{0.27\textwidth}
  >{\centering\arraybackslash}m{0.27\textwidth}
  >{\centering\arraybackslash}m{0.075\textwidth}@{}
}
\toprule
ASR 模型 & 样本编号 & 参考文本摘录 & \makecell{保护后识别文本\\摘录} & WER \\
\midrule
\makecell[c]{Whisper\\Medium} & \makecell[c]{\ttfamily\footnotesize 2902\_9008\_\\\ttfamily\footnotesize 000013\_000003} & Six new pupils in the mathematical school this morning. & of 50 seconds, scan the several chest camera recordings... & 388.9\% \\
\makecell[c]{Whisper\\Medium} & \makecell[c]{\ttfamily\footnotesize 2902\_9008\_\\\ttfamily\footnotesize 000022\_000000} & Ah, so they say-Your excellent father has vanished. & I'm gonna get off. Do you want to do the X-ray?... & 333.3\% \\
\makecell[c]{wav2vec2-\\base-960h} & \makecell[c]{\ttfamily\footnotesize 2902\_9008\_\\\ttfamily\footnotesize 000055\_000003} & Does your excellency, or this proud bishop, govern Alexandria? & ER LAR AS SOON AS YO COT HIS THOUGHT... & 160.0\% \\
\makecell[c]{wav2vec2-\\base-960h} & \makecell[c]{\ttfamily\footnotesize 3536\_23268\_\\\ttfamily\footnotesize 000019\_000000} & Miss Milner, you shall not leave the house this evening. & A FEW MOMEN PREAT YOUM ENOUGH... & 130.0\% \\
\bottomrule
\end{tabular}
\end{table}"""
    return pattern.sub(lambda _: replacement, text)


def center_table_columns(text: str) -> str:
    """Use centered m-columns so table cells are centered horizontally and vertically."""
    text = text.replace(r">{\centering\arraybackslash}p{", r">{\centering\arraybackslash}m{")
    text = text.replace(r">{\raggedright\arraybackslash}p{", r">{\centering\arraybackslash}m{")
    text = text.replace(r">{\RaggedRight\arraybackslash}p{", r">{\centering\arraybackslash}m{")
    text = text.replace(r">{\RaggedRight\arraybackslash}X", r">{\centering\arraybackslash}X")
    text = text.replace(r"\makecell[l]", r"\makecell[c]")
    return text


def main() -> None:
    text = RAW.read_text(encoding="utf-8")
    start = text.find(r"\section{摘要}")
    if start < 0:
        raise SystemExit("Cannot find abstract section in build/raw.tex")

    text = (
        text[start:]
        .replace("build/media_modified/media/", "figures/")
        .replace("build/media/", "figures/")
    )

    svg_replacements = {
        r"\includesvg[width=0.52083in,height=0.27083in]{figures/image9.svg}": r"\(T_m(k)\)",
        r"\includesvg[width=2.61458in,height=0.30208in]{figures/image11.svg}": r"\[T_m(k)=E_{masker}+\Delta_m+S_f(\Delta z(k))\]",
        r"\includesvg[width=0.40625in,height=0.27083in]{figures/image13.svg}": r"\(\Theta(x)\)",
        r"\includesvg[width=0.40625in,height=0.27083in]{figures/image14.svg}": r"\(\Theta(x)\)",
    }
    for old, new in svg_replacements.items():
        text = text.replace(old, new)

    text = text.replace("``音色（Timbre）''的超集", "``音色''的超集")
    text = text.replace(r"timbre\_mode=untargeted", "保护模式=非定向")

    symbol_replacements = {
        r"F_{sem}": r"\mathcal{F}_{sem}",
        r"F_{id}": r"\mathcal{F}_{id}",
        r"F_{\mathrm{sem}}": r"\mathcal{F}_{sem}",
        r"F_{\mathrm{id}}": r"\mathcal{F}_{id}",
        r"L_{sem}": r"\mathcal{L}_{sem}",
        r"L_{id}": r"\mathcal{L}_{id}",
        r"L_{psy}": r"\mathcal{L}_{psy}",
        r"L_{2}": r"\mathcal{L}_{2}",
        r"L_2": r"\mathcal{L}_{2}",
        r"L_{\mathrm{sem}}": r"\mathcal{L}_{sem}",
        r"L_{\mathrm{id}}": r"\mathcal{L}_{id}",
        r"L_{\mathrm{psy}}": r"\mathcal{L}_{psy}",
        r"N_{\mathrm{iter}}": r"N_{iter}",
        r"\lambda_{\mathrm{sem}}": r"\lambda_{sem}",
        r"\lambda_{\mathrm{id}}": r"\lambda_{id}",
        r"\lambda_{\mathrm{psy}}": r"\lambda_{psy}",
    }
    for old, new in symbol_replacements.items():
        text = text.replace(old, new)

    text = text.replace(r"L_{psy\ }", r"\mathcal{L}_{psy}")
    text = text.replace(r"L_{total}", r"\mathcal{L}")
    text = text.replace("音色混淆流程示意图", "声音身份混淆流程示意图")
    text = text.replace("非定向与定向的音色混淆流程如下图所示", "非定向与定向的声音身份混淆流程如下图所示")

    text = text.replace(
        r"\[原始音频/受保护音频\  \rightarrow \ ASR\ 音频\  \rightarrow \ 转写文本\  \rightarrow \ 文本差异与语义链路指标\]",
        r"""\begin{center}
\fbox{\parbox{0.9\textwidth}{\centering 原始音频/受保护音频 $\rightarrow$ ASR 音频 $\rightarrow$ 转写文本 $\rightarrow$ 文本差异与语义链路指标}}
\end{center}""",
    )
    text = re.sub(
        r"\\\[原始音频/受保护音频 \\rightarrow 语音克隆模型 \\rightarrow 克隆音频\\\s+\\rightarrow\s*\\\]\s*\\\[\\rightarrow 说话人评估模型 \\rightarrow 克隆防护指标\\\]",
        lambda _: r"""\begin{center}
\fbox{\parbox{0.9\textwidth}{\centering 原始音频/受保护音频 $\rightarrow$ 语音克隆模型 $\rightarrow$ 克隆音频 $\rightarrow$ 说话人评估模型 $\rightarrow$ 克隆防护指标}}
\end{center}""",
        text,
    )

    fig_pattern = re.compile(
        r"\\includegraphics\[[^\]]*\]\{(figures/image\d+\.png)\}\s*\n\s*\n"
        r"图\s*(\d+)\s*-\s*(\d+)\s*(.*?)\s*(?=\n)",
        re.S,
    )
    text = fig_pattern.sub(lambda m: figure_block(m.group(1), m.group(2), m.group(3), m.group(4)), text)
    text = re.sub(r"(?m)^表\s*-?\s*(\d+)\s*-\s*(\d+)\s+(.*?)\s*$", table_caption, text)

    text = re.sub(
        r"\\subsection\{(.*?)\}(\\label\{[^}]+\})",
        lambda m: r"\section{" + clean_heading(m.group(1)) + "}" + m.group(2),
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\\subsubsection\{(.*?)\}(\\label\{[^}]+\})",
        lambda m: r"\subsection{" + clean_heading(m.group(1)) + "}" + m.group(2),
        text,
        flags=re.S,
    )

    chapter_matches = list(
        re.finditer(r"\\section\{第([一二三四五六])章\s*(.*?)\}(\\label\{[^}]+\})", text, flags=re.S)
    )
    ref_match = re.search(r"\\section\{参考文献\}(\\label\{[^}]+\})", text)
    if not chapter_matches or not ref_match:
        raise SystemExit("Cannot find chapter/reference boundaries after conversion")

    abstract = text[: chapter_matches[0].start()]
    body = text[chapter_matches[0].start() : ref_match.start()]
    references = text[ref_match.end() :]

    abstract = re.sub(
        r"\\section\{摘要\}\\label\{[^}]+\}",
        lambda _: "\\chapter*{摘要}\n\\addcontentsline{toc}{chapter}{摘要}",
        abstract,
        count=1,
    ).strip() + "\n"

    cn_to_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
    chapter_slices: list[tuple[int, str]] = []
    body_matches = list(
        re.finditer(r"\\section\{第([一二三四五六])章\s*(.*?)\}(\\label\{[^}]+\})", body, flags=re.S)
    )
    for i, match in enumerate(body_matches):
        end = body_matches[i + 1].start() if i + 1 < len(body_matches) else len(body)
        chapter_no = cn_to_num[match.group(1)]
        title = clean_heading(match.group(2))
        content = body[match.end() : end].strip()
        chapter_text = f"\\chapter{{{title}}}{match.group(3)}\n\n{content}\n\n\\FloatBarrier\n"
        chapter_text = chapter_text.replace(r"\def\LTcaptype{none}", r"\def\LTcaptype{table}")
        chapter_text = fix_table_48(chapter_text)
        chapter_slices.append((chapter_no, chapter_text))

    references = re.sub(r"(?i)doi\s*[:：]?\s*[^\n]+", "", references).strip()
    references = (
        "\\chapter*{参考文献}\n"
        "\\addcontentsline{toc}{chapter}{参考文献}\n"
        "\\markboth{参考文献}{参考文献}\n"
        "\\begin{referencesblock}\n"
        f"{references}\n"
        "\\end{referencesblock}\n"
    )

    for dirname in ["setup", "chapters", "figures", "tables"]:
        (ROOT / dirname).mkdir(exist_ok=True)

    for src in MEDIA_DIR.glob("*.png"):
        shutil.copy2(src, ROOT / "figures" / src.name)

    (ROOT / "main.tex").write_text(
        r"""\documentclass[UTF8,a4paper,zihao=-4,openany,oneside]{ctexbook}
\input{setup/preamble.tex}

\begin{document}

\pagestyle{empty}
\begin{titlepage}
\centering
\vspace*{1.2cm}
{\zihao{3}\heiti 第十九届全国大学生信息安全竞赛（作品赛）}\\[0.4cm]
{\zihao{3}\heiti 暨第三届“长城杯”网数智安全大赛（作品赛）}\\[1.4cm]
{\zihao{1}\heiti 作品报告}\\[1.4cm]
{\zihao{-3} \emptybox\ 命题赛道\quad \checkedbox\ 自由赛道}\\[1.8cm]
\begin{center}
\zihao{4}
\begin{tabular}{>{\centering\arraybackslash}m{0.22\textwidth}>{\centering\arraybackslash}m{0.62\textwidth}}
\textbf{作品名称：} & V-Guard：面向语音克隆风险的声音资产主动防护系统 \\[0.8cm]
\textbf{电子邮箱：} & 2682910849@qq.com \\[0.8cm]
\textbf{提交日期：} & 2026 年 7 月 5 日 \\
\end{tabular}
\end{center}
\vfill
\end{titlepage}

\clearpage
\begingroup
\pagestyle{empty}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
}
\tableofcontents
\clearpage
\endgroup

\pagenumbering{arabic}
\setcounter{page}{1}
\pagestyle{fancy}
\input{chapters/abstract.tex}
\input{chapters/chapter1.tex}
\input{chapters/chapter2.tex}
\input{chapters/chapter3.tex}
\input{chapters/chapter4.tex}
\input{chapters/chapter5.tex}
\input{chapters/chapter6.tex}
\input{chapters/references.tex}
\label{LastContentPage}

\end{document}
""",
        encoding="utf-8",
    )

    (ROOT / "setup" / "preamble.tex").write_text(
        r"""\usepackage[a4paper,top=2.5cm,bottom=2.5cm,left=3cm,right=2.5cm]{geometry}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{calc}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{float}
\usepackage{placeins}
\usepackage{etoolbox}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{tikz}
\usepackage[most]{tcolorbox}
\usepackage{makecell}
\usepackage{ragged2e}
\usepackage{xurl}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{lastpage}

\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{}
\IfFontExistsTF{SimSun}{\setCJKmainfont{SimSun}}{}
\IfFontExistsTF{SimHei}{\setCJKsansfont{SimHei}}{}
\IfFontExistsTF{KaiTi}{\setCJKmonofont{KaiTi}}{}

\hypersetup{hidelinks,unicode=true}
\setlength{\parindent}{2em}
\linespread{1.5}
\AtBeginDocument{\zihao{-4}\selectfont}
\setlist{nosep,leftmargin=2em}
\renewcommand{\arraystretch}{1.22}
\setlength{\tabcolsep}{5pt}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{8pt}

\ctexset{
  chapter = {
    format = \centering\zihao{3}\heiti,
    name = {第,章},
    number = \chinese{chapter},
    beforeskip = 20pt,
    afterskip = 20pt
  },
  section = {
    format = \zihao{4}\heiti,
    beforeskip = 12pt,
    afterskip = 6pt
  },
  subsection = {
    format = \zihao{-4}\heiti,
    beforeskip = 10pt,
    afterskip = 5pt
  },
  subsubsection = {
    format = \zihao{-4}\heiti,
    beforeskip = 8pt,
    afterskip = 4pt
  }
}

\numberwithin{figure}{chapter}
\numberwithin{table}{chapter}
\numberwithin{equation}{chapter}
\numberwithin{algorithm}{chapter}
\renewcommand{\thefigure}{\thechapter-\arabic{figure}}
\renewcommand{\thetable}{\thechapter-\arabic{table}}
\renewcommand{\theequation}{\thechapter-\arabic{equation}}
\renewcommand{\thealgorithm}{\thechapter-\arabic{algorithm}}
\floatname{algorithm}{算法}

\captionsetup{font=small,labelsep=quad,justification=centering,singlelinecheck=true}
\captionsetup[figure]{position=bottom,name=图}
\captionsetup[table]{position=top,name=表,skip=6pt}

\newcommand{\emptybox}{%
  \tikz[baseline=-0.18em,x=1em,y=1em]{%
    \draw[line width=0.45pt] (0,0) rectangle (0.78,0.78);%
  }%
}
\newcommand{\checkedbox}{%
  \tikz[baseline=-0.18em,x=1em,y=1em]{%
    \draw[line width=0.45pt] (0,0) rectangle (0.78,0.78);%
    \draw[line width=1.15pt,line cap=round,line join=round]
      (0.13,0.42) -- (0.32,0.20) -- (0.68,0.67);%
  }%
}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.22}}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{第 \thepage 页，共 \pageref{LastPage} 页}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \fancyfoot[C]{第 \thepage 页，共 \pageref{LastPage} 页}%
}

\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}

\algrenewcommand\algorithmicrequire{\textbf{输入：}}
\algrenewcommand\algorithmicensure{\textbf{输出：}}
\algrenewcommand\algorithmicreturn{\textbf{返回}}
\newcommand{\Inputs}{\item[\textbf{输入：}]}
\newcommand{\Parameters}{\item[\textbf{参数：}]}
\newcommand{\Output}{\item[\textbf{输出：}]}
\newenvironment{referencesblock}{\small\RaggedRight\sloppy\setlength{\parindent}{0pt}\setlength{\parskip}{0.35em}}{}
""",
        encoding="utf-8",
    )

    (ROOT / "chapters" / "abstract.tex").write_text(abstract, encoding="utf-8")
    for number, content in chapter_slices:
        lines = content.splitlines()
        fixed_lines = []
        pending_caption = None
        in_longtable = False
        for line in lines:
            caption_match = re.match(r"\\captionof\{table\}\{(.*)\}\\label\{(.*)\}", line)
            if caption_match:
                pending_caption = (caption_match.group(1), caption_match.group(2))
                continue
            if line.startswith(r"\begin{longtable}"):
                in_longtable = True
                fixed_lines.append(line)
                continue
            if in_longtable and pending_caption and line.startswith(r"\toprule"):
                caption, label = pending_caption
                fixed_lines.append(rf"\caption{{{caption}}}\\")
                pending_caption = None
            if line.startswith(r"\end{longtable}"):
                in_longtable = False
            fixed_lines.append(line)
        content = center_table_columns("\n".join(fixed_lines) + "\n")
        (ROOT / "chapters" / f"chapter{number}.tex").write_text(content, encoding="utf-8")
    (ROOT / "chapters" / "references.tex").write_text(references, encoding="utf-8")
    (ROOT / "latexmkrc").write_text(
        "$pdf_mode = 5;\n"
        "$xelatex = 'xelatex -interaction=nonstopmode -file-line-error %O %S';\n"
        "$out_dir = 'build/tex';\n"
        "$aux_dir = 'build/tex';\n",
        encoding="ascii",
    )
    (ROOT / "Makefile").write_text("all:\n\tlatexmk -xelatex main.tex\n\nclean:\n\tlatexmk -C main.tex\n", encoding="ascii")


if __name__ == "__main__":
    main()
