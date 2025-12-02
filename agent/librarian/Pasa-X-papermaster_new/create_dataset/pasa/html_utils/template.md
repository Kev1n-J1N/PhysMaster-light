# ACL_PATTERN —— 会议论文格式
## 正则表达：
```
r"(?P<authors>.+?)\.\s+"
r"(?P<year>\d{4}[a-z]?)\.\s+"
r"(?P<title>[^.]+?)\.\s+"
r"In Proceedings of (?P<journal>[^,]+)"
```
#### 格式结构：作者. 年份. 标题. In Proceedings of 会议名称,...
#### 示例：Qingyun Wang, Doug Downey, Heng Ji, and Tom Hope. 2024a. SciMON: Scientific inspiration machines optimized for novelty. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ...

# REPORT_PATTERN —— 技术报告格式
## 正则表达：
```
r"(?P<authors>.+?),\s+"
r"(?P<title>[^,]+?),\s+"
r"(?P<journal>[^,\(]+)
```
#### 格式：作者, 标题, 报告编号, (...)
#### 示例：CMS Collaboration, The Phase-2 Upgrade of the CMS L1 Trigger Interim Technical Design Report, CERN-LHCC-2017-013, (2017)


# JOURNAL_PATTERN —— 期刊文章格式（如 NIM） 
# 正则表达：
```
r"(?P<authors>.+?),\s+"
r"(?P<title>[^,]+?),\s+"
r"(?P<journal>[^,]+?),\s+p\.\s+\d+"
```
#### 格式：作者, 标题, 期刊名, p. 页码, (...)
#### 示例：G. Hall, A time-multiplexed track-trigger for the CMS HL-LHC upgrade, Nucl. Instrum. Methods Phys. Res. 824, p. 292 – 295, (2016)
