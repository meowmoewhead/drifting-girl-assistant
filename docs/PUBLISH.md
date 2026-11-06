# GitHub 发布步骤

本项目已有公开仓库。日常更新在现有 main 历史上添加提交和新版本标签，不重写旧版本。下方初始化与关联步骤仅适用于首次创建独立仓库，已有仓库请跳过。

## 1. 提交源码

确认 `config.json`、状态、日志、构建目录和可执行文件处于忽略列表。`config/default.json`、必要模板、模型、测试样例和许可证应被提交。

在此目录中执行（若已有本地发布提交，可跳过初始化和首次提交）：

```powershell
git init -b main
git add .
git diff --cached --stat
git commit -m "release: public preview v0.2.0"
```

不要把正在使用的开发仓库或其历史直接覆盖到这里。本目录已经保留必要代码和来源声明。

## 2. 关联空仓库

把下面的 `OWNER` 和仓库名换成实际值。创建空仓库时不用自动生成 README 或 LICENSE，本项目已经提供。

```powershell
git remote add origin https://github.com/OWNER/drifting-girl-assistant.git
git push -u origin main
git tag v0.2.0
git push origin v0.2.0
```

## 3. 创建 Release

从 `v0.2.0` 创建预发布，标题可用“漂流助手 v0.2.0 公开预览”。填写版本说明、支持环境及必要限制，上传：

- `mumu-clicker-0.2.0-win64.zip`：普通用户使用的便携版。
- `drifting-girl-assistant-0.2.0-source.zip`：与本次发行对应的完整源码快照。
- `SHA256SUMS.txt`：附件校验值。

可再附 `build-report.json`。EXE和便携ZIP通过 Release 分发，不提交到 Git 历史。源码标签应对应本次构建使用的内容。

功能介绍使用 README 和 `docs/INTRO.md` 中的界面截图，不上传视频。发布成功后核对 README 的下载说明和图文链接。

官方参考：[上传本地项目](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。
