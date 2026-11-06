# 项目目录

目录职责参考 [ALAS](https://github.com/LmeSzinc/AzurLaneAutoScript)：部署、开发工具、配置、资源和文档分别维护。保留本项目现有 `drift` 包名与 `web` 路径，避免破坏源码导入和旧启动入口。

```text
drifting-girl-assistant/
├── desktop_entry.py       # Windows 桌面入口
├── app.py                 # 命令行与旧入口兼容
├── 启动.cmd                  # 兼容入口；便携版直接打开 mumu-panel.exe
├── drift/                 # 主程序，对应 ALAS 的 module 职责
│   ├── core/              # 图像、持久化、生命周期与工作流
│   ├── device/            # ADB、原生截图、模拟器选择
│   ├── perception/        # 数字、菜单、餐品等识别
│   ├── tasks/             # 游戏任务与调度
│   ├── desktop/           # 原生桌面面板与明暗主题
│   ├── update/            # 下载、校验、安装与回滚
│   ├── ui/                # 通用界面操作
│   └── web/               # 本机服务及兼容网页入口
├── alas_core/             # 已注明来源的 ALAS 基础组件
├── config/default.json    # 公开默认配置：所有任务关闭
├── deploy/                # 构建、打包、发行包验证
├── dev_tools/             # 离线样本、配置维护、回放及开发工具
├── assets/                # 游戏识别图片
├── models/                # 固定识别模型与校验信息
├── web/                   # 兼容网页的静态文件
├── docs/                  # 使用介绍、开发、发布与目录说明
├── tests/                 # 回归测试和公开样例
└── licenses/              # 第三方许可
```

根目录的旧任务模块是很薄的导入兼容层；实际实现位于 `drift/`。新增业务代码应放进相应子包，不再添加根目录任务实现。

## 配置与运行数据

`config/default.json` 是程序附带的模板。首次启动仅在根目录 `config.json` 不存在时创建本机配置；已有安装继续读取原文件。日常、标尺、购买、任务交接、调度等保存记录仍使用原路径和内容。本次目录整理不搬动运行记录，不重置截止时间或已用次数。

更新器只允许替换明确列出的程序文件、资源、文档及默认模板；`config.json`、自定义模板、日志和任务记录不属于更新载荷。`deploy/` 和 `dev_tools/` 随源码提供，普通 Windows 便携包不包含开发工具、测试或诊断资料。

## 开发入口

- `python deploy/verify_release.py`：检查公开配置、模板、模型和源码。
- `python deploy/build_release.py`：按固定 `deploy/mumu-panel.spec` 构建桌面便携版。
- `python dev_tools/test_gate.py diagnostics/full-regression`：按模块隔离执行完整回归。
- `python desktop_entry.py`：运行桌面面板；`python app.py serve` 保留本机网页服务入口。
