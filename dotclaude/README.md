# dotclaude —— 全局 Claude 配置的备份

这里的文件**不是给本项目用的**，是 `~/.claude/` 的备份，用 Git 带到另一台机器上。

| 文件 | 装到哪 | 作用 |
|---|---|---|
| `CLAUDE.md` | `~/.claude/CLAUDE.md` | 全局排版规则（数学公式写法），对你**所有项目**生效 |
| `skills/math-rendering/SKILL.md` | `~/.claude/skills/math-rendering/SKILL.md` | 换客户端后重新探测公式渲染的方法 |

**本项目内不装也没关系** —— 项目自己的 `CLAUDE.md` 规则 11 已经包含同样的排版规则。
装了只是让你在**别的项目**里也享受同样的排版。

**在新机器上安装：** 开 Claude Code 说「把 dotclaude 装到全局」即可。
Claude 会先检查那台机器上有没有已存在的 `~/.claude/CLAUDE.md`，**有的话合并而不是覆盖**。

2026-09-18 在笔记本上建立。**改了全局文件后记得同步回这里**，否则两台机器会不一致。
