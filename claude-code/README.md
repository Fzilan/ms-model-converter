# Claude Code Packs for Model Conversion

Self-contained Claude Code packages designed for HF.transformers migration from PyTorch to MindSpore.

```
pack-{i}/
├── CLAUDE.md          
├── .claude/           # from one of the pack
    ├── ...            # skills, tools or something
```

## TLDR
对于transformer 模型转换，当前设计了三个思路
1. 包俩 skills 和 代码片段， workflow 按需触发
2. 包个综合 skill-`convert-model`把几个步骤串起来，但是对于单个模型转换的实验场景，暂时不测试，该思路适合整体代码仓升级的时候的其中模型转换的大 skill 封装。
3. 不写 skills, 只提供 snippets + workflow，对于单模型转换应该够用了；需要测试一下效果，并且对比思路1 哪个更省token。

对于单模型转换场景在cc上先试试对比 思路1 和 思路3；思路2 这种综合 skill 的包装估计适合更复杂的项目升级，workflow.md 也应该作为这个大skill的一部分。


## usages
Choose one of the pack and place it under the migrate working space and start `cluade` there.

```
workspace/
├── CLAUDE.md          # from one of the pack
├── .claude/           # from one of the pack
├── transformers       # hf transformers - source repo
└── mindone            # mindone transformers - target repo
```

## how they designed?

1. Pack-1: Minimal skills + snippets, no overlaps between skills.
No umbrella skills, rely on workflow and might trigger `register-auto-maps` and `test-author` skills.

2. Pack-2: One umbrella skill + snippets
Keep only the umbrella skill - `convert-model` that indicate comprehensive conversion things, that might combine the usage of several snippets together.
(I think the pack could be refactored in this way later, that we designed a more complex migration projects)

3. Pack-3: WORKFLOW.md + snippets, no skills trigger
Lighter one. Removed all skills. Repointed all snippets references to CLAUDE.md.
## TODO
I will test the single model conversion task usage pack 1 and pack 3 latter to compare the preformance.


