# Heretic-Scalpel Swarm Forge

在 `Heretic-Scalpel-E2B` 上做**三段式 MoE 微调**（大核汇聚 + 小核稀疏专家），
配一套 Web 界面来定义数据集、分配架构、一键开训。

```
backend/src/forge_core/    纯 ML 库（datagen / modeling / train / export）
backend/src/forge_web/     FastAPI 后端（进程隔离 + SSE）
frontend/                  pnpm + React + Vite
```

## 快速开始

```bash
./dev.sh          # 启动（后端 :8848 + 前端 :5173）
./dev.sh stop     # 停止
./dev.sh status   # 查看状态
./dev.sh restart  # 重启
```

每个服务用 `setsid` 单独开进程组并记录 PID，所以即使它是 detached
启动的（`setsid nohup ./dev.sh &`），之后在任意 shell 里
`./dev.sh stop` 都能干净收掉整棵进程树 —— 包括 uvicorn `--reload`
fork 出的 worker 和 vite fork 出的 esbuild。

如果端口被 PID 文件之外的进程占着，`stop` 会按**工作目录**判断归属，
只回收 cwd 在本项目内的孤儿，不会误杀别的项目占着同端口的服务。

单独跑 CLI：

```bash
export PYTHONPATH=backend/src

python -m forge_core.cli datagen --out data/dual_contrast_data.jsonl --audit
python -m forge_core.cli train  --data data/dual_contrast_data.jsonl \
    --max-steps 200 --device cuda:0 --eval-every 25
python -m forge_core.cli export --checkpoint outputs/swarm_weights.pt --out outputs/export
python -m forge_core.cli publish outputs/export your-name/your-repo --dry-run
```

旧入口 `python scalpel_forge.py train` 仍可用（转发到 `forge_core.cli`）。

---

## 架构

每个 decoder 层的 MLP 被换成 `SwarmWrapper`：

```
x ─┬─► 底座 MLP（只读，恒有一个单位权重）
   └─► 大核 [1,M)  dense LoRA，按 router softmax 加权求和
   └─► 小核 N 个 rank 更低的专家，每 token 取 top-k 稀疏激活
```

底座恒定占一个单位权重，所以**路由没训好时模型退化为原底座行为**，
不会把底座改坏——这是刻意的安全设计。

---

## 本轮修复的实质问题

这些都是**不报错、只会静默毁掉训练**的类型，因此每一个都配了回归测试。

### 1. 因果 LM 忘记错位（最严重）

自己算交叉熵时漏了 `logits[t] → labels[t+1]` 的 shift，等于让模型"预测自己
已经看到的 token"。实测 loss 17~19，而 `ln(262144) = 12.5`——**比随机初始化
还差**。训练照跑、loss 照打、什么都不报错。

现在抽成 `causal_lm_loss()`，并用"完美预测器"锁定：
`train_selftest.py::test_causal_loss_shifts_logits`。

### 2. 路由目标越界（只在 N<32 时炸）

小核目标原本直接取领域全局下标 `DOMAIN_INDEX[d]`（0~31），与
`num_micro_experts` 毫无关系。只要用户把 N 配成小于 32，`cross_entropy`
就抛 device-side assert；而 C++ 断言会把栈指向错误的位置（实测报在
`repeat_interleave` 上，与真正原因毫无关系）。

早期测试全用 N=32，恰好躲过，直到 Web 层允许自由填 N 才暴露。
现在 `resolve_targets` 取模并显式校验。

### 3. 负载均衡损失梯度为 0

`f`（dispatch 占比）来自 `argmax`，本就不可微，Switch 损失的梯度**完全**
来自 `P` 项。而原实现把概率 `.detach()` 了——整个均衡项退化成常量，
指标照样打印、看着"在算"，但对 router 毫无约束。

实测后果：35 层大核分布全部塌到单一专家（`Arts_Anchor` 只剩 0.029，
而数据本身均衡的 0.25）。

### 4. 大核没有任何均衡约束

原先只给小核加均衡项。CE 只要求"预测对"，一旦某类更好预测，router 就把
概率全压过去。32 个领域映射到 4 个大核，塌一个就等于 1/4 的领域失去专属容量。
现在 `_dense_balance()` 对大核同样生效。

### 5. `[B,T,262144]` 的 logits 尖峰

vocab 262144，`B=4,T=256` 一份 bf16 logits 就是 0.5 GiB，再 `.float()` 翻倍。
改成 `losses.py::chunked_causal_ce`——只跑主干拿 `[B,T,1536]`，
把需要监督的位置切块过 `lm_head`。

⚠️ Gemma4 在 `lm_head` 之后还有 **logit softcapping**（`tanh(x/30)*30`）。
漏掉这一步 logits 会有 ±21 的偏差，loss 看着"能动"，但优化的其实是另一个
目标函数。`losses.py` 显式复刻并断言其不可省。

### 6. 导出时切自己的源码

旧实现在导出时读 `__file__`，用 `src.split('# ------')[2]` 按注释横线
计数切段再拼进 `modeling_scalpel.py`。改任何一处注释横线，切出来的就是错的
代码块，**而且不报错**。

现在改为 `shutil.copy` 真实模块，发布包里的 `swarm_forge.py` 就是
`forge_core/modeling/swarm.py` 的逐字节副本，等价性由
`export/selftest.py` 断言（前向逐位一致 + 路由 argmax 一致）。

同时把 `os.system("huggingface-cli upload ...")` 换成
`HfApi().create_repo()` / `upload_folder()`——旧的丢弃了返回码，
上传失败照样打印"🎉 发布完成"，而且仓库名直接插进 shell 字符串。

### 7. 数据集"100% 唯一"是假的

旧审计只查 `(prompt, response)` 字节级去重，然后打上"唯一率 = 100%"。
但同一个 case 换 ` ``` ` / `~~~` 围栏、换全角/半角标点各渲染一次，
字节就不同了——于是**换皮版本把配额吃光**，题面唯一率其实只有 91.4%。

现在 `sample_domain` 两轮采样：先只要"新内容"，内容穷尽了才用装饰变体补齐。
内容唯一率 89.9% → **98.2%**。审计也改成三个口径分开报。

> 题面唯一率 94.3% **不是门禁**：同一个问题配多个不同答案是 dual-contrast
> 的设计意图（教会模型"一道题有多种有效解法"，同时阻止背固定答案）。
> 真正的门禁是"重复题面的答案必须互不相同"，这条现在有断言。

### 8. 路由指标把 35 层求和后当单值报

`micro_loss` 显示 97.06 看着像炸了，其实 97.06 / 35 ≈ ln(32)。
对外报告改为每层均值，求和值另存为 `*_sum` 字段。

---

## 环境地雷

本机 `ALL_PROXY=socks://127.0.0.1:10808`，`huggingface_hub` 解析不了
`socks://` scheme，任何模型加载都会抛
`ValueError: Unknown scheme for proxy URL`。

直接跑 CLI 必须先清掉：

```bash
env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 python -m forge_core.cli train ...
```

Web 层在子进程启动前会自动清（`forge_core_bridge.sanitize_env`）。

---

## 安全边界

Web 服务**只绑 `127.0.0.1` 且没有鉴权**。这是有意的：它能读任意本地路径、
启动训练子进程、往文件系统写——等同本机的远程代码执行能力。
要对外提供服务，请放在带鉴权的反向代理后面，并把路径白名单收紧
（`FORGE_WEB_ALLOW_PATHS`）。

后端进程**从不 import torch**，连 `/api/health` 的显存都是用 `nvidia-smi`
读的——一旦它初始化了 CUDA context，就会占住一块显存与训练进程抢。

---

## 测试

```bash
export PYTHONPATH=backend/src
python -m forge_core.modeling.selftest    # 11  路由/LoRA/热调不变量
python -m forge_core.train_selftest       # 13  损失 shift / 路由目标 / 调度
python -m forge_core.export.selftest      #  7  发布包与训练期前向一致
python -m forge_core.losses               #  4  分块 CE 与整份计算逐位一致
python -m forge_web.selftest              # 13  路径穿越 / SSE / 并发上限
python -m forge_core.cli smoke            # CPU 最小闭环
```

## 实测基线

200 步、真实 6400 条数据、RTX 5090（底座与 ComfyUI 共享显存）：

| 指标 | step 0 | step 200 |
|---|---|---|
| held-out LM loss | 2.673 | **2.281** |
| 大核 CE（ln4 = 1.386） | 1.334 | 1.294 |
| 小核 CE（ln32 = 3.466） | 3.347 | 3.123 |
| 死掉的大核 / 小核 | — | 0 / 0 |
