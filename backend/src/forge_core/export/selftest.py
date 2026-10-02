"""发布包等价性自检。

发布包里的 ``swarm_forge.py`` 是 :mod:`forge_core.modeling.swarm` 的
逐字节副本。副本与本体一旦漂移（例如有人只改了训练侧），
用户 ``from_pretrained`` 出来的模型就会和训练结果不一致——
而且**不会报错**，只会安静地给出不同的路由。

所以这里断言两件事：

1. **字节一致**：发布包里的 ``swarm_forge.py`` 与训练侧模块逐字节相同；
2. **前向一致**：给两组相同的权重与输入，两边的输出逐位相同、
   路由 argmax 相同。

第 2 条比第 1 条更强，即使将来打包方式变了也仍然有效。
"""

from __future__ import annotations

import hashlib
import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from ..modeling import SwarmWrapper, wrap_layers
from ..modeling.swarm import ScalpelLoRA


H = 32
M, N = 4, 6
TOP_K = 2


class _MLP(nn.Module):
    """底座 MLP 的最小替身。"""

    def __init__(self, h: int = H, i: int = H * 2) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self, h: int = H) -> None:
        super().__init__()
        self.mlp = _MLP(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


def _build(seed: int) -> tuple[nn.Module, list[SwarmWrapper]]:
    torch.manual_seed(seed)
    model = nn.Sequential(_Block(), _Block())
    wrappers = wrap_layers(
        model, list(model), num_macro=M, num_micro=N, micro_top_k=TOP_K,
        hidden_dim=H, param_dtype=torch.float32, device="cpu")
    model.eval()
    return model, wrappers


def _randomize(wrappers: list[SwarmWrapper]) -> None:
    """把零初始化的 LoRA_B 随机化——否则两边都是恒等映射，测不出差异。"""
    g = torch.Generator().manual_seed(1234)
    for w in wrappers:
        for p in w.parameters():
            if p.abs().sum() == 0:
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)


_BUNDLE_DIR: Path | None = None


def _make_bundle(tmp: Path) -> Path:
    """造一个真实发布包（走 build_bundle，而不是手工拼文件）。"""
    import torch as _t

    from ..schema import ArchProfile, ForgeConfig
    from .bundle import build_bundle

    cfg = ForgeConfig(project_name="selftest", base_model_id="dummy/base")
    cfg.arch = ArchProfile(
        num_macro_cores=M, macro_names=[f"M{i}" for i in range(M)],
        num_micro_experts=N, macro_rank=4, micro_rank=2)

    _, wrap = _build(7)
    sd = {"__meta__": {"step": 3, "project": "selftest",
                       "base_model_id": "dummy/base", "hidden_dim": H,
                       "arch": cfg.arch.model_dump(mode="json"),
                       "format": "forge-swarm-v2"}}
    for i, w in enumerate(wrap):
        sd[f"layer{i}.router_macro"] = w.router_macro.state_dict()
        sd[f"layer{i}.router_micro"] = w.router_micro.state_dict()
    ckpt = tmp / "ckpt.pt"
    _t.save(sd, ckpt)

    out = tmp / "bundle"
    build_bundle(cfg, ckpt, out)
    return out


def _bundle() -> Path:
    """整个模块共用一个发布包，避免反复搭。"""
    global _BUNDLE_DIR
    if _BUNDLE_DIR is None:
        tmp = Path(tempfile.mkdtemp(prefix="forge-bundle-"))
        _BUNDLE_DIR = _make_bundle(tmp)
    return _BUNDLE_DIR


def _load_copy(bundle: Path):
    """把发布包里的副本当独立模块导入。"""
    sys.path.insert(0, str(bundle))
    try:
        import configuration_scalpel            # noqa: F401
        import swarm_forge
        return swarm_forge
    finally:
        sys.path.remove(str(bundle))


def test_files_are_byte_identical() -> None:
    """发布包里的路由实现必须与训练侧逐字节相同。"""
    train_side = Path(__file__).resolve().parent.parent / "modeling" / "swarm.py"
    export_side = _bundle() / "swarm_forge.py"
    assert export_side.exists(), f"发布包里缺少 {export_side}"

    a = hashlib.sha256(train_side.read_bytes()).hexdigest()
    b = hashlib.sha256(export_side.read_bytes()).hexdigest()
    assert a == b, (
        "发布包的 swarm_forge.py 与 forge_core.modeling.swarm 已经漂移！\n"
        f"  训练侧 sha256={a[:16]}…\n"
        f"  发布侧 sha256={b[:16]}…\n"
        "请重新执行 export（打包是拷贝而非手工维护两份）。")


def test_forward_matches_copy() -> None:
    """训练侧与发布侧的前向输出、路由 argmax 必须逐位一致。"""
    bundle = _bundle()
    with tempfile.TemporaryDirectory() as td:
        mod = _load_copy(bundle)

        model_a, wrap_a = _build(7)
        model_b = nn.Sequential(_Block(), _Block())
        wrap_b = wrap_layers(
            model_b, list(model_b), num_macro=M, num_micro=N,
            micro_top_k=TOP_K, hidden_dim=H, param_dtype=torch.float32,
            device="cpu")
        model_b.eval()

        _randomize(wrap_a)
        _randomize(wrap_b)

        # 底座也必须一致，否则差异会被误记到路由头上
        model_b.load_state_dict(model_a.state_dict(), strict=False)
        for wa, wb in zip(wrap_a, wrap_b):
            wb.load_state_dict(wa.state_dict())

        torch.manual_seed(99)
        x = torch.randn(2, 7, H)

        with torch.no_grad():
            ya = model_a(x)
            am_a = wrap_a[0].last_macro_logits.argmax(-1)
            at_a = wrap_a[0].last_micro_topk
            yb = model_b(x)
            am_b = wrap_b[0].last_macro_logits.argmax(-1)
            at_b = wrap_b[0].last_micro_topk

        assert torch.equal(ya, yb), (
            f"前向不一致，最大偏差 {float((ya - yb).abs().max())}")
        assert torch.equal(am_a, am_b), "大核路由 argmax 不一致"
        assert torch.equal(at_a, at_b), "小核 top-k 不一致"

        # 副本是**另一个模块对象**，所以它的 SwarmWrapper 与训练侧的
        # 不是同一个类（issubclass 必然为 False，源码相同也一样）。
        # 因此只能核对公开符号与构造签名，不能核对类身份。
        public = {"ScalpelLoRA", "SwarmWrapper", "locate_layers",
                  "wrap_layers", "hidden_dim_of", "strip_multimodal"}
        missing = public - set(dir(mod))
        assert not missing, f"发布包缺少符号：{missing}"
        assert (inspect.signature(mod.SwarmWrapper.__init__)
                == inspect.signature(SwarmWrapper.__init__)), \
            "SwarmWrapper 构造签名与训练侧不一致"
        assert (inspect.signature(mod.ScalpelLoRA.__init__)
                == inspect.signature(ScalpelLoRA.__init__)), \
            "ScalpelLoRA 构造签名与训练侧不一致"


def test_no_relative_imports() -> None:
    """发布包模块不得含 ``from .`` 相对导入。

    它们会被拷到模型仓库根目录由 ``sys.path`` 解析，
    相对导入在那种加载方式下必然 ImportError。
    """
    bundle = _bundle()
    for name in ("swarm_forge", "telemetry", "configuration_scalpel",
                 "modeling_scalpel"):
        text = (bundle / f"{name}.py").read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("from .", "from ..")):
                raise AssertionError(
                    f"{name}.py:{i} 含相对导入：{stripped!r}\n"
                    "发布包模块必须只依赖 torch / transformers / 标准库。")


def test_bundle_roundtrip() -> None:
    """build_bundle 产出的目录必须包含全部必需文件。"""
    import json

    info = None
    with tempfile.TemporaryDirectory() as td:
        out = _make_bundle(Path(td))
        info = {"files": sorted(p.name for p in out.iterdir())}

        need = {"config.json", "swarm_weights.pt", "modeling_scalpel.py",
                "configuration_scalpel.py", "swarm_forge.py", "telemetry.py",
                "README.md"}
        assert need <= set(info["files"]), need - set(info["files"])

        conf = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert conf["hidden_dim"] == H, "hidden_dim 必须来自 checkpoint"
        assert conf["num_micro_experts"] == N
        assert conf["auto_map"]["AutoModelForCausalLM"] == \
            "modeling_scalpel.ScalpelUniversalForCausalLM"
        assert conf["architectures"] == ["ScalpelUniversalForCausalLM"]


def test_bundle_rejects_bad_checkpoint() -> None:
    """格式不对的检查点必须报错，而不是导出一个半残的包。"""
    import torch as _t

    from ..schema import ForgeConfig
    from .bundle import build_bundle

    cfg = ForgeConfig(project_name="x", base_model_id="dummy/base")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bad = tmp / "bad.pt"
        _t.save({"__meta__": {"format": "legacy-scalpel"}}, bad)
        try:
            build_bundle(cfg, bad, tmp / "out")
        except ValueError as e:
            assert "格式" in str(e), e
        else:
            raise AssertionError("格式不对的检查点竟然被接受了")

        missing_hd = tmp / "nohd.pt"
        _t.save({"__meta__": {"format": "forge-swarm-v2"}}, missing_hd)
        try:
            build_bundle(cfg, missing_hd, tmp / "out2")
        except ValueError as e:
            assert "hidden_dim" in str(e), e
        else:
            raise AssertionError("缺 hidden_dim 的检查点竟然被接受了")


def test_publish_dry_run_no_network() -> None:
    """dry-run 必须纯本地，且能查出残缺的包。"""
    from .publish import publish

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            publish(tmp, "me/model", dry_run=True)
        except RuntimeError as e:
            assert "不完整" in str(e), e
        else:
            raise AssertionError("残缺的发布包竟然通过了 dry-run")

        for f in ("config.json", "swarm_weights.pt", "modeling_scalpel.py",
                  "configuration_scalpel.py", "swarm_forge.py",
                  "telemetry.py"):
            (tmp / f).write_text("{}", encoding="utf-8")

        plan = publish(tmp, "me/model", dry_run=True)
        assert plan["dry_run"] is True
        assert plan["repo_id"] == "me/model"


def test_scalpel_lora_zero_init_is_identity() -> None:
    """零初始化的 lora_B 必须让 LoRA 严格等于恒等映射。

    这是"未训练的 swarm 不会破坏底座"的前提。
    """
    lora = ScalpelLoRA(H, 4)
    x = torch.randn(3, H)
    assert torch.equal(lora(x), torch.zeros_like(x))


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                                  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
