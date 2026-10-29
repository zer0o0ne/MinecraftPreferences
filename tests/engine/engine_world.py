"""Configs of specs/engine.md for its tests: the starting values of Interface, a valid tiny
config and a writer of YAML config files; the simulator's variables of (7)."""

from pathlib import Path

import yaml

from .jars import MODULES

VARS = ("MINESTUDIO_DIR", "MINESTUDIO_GPU_RENDER")
JAR = "engine/build/libs/mcprec-6.13.jar"  # J relative to `minestudio_dir`
LWJGL_SHA256 = {  # Interface, starting values: (classes, natives) per module
    "lwjgl": ("f9928c3b4b540643a1bbd59286d3c7175e470849261a0c29a81389f52265ad8b",
              "3db6b395fbb689cb18bd27ad20055f4bc695ee93591d71d1ea45f31281c6f003"),
    "lwjgl-opengl": ("10bcc37506e01d1477d65f1fcf0aa672c95eb785265b28b7f321c8381093eda2",
                     "d8e3188d44cfbf8937dff09cb8f0a553f8701f0eb7e664a3cca70b70616d54ec"),
    "lwjgl-glfw": ("ef279cdaeede4ca344e756a56f85a85cbd039f7931011975672dc1b900b59c8f",
                   "e48917eef4178506f6d266c4528c12faf961b379f237f7779614cf5dcf51d233"),
    "lwjgl-stb": ("86a7d7af24419c8ebf1d3a2f50797acaa82aaa2abf32e3d1ebf5c31c84708dae",
                  "a8a48d4ef187322e20331a2da7d96a84a348af7300639366ffca48cadf2b5b51"),
    "lwjgl-openal": ("9b1bfdf60a3fe7ba6c48976411d311575fb2b7a7e7457e5589d1588719820fb0",
                     "617e1738b66cc5912a984dac86e083567b8f5070800c5420b065b6164d1fcaf6"),
    "lwjgl-jemalloc": ("6f790dbe775eb56be638e9d97b355313c0caf374f13545996670756a15cbfe6f",
                       "80f0adcdfbac0b78907b1a72c2ed80d415b6b865afadddaa4c10374729b2b3df"),
    "lwjgl-tinyfd": ("7a494dc8751e5ef8cb79ce7168c7db56664813782b9192bb1130e4329b914011",
                     "1b88f9bd990dc6a6b59ec914e49502cb9bb511cc1ab764cfb4bb20f15c752b95"),
}  # fmt: skip
STARTING = {
    "minestudio_dir": "data/minestudio",
    "engine_url": "https://huggingface.co/CraftJarvis/SimulatorEngine/resolve/"
    "48d4809cfddc7e2b85295e8c39b3c5e8c6d46ae7/engine.zip",
    "engine_size": 458_106_630,
    "engine_sha256": "293fac6ac72245b3365dce0e8bfbb6396fb94df29b23b6538f3bd7e2eec13ec6",
    "lwjgl_url": "https://repo1.maven.org/maven2/org/lwjgl",
    "lwjgl_version": "3.2.3",
    "lwjgl_sha256": {
        m: {"classes": c, "natives": n} for m, (c, n) in LWJGL_SHA256.items()
    },
    "http_timeout": 60,
    "http_retries": 3,
    "retry_delay": 10,
    "render_size": [640, 360],
    "seed": 0,
    "warmup_steps": 20,
    "steps": 200,
    "min_rate": 20,
}
FIELDS = tuple(STARTING)
TINY = {  # values unlike the starting ones, so that hard-coded values fail
    "minestudio_dir": "ms/sim",
    "engine_url": "http://127.0.0.1:9/hf/engine.zip",
    "engine_size": 1234,
    "engine_sha256": "0123456789abcdef" * 4,
    "lwjgl_url": "http://127.0.0.1:9/maven2/org/lwjgl",
    "lwjgl_version": "9.8.7",
    "lwjgl_sha256": {
        m: {"classes": f"{2 * i:x}" * 64, "natives": f"{2 * i + 1:x}" * 64}
        for i, m in enumerate(MODULES)
    },
    "http_timeout": 2.5,
    "http_retries": 0,
    "retry_delay": 0.0,
    "render_size": [64, 36],
    "seed": 7,
    "warmup_steps": 0,
    "steps": 1,
    "min_rate": 0.5,
}


def write_config(path: Path, mapping: dict) -> Path:
    path.write_text(yaml.safe_dump(mapping, sort_keys=False))
    return path
