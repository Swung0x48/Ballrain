# 第三方源码（原样收录）

| 目录 | 版本 / 来源 | 许可证 | 用途 | 改动 |
|---|---|---|---|---|
| `miniz/` | miniz 3.1.0（取自 Ballanced 仓库） | MIT（`miniz/LICENSE`） | npz 读取（解压）、诊断导出的 zip | 无 |
| `zstd/` | zstd 1.5.7 单文件版（BallanceMMO 子模块里的 `build/single_file_libs/combine.py` 生成） | BSD（文件头） | 数据段的观测压缩 | 无 |
| `ftxui/` | FTXUI v7.0.3（github.com/ArthurSonzogni/FTXUI，tag v7.0.3；tarball 与 vcpkg port 的 SHA-512 一致） | MIT（`ftxui/LICENSE`） | 终端界面 `ballrain_pool` | 只收录 `CMakeLists.txt`、`cmake/`、`include/`、`src/`、`LICENSE`；`CMakeLists.txt` 里 `examples/`、`doc/` 两行改成目录存在时才加载 |
| `ed25519/` | orlp/ed25519 提交 b1f19fab（tarball 与 vcpkg port 的 SHA-512 一致） | zlib（`ed25519/license.txt`） | 发布清单签名的校验（客户端）、签名和生成密钥（`nc_release`） | 无；`seed.c`（用系统随机数）只编进 `nc_release` |
