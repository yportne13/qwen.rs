# qwen.rs

一个非官方的使用 rust 实现的[通义千问-7B-chat](https://github.com/QwenLM/Qwen)，能够在 cpu 上运行，不需要任何依赖。基于 [candle](https://github.com/huggingface/candle) 和 [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs)，大量参考了 candle 中的 llama 的例子。

实现了一个原始版本和一个量化版本（目前量化版本还未调通，基于 ggml，官方的转换脚本似乎是早期版本，candle 不支持）。原始版本需要下载官方的权重文件（[Hugging Face](https://huggingface.co/Qwen/Qwen-7B-Chat/tree/main) 或 [ModelScope](https://modelscope.cn/models/qwen/Qwen-7B-Chat/files)）下的八个后缀为 `.safetensors` 的文件。量化版本需要下载的权重文件还在调。

## 使用方法

在你想要运行这个程序的目录下创建 weight 文件夹，将权重文件放入。

在 [release](https://github.com/yportne13/qwen.rs/releases/tag/v1-26f5754) 页面下载可执行文件，windows 就下 .exe 后缀的，linux 就下另一个，mac 的 ci 出了点问题没有导出。当在命令行看到 `>>>` 就可以开始对话了。如果闪退了一般是权重文件没有正确放置。

如果想要自行编译运行，下载安装 rust，然后克隆当前项目，并在当前项目下命令行输入 `cargo run --release` 即可。想生成可执行文件则是 `cargo build --release`，然后即可在 `target/release` 目录下找到可执行程序。

## 使用协议

模型本身的协议参见[官方仓库的说明](https://github.com/QwenLM/Qwen/blob/main/README_CN.md#%E4%BD%BF%E7%94%A8%E5%8D%8F%E8%AE%AE)。本仓库的代码为 Apache 和 MIT 协议。
