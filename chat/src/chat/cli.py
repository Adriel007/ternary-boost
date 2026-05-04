"""Interactive CLI chat for ternary-quantized LLMs.

Usage:
  tchat                          # Use default config
  tchat --model llama2-7b-ternary
  tchat --config                 # Interactive config editor
  tchat -m ./path/to/model --device cpu
"""

import argparse
import json
import os
import readline
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from shared.logging import get_logger

from .conversation import Conversation
from .config import (
    ChatConfig,
    ModelEntry,
    load_config,
    save_config,
    list_models,
    add_custom_model,
    MODEL_REGISTRY,
    CONFIG_DIR,
)
from .model_loader import load_model, unload_model

logger = get_logger("tchat")

BANNER = """
[bold cyan]╔══════════════════════════════════════════╗
║   TernaryBoost Chat - 1.58-bit LLM CLI   ║
╚══════════════════════════════════════════╝[/]
Type [bold]/help[/] for commands, [bold]/quit[/] to exit.

[dim]Any FP16 model → automatically compressed to ternary on first load.
Cached models load instantly on subsequent runs.[/]
"""

HELP_TEXT = """
[bold]Commands:[/]
  [bold cyan]/help[/]         Show this message
  [bold cyan]/clear[/]        Clear conversation history
  [bold cyan]/system[/] TEXT  Set system prompt
  [bold cyan]/save[/] PATH    Save conversation to file
  [bold cyan]/load[/] PATH    Load conversation from file
  [bold cyan]/config[/]       Show or edit configuration
  [bold cyan]/model[/] NAME   Switch to a different model (compresses on first use)
  [bold cyan]/models[/]       List available models
  [bold cyan]/add-model[/]    Register any HuggingFace model for compression
  [bold cyan]/thinking[/]     Toggle chain-of-thought reasoning mode
  [bold cyan]/stats[/]        Show generation statistics
  [bold cyan]/cache[/]        Show cached compressed models
  [bold cyan]/quit[/] or [bold cyan]/exit[/]  Exit

[bold]Tips:[/]
  - First load compresses the model via PT-BitNet → ParetoQ+ZeroQAT → Tequila
  - This takes 3–15 minutes depending on model size and GPU
  - Cached ternary models load instantly on subsequent runs
  - Use [bold cyan]/model[/] to switch between models — each compressed independently
  - /thinking mode adds "Let me think step by step..." prefix
"""


class ChatCLI:
    def __init__(self, config: Optional[ChatConfig] = None, model_entry: Optional[ModelEntry] = None):
        self.console = Console()
        self.config = config or load_config()
        self.conversation = Conversation(
            system_prompt=self.config.system_prompt,
            max_turns=self.config.max_turns,
        )
        self.model = None
        self.tokenizer = None
        self.model_entry = model_entry
        self.stats = {"total_tokens": 0, "total_time": 0.0, "num_generations": 0}
        self._setup_readline()

    def _setup_readline(self) -> None:
        histfile = CONFIG_DIR / "history"
        if histfile.exists():
            readline.read_history_file(str(histfile))

    def _save_history(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(CONFIG_DIR / "history"))

    def run(self) -> None:
        self.console.print(BANNER)
        self.console.print(HELP_TEXT)

        try:
            self._ensure_model_loaded()
            self._repl()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted.[/]")
        except Exception as e:
            self.console.print(f"\n[red]Fatal error: {e}[/]")
            logger.exception("Fatal error in chat loop")
        finally:
            self._cleanup()

    def _repl(self) -> None:
        while True:
            try:
                prompt = self._get_prompt()
                user_input = self.console.input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Goodbye.[/]")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                result = self._handle_command(user_input)
                if result == "quit":
                    break
                continue

            self.conversation.add("user", user_input)
            self._generate_response()

    def _get_prompt(self) -> str:
        model_name = self.model_entry.name if self.model_entry else self.config.model
        thinking_indicator = " [dim]⟐think[/]" if self.config.enable_thinking else ""
        speed_indicator = self._detect_speed_mode()
        turn = self.conversation.turn_count + 1
        return f"[bold green]▸ {model_name}{speed_indicator}{thinking_indicator} [{turn}][/] "

    def _detect_speed_mode(self) -> str:
        """Detect whether ternary optimization is active."""
        if self.model is None:
            return ""
        for module in self.model.modules():
            name = type(module).__name__
            if name == "UltraQuantLinear":
                return " [bold green][1.58b][/]"
            elif name == "QuantizeLinear":
                return " [bold yellow][1.58b][/]"
        # After baking: nn.Linear with ternary weights → show blue
        has_linear = any(isinstance(m, torch.nn.Linear) for m in self.model.modules())
        if has_linear:
            return " [bold blue][1.58b-baked][/]"
        return ""

    def _generate_response(self) -> None:
        if self.model is None and self.model_entry.backend != "bitnet_cpp":
            self.console.print("[red]No model loaded. Use /model to switch.[/]")
            return

        gen_kwargs = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_new_tokens": self.config.max_new_tokens,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": self.config.temperature > 0,
        }

        if self.config.enable_thinking:
            self.conversation.add("thinking", self.config.thinking_prefix)

        prompt_text = self.conversation.to_prompt(self.tokenizer, self.config.enable_thinking)

        try:
            if self.model_entry.backend == "bitnet_cpp":
                response = self._generate_bitnet(prompt_text, gen_kwargs)
            else:
                response = self._generate_transformers(prompt_text, gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            self.console.print("[red]CUDA out of memory. Try:[/]")
            self.console.print("  /config → set max_length lower (e.g., 1024)")
            self.console.print("  /model <name> --device cpu")
            self.conversation.clear()
            return

        if self.config.enable_thinking:
            self.conversation.messages.pop()

        self.conversation.add("assistant", response)
        self.console.print()

        if self.config.show_stats and self.stats["num_generations"] > 0:
            avg_time = self.stats["total_time"] / self.stats["num_generations"]
            self.console.print(
                f"[dim]({len(response.split())} words, "
                f"{len(response)} chars, "
                f"{avg_time:.1f}s avg gen time)[/]"
            )

    def _generate_transformers(self, prompt: str, gen_kwargs: dict) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.model_entry.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        start_time = time.time()
        self.console.print()

        if self.config.stream:
            response = self._stream_generate(inputs, gen_kwargs)
        else:
            with self.console.status("[cyan]Generating...[/]", spinner="dots"):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **gen_kwargs,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            self.console.print(Panel(Markdown(response.strip()), border_style="cyan"))

        elapsed = time.time() - start_time
        self.stats["total_time"] += elapsed
        self.stats["total_tokens"] += len(response.split())
        self.stats["num_generations"] += 1

        return response.strip()

    def _stream_generate(self, inputs: dict, gen_kwargs: dict) -> str:
        from transformers import TextStreamer, TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs.update({
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **inputs,
        })

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        first_token = True

        for text in streamer:
            if first_token:
                self.console.print()
                self.console.print("[bold cyan]▌[/] ", end="")
                first_token = False
            accumulated += text
            self.console.print(text, end="", markup=False)

        thread.join()
        return accumulated.strip()

    def _generate_bitnet(self, prompt: str, gen_kwargs: dict) -> str:
        """Generate using external bitnet.cpp binary."""
        import subprocess
        import tempfile

        bitnet_bin = shutil.which("bitnet")
        if not bitnet_bin:
            raise RuntimeError(
                "bitnet.cpp binary not found. Install from https://github.com/microsoft/BitNet"
            )

        model_path = os.path.join(self.model_entry.path, "model.bitnet")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            start_time = time.time()
            self.console.print()

            with self.console.status("[cyan]Generating (CPU)...[/]", spinner="dots"):
                result = subprocess.run(
                    [bitnet_bin, model_path, "-f", prompt_file,
                     "-n", str(gen_kwargs["max_new_tokens"]),
                     "-t", str(gen_kwargs["temperature"])],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

            response = result.stdout.strip()
            elapsed = time.time() - start_time

            self.console.print(Panel(Markdown(response), border_style="cyan"))

            self.stats["total_time"] += elapsed
            self.stats["total_tokens"] += len(response.split())
            self.stats["num_generations"] += 1

            return response
        except subprocess.TimeoutExpired:
            self.console.print("[red]Generation timed out (300s).[/]")
            return "[Generation timed out]"
        finally:
            os.unlink(prompt_file)

    def _ensure_model_loaded(self) -> None:
        if self.model_entry is None:
            registry = list_models()
            if self.config.model in registry:
                self.model_entry = registry[self.config.model]
            else:
                self.console.print(
                    f"[yellow]Model '{self.config.model}' not found in registry.[/]"
                )
                available = list(registry.keys())
                if available:
                    self.console.print(f"Available: {', '.join(available)}")
                    self.config.model = available[0]
                    self.model_entry = registry[available[0]]
                    self.console.print(f"[green]Using: {self.config.model}[/]")
                else:
                    self.console.print("[red]No models registered. Use /add-model[/]")
                    return

        self.console.print(
            f"\n[bold]Model:[/] {self.model_entry.name} "
            f"([dim]{self.model_entry.path}[/])"
        )
        self.console.print(f"[dim]Cache: {Path(self.config.cache_dir).resolve()}/[/]")

        self.model, self.tokenizer = load_model(self.model_entry, self.config.cache_dir)

        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.conversation.set_system(self.config.system_prompt)
        self.console.print("[green]Ready.[/]\n")

    def _handle_command(self, cmd: str) -> Optional[str]:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/system": self._cmd_system,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/config": self._cmd_config,
            "/model": self._cmd_model,
            "/models": self._cmd_models,
            "/add-model": self._cmd_add_model,
            "/thinking": self._cmd_thinking,
            "/stats": self._cmd_stats,
            "/cache": self._cmd_cache,
            "/quit": lambda _: "quit",
            "/exit": lambda _: "quit",
        }

        handler = handlers.get(command)
        if handler:
            return handler(args)
        else:
            self.console.print(f"[yellow]Unknown command: {command}[/]")
            self.console.print("Type [bold]/help[/] for available commands.")
            return None

    def _cmd_help(self, _: str) -> None:
        self.console.print(HELP_TEXT)

    def _cmd_clear(self, _: str) -> None:
        self.conversation.clear()
        self.console.print("[green]Conversation cleared.[/]")

    def _cmd_system(self, prompt: str) -> None:
        if not prompt:
            current = self.config.system_prompt
            self.console.print(f"[dim]Current system prompt:[/]\n{current}")
            return
        self.config.system_prompt = prompt
        self.conversation.set_system(prompt)
        self.console.print(f"[green]System prompt set.[/]")

    def _cmd_save(self, path: str) -> None:
        path = path or f"chat_{int(time.time())}.json"
        data = {
            "config": self.config.to_dict(),
            "messages": [
                {"role": m.role, "content": m.content}
                for m in self.conversation.messages
            ],
            "stats": self.stats,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.console.print(f"[green]Conversation saved to {path}[/]")

    def _cmd_load(self, path: str) -> None:
        if not path:
            self.console.print("[yellow]Usage: /load PATH[/]")
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.conversation = Conversation(
            system_prompt=data.get("config", {}).get("system_prompt", ""),
            max_turns=self.config.max_turns,
        )
        for msg in data.get("messages", []):
            self.conversation.add(msg["role"], msg["content"])
        self.console.print(f"[green]Loaded {len(data.get('messages', []))} messages from {path}[/]")

    def _cmd_config(self, _: str) -> None:
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        for key, value in self.config.to_dict().items():
            table.add_row(key, str(value))
        self.console.print(table)
        self.console.print("[dim]To edit, run: tchat --config[/]")

    def _cmd_model(self, args: str) -> None:
        registry = list_models()

        if not args:
            current = self.config.model
            self.console.print(f"[dim]Current model: {current}[/]")
            self.console.print("Available models:")
            for name, entry in registry.items():
                marker = " [bold cyan]◀[/]" if name == current else ""
                self.console.print(f"  {name} ({entry.description}){marker}")
            self.console.print("\n[dim]Use /model NAME to switch[/]")
            return

        switch_args = args.split()
        name = switch_args[0]
        device = None

        for i, arg in enumerate(switch_args):
            if arg == "--device" and i + 1 < len(switch_args):
                device = switch_args[i + 1]

        if name not in registry:
            self.console.print(f"[red]Model '{name}' not found.[/]")
            self.console.print(f"Available: {', '.join(registry.keys())}")
            self.console.print("Use [bold]/add-model[/] to register a custom model.")
            return

        self._switch_model(registry[name], device)

    def _cmd_models(self, _: str) -> None:
        registry = list_models()
        self.console.print(f"[bold]Registered models ({len(registry)}):[/]\n")
        for name, entry in registry.items():
            self.console.print(
                f"  [bold cyan]{name}[/]")
            self.console.print(f"    path: {entry.path}")
            self.console.print(f"    backend: {entry.backend}, device: {entry.device}")
            self.console.print(f"    {entry.description}")
            self.console.print()

    def _cmd_add_model(self, _: str) -> None:
        self.console.print("[bold]Add Custom Model[/]\n")

        name = self.console.input("  Name (e.g., my-llama-ternary): ").strip()
        if not name:
            self.console.print("[yellow]Cancelled.[/]")
            return

        path = self.console.input("  Path to model directory: ").strip()
        if not path or not os.path.exists(path):
            self.console.print(f"[red]Path not found: {path}[/]")
            return

        backend = self.console.input("  Backend [transformers/bitnet_cpp] (default: transformers): ").strip()
        backend = backend or "transformers"

        device = self.console.input("  Device [cuda/cpu] (default: cuda): ").strip()
        device = device or "cuda"

        description = self.console.input("  Description: ").strip()

        entry = ModelEntry(
            name=name,
            path=path,
            backend=backend,
            description=description,
            device=device,
        )
        add_custom_model(entry)
        self.console.print(f"[green]Model '{name}' added to registry.[/]")
        self.console.print(f"Use [bold]/model {name}[/] to switch.")

    def _cmd_thinking(self, _: str) -> None:
        self.config.enable_thinking = not self.config.enable_thinking
        status = "ON" if self.config.enable_thinking else "OFF"
        self.console.print(f"[bold]Thinking mode: [{'green' if self.config.enable_thinking else 'red'}]{status}[/][/]")
        if self.config.enable_thinking:
            self.console.print(
                f"[dim]Prefix: \"{self.config.thinking_prefix.strip()}\"[/]"
            )
            self.console.print(
                "[dim]The model will generate reasoning before the final answer.[/]"
            )

    def _cmd_stats(self, _: str) -> None:
        table = Table(title="Generation Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total generations", str(self.stats["num_generations"]))
        table.add_row("Total tokens", str(self.stats["total_tokens"]))
        table.add_row("Total time", f"{self.stats['total_time']:.2f}s")
        if self.stats["num_generations"] > 0:
            avg_tokens = self.stats["total_tokens"] / self.stats["num_generations"]
            avg_time = self.stats["total_time"] / self.stats["num_generations"]
            table.add_row("Avg tokens/gen", f"{avg_tokens:.1f}")
            table.add_row("Avg time/gen", f"{avg_time:.2f}s")
            table.add_row("Tokens/sec", f"{avg_tokens / max(avg_time, 0.01):.1f}")
        table.add_row("Conversation turns", str(self.conversation.turn_count))
        table.add_row("Est. context tokens", str(self.conversation.token_estimate))
        table.add_row("Thinking mode", "ON" if self.config.enable_thinking else "OFF")
        self.console.print(table)

    def _cmd_cache(self, _: str) -> None:
        from .model_loader import list_cache, clear_cache
        cache_root = Path(self.config.cache_dir).resolve()
        entries = list_cache(str(cache_root))
        if not entries:
            self.console.print("[dim]No cached models. Models are cached after first compression.[/]")
            self.console.print(f"Cache directory: {cache_root}/ternary/")
            return
        self.console.print(f"[bold]Cached ternary models ({len(entries)}):[/]\n")
        for e in entries:
            name = Path(e.get("source_model", e["path"])).name
            mins = e.get("total_time_s", 0) / 60
            params_m = e.get("params", 0) / 1e6
            self.console.print(
                f"  [bold cyan]{name}[/] — {params_m:.0f}M params, "
                f"compressed in {mins:.1f}min"
            )
        self.console.print(f"\n[dim]Cache: {cache_root}/ternary/[/]")

    def _switch_model(self, entry: ModelEntry, device: Optional[str] = None) -> None:
        if device:
            entry.device = device

        self.console.print(f"[yellow]Switching to {entry.name}...[/]")

        if self.model is not None:
            unload_model(self.model)

        self.model_entry = entry
        self.config.model = entry.name

        self.console.print(f"[dim]Loading {entry.name}...[/]")
        self.model, self.tokenizer = load_model(entry, self.config.cache_dir)

        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        save_config(self.config)
        self.console.print(f"[green]Model switched to {entry.name}[/]")

    def _cleanup(self) -> None:
        self._save_history()
        save_config(self.config)
        if self.model is not None:
            unload_model(self.model)

        if self.stats["num_generations"] > 0:
            avg_time = self.stats["total_time"] / self.stats["num_generations"]
            self.console.print(
                f"\n[dim]Session: {self.stats['num_generations']} gens, "
                f"{self.stats['total_tokens']} tokens, "
                f"{avg_time:.1f}s avg[/]"
            )


def interactive_config() -> ChatConfig:
    """Run interactive configuration editor."""
    console = Console()
    config = load_config()

    console.print("[bold]Interactive Configuration[/]\n")

    registry = list_models()
    config.model = console.input(
        f"  Model [{config.model}]: "
    ).strip() or config.model

    config.cache_dir = console.input(
        f"  Cache dir [{config.cache_dir}]: "
    ).strip() or config.cache_dir

    config.system_prompt = console.input(
        f"  System prompt [{config.system_prompt[:50]}...]: "
    ).strip() or config.system_prompt

    max_turns = console.input(f"  Max turns [{config.max_turns}]: ").strip()
    config.max_turns = int(max_turns) if max_turns else config.max_turns

    temp = console.input(f"  Temperature [{config.temperature}]: ").strip()
    config.temperature = float(temp) if temp else config.temperature

    top_p = console.input(f"  Top-p [{config.top_p}]: ").strip()
    config.top_p = float(top_p) if top_p else config.top_p

    max_tokens = console.input(f"  Max new tokens [{config.max_new_tokens}]: ").strip()
    config.max_new_tokens = int(max_tokens) if max_tokens else config.max_new_tokens

    streaming = console.input(f"  Stream output [Y/n]: ").strip().lower()
    config.stream = streaming != "n"

    thinking = console.input(f"  Enable thinking/COT mode [y/N]: ").strip().lower()
    config.enable_thinking = thinking == "y"

    save_config(config)
    console.print(f"\n[green]Configuration saved to {CONFIG_DIR}/config.json[/]")
    return config


def main():
    parser = argparse.ArgumentParser(description="TernaryBoost Chat CLI")
    parser.add_argument("--model", "-m", type=str, help="Model name or path")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--config", "-c", action="store_true", help="Edit configuration")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory (default: ./cache)")
    parser.add_argument("--lambada-granularity", type=str, default=None,
                        choices=["per_channel", "per_element"],
                        help="Lambada granularity: per_channel (~6 MB RAM) or per_element (~2.5 GB)")

    args = parser.parse_args()

    if args.list_models:
        console = Console()
        for name, entry in list_models().items():
            console.print(f"  [bold]{name}[/] — {entry.description}")
        return

    if args.config:
        interactive_config()
        return

    config = load_config()
    if args.no_stream:
        config.stream = False
    if args.cache_dir:
        config.cache_dir = args.cache_dir

    model_entry = None
    if args.model:
        registry = list_models()
        if args.model in registry:
            model_entry = registry[args.model]
            if args.device:
                model_entry.device = args.device
        else:
            model_entry = ModelEntry(
                name=os.path.basename(args.model),
                path=args.model,
                backend="transformers",
                description="Custom model (CLI argument)",
                device=args.device or "cuda",
            )
    if model_entry and args.lambada_granularity:
        model_entry.lambada_granularity = args.lambada_granularity

    cli = ChatCLI(config=config, model_entry=model_entry)
    cli.run()


if __name__ == "__main__":
    main()
