import subprocess
import sys
import os
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

def run_command(command_list):
    """Prints and runs a command list, streaming the output."""
    command_str = " ".join(command_list)
    console.print("\n[bold cyan]Running command:[/bold cyan]")
    console.print(Syntax(command_str, "bash", theme="monokai", line_numbers=False))
    
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        for line in iter(process.stdout.readline, ''):
            console.print(line, end='')
        
        process.wait()
        if process.returncode != 0:
            console.print(f"\n[bold red]Command failed with exit code {process.returncode}[/bold red]")
    except FileNotFoundError:
        console.print(f"[bold red]Error: Command '{command_list[0]}' not found. Is your virtual environment activated? Is torchrun in your PATH?[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    console.print("\n[bold green]Command finished.[/bold green]")

def handle_build_dataset():
    """Guides the user through building a dataset."""
    console.print(Panel("[bold green]1. Build Dataset[/bold green]", subtitle="Create a new dataset from the original sources."))

    prompt_set = Prompt.ask("Which prompt set to build?", choices=["1-2", "7", "all"], default="all")
    if prompt_set == "all":
        console.print("[yellow]Note: 'all' will combine prompt sets 1-2 and 7 (set 3-6 is excluded).[/yellow]")

    output_dir = Prompt.ask("Enter the output directory name", default=f"data/asappp_{prompt_set}_combined")
    max_tokens = Prompt.ask("Enter max token length for the model", default="1024")

    limit_data = Confirm.ask("Limit the number of training essays? (for experiments)", default=False)
    limit_amount = None
    if limit_data:
        limit_amount = Prompt.ask("How many unique training essays to include?", default="1000")

    command = [
        sys.executable, "dataset/build_asappp_dataset.py",
        "--prompt-set", prompt_set,
        "--output-dir", output_dir,
        "--max-tokens", max_tokens
    ]
    if limit_amount:
        command.extend(["--limit-train-essays", limit_amount])

    run_command(command)

def handle_run_training():
    """Guides the user through running a training session."""
    console.print(Panel("[bold yellow]2. Run Training[/bold yellow]", subtitle="Start or resume a training run."))

    hardware = Prompt.ask("Which hardware are you using?", choices=["Apple Silicon", "Apple Silicon (Baseline Model)", "NVIDIA Multi-GPU"], default="Apple Silicon")
    data_path = Prompt.ask("Enter the path to your dataset directory")
    run_name = Prompt.ask("Enter a name for this run (for W&B)", default="new-run")
    eval_interval = Prompt.ask("How often (in epochs) should evaluation run?", default="5")

    # --- Resume from checkpoint ---
    resume_path = None
    if Confirm.ask("\nResume from a previous checkpoint?", default=False):
        resume_path = Prompt.ask("Enter the path to the checkpoint file (.pt)")

    # Construct command
    if hardware == "Apple Silicon":
        script_name = "train_aes_m2_regression.py"
    elif hardware == "Apple Silicon (Baseline Model)":
        script_name = "train_aes_baseline_regression.py"
    else: # NVIDIA Multi-GPU
        num_gpus = Prompt.ask("How many GPUs?", default="2")
        command = [
            "torchrun", f"--nproc_per_node={num_gpus}",
            "train_aes_h200_regression.py",
            "--data-path", data_path,
            "--eval-interval", eval_interval,
            "--use-wandb",
            "--run-name", run_name
        ]
    
    if hardware.startswith("Apple Silicon"):
        command = [
            sys.executable, script_name,
            "--data-path", data_path,
            "--eval-interval", eval_interval,
            "--use-wandb",
            "--run-name", run_name
        ]
    
    if resume_path:
        command.extend(["--resume-from-checkpoint", resume_path])

    run_command(command)

def handle_evaluate():
    """Guides the user through evaluating a checkpoint."""
    console.print(Panel("[bold blue]3. Evaluate Model[/bold blue]", subtitle="Evaluate a saved checkpoint."))

    checkpoint_path = Prompt.ask("Enter the path to the model checkpoint (.pt file)")
    data_path = Prompt.ask("Enter the path to the dataset directory it was trained on")

    command = [
        sys.executable, "evaluate_aes.py",
        "--checkpoint", checkpoint_path,
        "--data-path", data_path
    ]

    run_command(command)

def main():
    """Main function to display menu and handle user choice."""
    while True:
        console.print(Panel("[bold green]Tiny-AES Project Manager[/bold green]", subtitle="Your interactive assistant"))
        console.print("1. [bold green]Build Dataset[/bold green]")
        console.print("2. [bold yellow]Run Training[/bold yellow]")
        console.print("3. [bold blue]Evaluate Model[/bold blue]")
        console.print("4. [bold red]Exit[/bold red]")
        
        choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4"], default="1")

        if choice == '1':
            handle_build_dataset()
        elif choice == '2':
            handle_run_training()
        elif choice == '3':
            handle_evaluate()
        elif choice == '4':
            console.print("[bold]Goodbye![/bold]")
            break
        
        if Confirm.ask("\nPress Enter to return to the main menu...", default=True):
            console.clear()
        else:
            break

if __name__ == "__main__":
    main()