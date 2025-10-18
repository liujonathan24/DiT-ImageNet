import argparse
import re
import matplotlib.pyplot as plt
import os

def main():
    """Parses a log file and plots the average loss per epoch."""
    parser = argparse.ArgumentParser(description="Plot average loss per epoch from a log file.")
    parser.add_argument("log_file", type=str, help="Path to the training log file.")
    parser.add_argument(
        "--output_path", '-o',
        type=str, 
        default="epoch_loss_plot.png", 
        help="Path to save the output plot image."
    )
    args = parser.parse_args()

    epochs = []
    losses = []

    # Regex to find lines with the format: "... Epoch X Average Loss: Y.YYYY"
    pattern = re.compile(r"Epoch (\d+) Average Loss: (\d+\.\d+)")

    print(f"Reading log file: {args.log_file}")
    try:
        with open(args.log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    epochs.append(epoch)
                    losses.append(loss)
    except FileNotFoundError:
        print(f"Error: Log file not found at {args.log_file}")
        return

    if not epochs:
        print("No epoch loss data found in the log file. Nothing to plot.")
        return

    print(f"Found data for {len(epochs)} epochs.")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.title("Average Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output_path)
    print(f"Plot saved to {os.path.abspath(args.output_path)}")

if __name__ == "__main__":
    main()
