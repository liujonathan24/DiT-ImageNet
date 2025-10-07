import pandas as pd
import re
import matplotlib.pyplot as plt
import argparse

def main(args):
    # Read the log file
    with open(args.log_file, 'r') as f:
        log_data = f.read()

    # Regex to find lines with 'Epoch X | Step Y | Loss: Z' and capture X and Z
    # Group 1: Epoch number (\d+)
    # Group 2: Loss value ([\d\.]+)
    pattern = re.compile(r"Epoch (\d+) \|.*Loss: ([\d\.]+)")

    # List to store the extracted data
    extracted_data = []

    # Iterate through the log lines and extract data
    for line in log_data.splitlines():
        match = pattern.search(line)
        if match:
            # Convert captured groups to appropriate types
            epoch = int(match.group(1))
            loss = float(match.group(2))
            extracted_data.append({'Epoch': epoch, 'Loss': loss})

    # Create a Pandas DataFrame
    df = pd.DataFrame(extracted_data)

    # --- Plotting the data ---

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot the Loss vs. Epoch
    plt.plot(df['Epoch'], df['Loss'], marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.title('Training Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set x-axis ticks to be integers for each epoch
    if not df.empty:
        plt.xticks(df['Epoch'].unique())

    # Save the plot to a file
    plt.savefig('loss_vs_epoch.png')
    print("Plot saved to loss_vs_epoch.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", "-l", help="Path to the log file.")
    args = parser.parse_args()
    main(args)
