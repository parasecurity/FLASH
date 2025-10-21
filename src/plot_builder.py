import os
import csv
import datetime
import argparse
import matplotlib.pyplot as plt


def generate_plot_from_csv(num_clients, folder_name):
    """
    Generates a plot using data from CSV files.
    Arguments:
        num_clients: Number of clients to generate.
        folder_name: Name of the folder where the plot will be generated.
     """

    csv_folder = f"./CSV/{folder_name}"
    model = str(folder_name.split('_')[0])

    # Initialize the main figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Round')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'F1 Score per Round for Clients and Aggregation for {model}')
    ax.grid(True)

    # Use a colormap with 256 colors
    colormap = plt.cm.viridis  # Change colormap here if desired
    colors = [colormap(i / (num_clients - 1)) for i in range(num_clients)]

    # Plot client data from CSV
    for client_id in range(num_clients):
        color = colors[client_id % len(colors)]

        client_filepath = os.path.join(csv_folder, f'{model}_client_{client_id}_data.csv')
        if os.path.exists(client_filepath):
            with open(client_filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                rounds, f1s = zip(*((int(row[0]), float(row[1])) for row in reader))
                ax.plot(rounds, f1s, marker='o', color=color, linestyle='-', label=f'Test Client {client_id}')

        train_client_filepath = os.path.join(csv_folder, f'{model}_train_client_{client_id}_data.csv')
        if os.path.exists(train_client_filepath):
            with open(train_client_filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                rounds, f1s = zip(*((int(row[0]), float(row[1])) for row in reader))
                ax.plot(rounds, f1s, marker='o', color=color, linestyle='--', label=f'Train Client {client_id}')

    # Plot aggregation data from CSV
    w_aggregation_filepath = os.path.join(csv_folder, f'{model}_weighted_aggregation_data.csv')
    if os.path.exists(w_aggregation_filepath):
        with open(w_aggregation_filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            rounds, f1s = zip(*((int(row[0]), float(row[1])) for row in reader))
            ax.plot(rounds, f1s, marker='x', linestyle='--', color='black', label='Weighted Aggregation')

    uw_aggregation_filepath = os.path.join(csv_folder, f'{model}_unweighted_aggregation_data.csv')
    if os.path.exists(uw_aggregation_filepath):
        with open(uw_aggregation_filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            rounds, f1s = zip(*((int(row[0]), float(row[1])) for row in reader))
            ax.plot(rounds, f1s, marker='+', linestyle='--', color='grey', label='Unweighted Aggregation')

    noFed_aggregation_filepath = os.path.join(csv_folder, f'{model}_nofed_data.csv')
    if os.path.exists(noFed_aggregation_filepath):
        with open(noFed_aggregation_filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                f1s = float(row[0])
            ax.axhline(y=f1s, color='red', linestyle='--', linewidth=1.5, label=f'NoFed')

    # Add legend outside the plot (to the right)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust the plot to fit the legend
    plt.tight_layout()

    # Save the plot
    plot_filename = f'{model}_plot.png'
    plot_filepath = os.path.join(csv_folder, plot_filename)
    fig.savefig(plot_filepath)
    print(f'Saved plot as {plot_filename}')

    # Close the figure to free up memory
    plt.close(fig)
    return

def main(numclients, folder_name):
    generate_plot_from_csv(numclients,folder_name)

parser = argparse.ArgumentParser(description="Specify parameters.")
parser.add_argument('--c', type=int, help='Number of clients', required=True)
parser.add_argument('--fn', type=str, help='Folder name (where to get data and save png)', required=True)

args = parser.parse_args()


if __name__ == "__main__":
    main(args.c,args.fn)