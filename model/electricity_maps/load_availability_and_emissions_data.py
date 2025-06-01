from datetime import timedelta
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Global base date
BASE_DATE = pd.Timestamp("2023-01-15 00:00:00", tz="UTC")

# ----------------------- Carbon Intensity Functions -----------------------

def load_carbon_intensity_data(file_path: str) -> pd.DataFrame:
    """
    Loads and processes the carbon intensity:
    """
    year_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    raw = year_df["Carbon Intensity gCO₂eq/kWh (direct)"].reset_index()
    raw = raw.rename(columns={"Datetime (UTC)": "ds", "Carbon Intensity gCO₂eq/kWh (direct)": "y"})
    raw["ds"] = pd.to_datetime(raw["ds"], utc=True)  # Ensure `ds` is of datetime64[ns, UTC] type
    raw["y"] = raw["y"].ffill()
    return raw


def get_next_100_hours_data_co(raw_data: pd.DataFrame) -> np.ndarray:
    """
    Retrieves the next 100 hours of carbon intensity probing based on the global base date.
    """
    next_100_hours = BASE_DATE + timedelta(hours=200)
    next_100_data = raw_data[(raw_data['ds'] >= BASE_DATE) & (raw_data['ds'] < next_100_hours)]

    if len(next_100_data) < 200:
        remaining_hours = 200 - len(next_100_data)
        start_data = raw_data.iloc[:remaining_hours]
        next_100_data = pd.concat([next_100_data, start_data])
    return next_100_data['y'].to_numpy()


def get_carbon_intensities_for_all_regions(limit=30) -> dict:
    """
    Loads and processes carbon intensity for all regions, ensuring consistent region selection.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'carbon_intensities_data')
    region_data = {}
    try:
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv') and f"_2023_" in f])  # Sorted alphabetically
        selected_files = all_files[:limit]  # Take the first limit regions consistently

        for filename in selected_files:
            region = filename.split('_')[0]
            file_path = os.path.join(data_dir, filename)
            raw_data = load_carbon_intensity_data(file_path)
            next_100_hours_data = get_next_100_hours_data_co(raw_data)
            region_data[region] = next_100_hours_data

        return region_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

# ----------------------- Availability/Curtailment Functions -----------------------

def load_availability() -> pd.DataFrame:
    """
    Loads and processes the curtailment availability.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "availability.csv")

    availability_df = pd.read_csv(file_path, index_col=0, parse_dates=[0], delimiter=',')
    availability_df.index = pd.to_datetime(availability_df.index, utc=True)  # Ensure index is datetime64[ns, UTC]
    availability_df = availability_df.ffill()
    return availability_df


def get_next_100_hours_curtailment(limit=30) -> dict:
    """
    Gets the next 100 hours of curtailment.
    """
    availability_data = load_availability()
    next_100_hours = BASE_DATE + timedelta(hours=200)

    hourly_data = availability_data[availability_data.index.minute == 0]
    next_100_availability = hourly_data[(hourly_data.index >= BASE_DATE) &
                                        (hourly_data.index < next_100_hours)]

    if len(next_100_availability) < 200:
        remaining_positions = 200 - len(next_100_availability)
        start_availability = hourly_data.iloc[:remaining_positions]
        next_100_availability = pd.concat([next_100_availability, start_availability])

    # Limit to 30 regions in alphabetical order
    all_regions = sorted(next_100_availability.columns.tolist())[:limit]
    next_100_availability = next_100_availability[all_regions]

    availability_dict = {
        region: next_100_availability[region].to_numpy()
        for region in next_100_availability.columns
    }

    return availability_dict

# ----------------------- Combination Function -----------------------

def combine_client_data(curtailment_data, intensity_data, curtailment_threshold):
    """
    Combine curtailment and intensity probing for regions, using indices to align probing.
    """
    curtailment_regions = list(curtailment_data.keys())
    intensity_regions = list(intensity_data.keys())

    num_regions = min(len(curtailment_regions), len(intensity_regions))

    combined_data = {}

    for i in range(num_regions):
        curtailment_array = np.array(curtailment_data[curtailment_regions[i]])
        intensity_array = np.array(intensity_data[intensity_regions[i]])

        if curtailment_array.size == 0 or intensity_array.size == 0:
            print(f"Skipping region at index {i} due to empty probing.")
            continue

        reformatted_curtailment = np.where(curtailment_array < curtailment_threshold, 1, 0)
        combined_array = np.where(reformatted_curtailment == 1, intensity_array, reformatted_curtailment)

        combined_data[curtailment_regions[i]] = combined_array

    return combined_data

# ----------------------- Plotting Function -----------------------

def plot_heatmap(data_dict, title, name=None):
    plt.figure(figsize=(12, 8))
    regions = list(data_dict.keys())
    data_matrix = np.array(list(data_dict.values()))
    plt.imshow(data_matrix, aspect="auto", cmap="magma", interpolation="nearest")
    plt.colorbar(label="Values")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Regions")
    plt.yticks(ticks=np.arange(len(regions)), labels=regions)
    plt.title(title)
    plt.tight_layout()
    if name:
        plt.savefig(f'plots/{name}_heatmap.png')
    else:
        plt.show()

def plot_combined_heatmap(combined_data: dict, output_name="combined_heatmap"):
    """
    Plot a heatmap from combined curtailment and carbon intensity data.
    """
    if not combined_data:
        print("[ERROR] No combined data to plot.")
        return

    regions = list(combined_data.keys())
    data_matrix = np.array([combined_data[r] for r in regions])

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(data_matrix, aspect="auto", cmap="magma", interpolation="nearest")

    ax.set_xlabel("Time (Hours)", fontsize=14)
    ax.set_ylabel("Regions", fontsize=14)
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("gCO₂eq/kWh or Curtailment Flag", fontsize=14)

    ax.set_title("Combined Curtailment + Carbon Intensity Probing", fontsize=16, pad=12)

    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/{output_name}.pdf"
    plt.savefig(plot_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Combined heatmap saved to {plot_path}")

def plot_region_intensity_lines(intensity_data: dict, output_path="plots/intensity_lines.pdf"):
    """
    Plot carbon intensity line plots for the first 3 regions in the input data (100-hour window).
    """
    import matplotlib.pyplot as plt

    selected_regions = list(intensity_data.keys())[:3]  # take first 3 regions
    color_palette = ["#2c89d9", "#E69F00", "gray"]  # blue, yellow-orange, gray

    fig, ax = plt.subplots(figsize=(6, 2))  # Wide and short

    for i, region in enumerate(selected_regions):
        y = intensity_data[region]
        x = np.arange(len(y))  # 0–99
        ax.plot(x, y, label=region, color=color_palette[i % len(color_palette)], linewidth=1.5)

    # Format axes
    ax.set_ylabel("gCO₂/kWh", fontsize=11)
    ax.set_xlabel("Hour", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(0, 100)

    # Legend (outside right)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Intensity line plot saved to {output_path}")

def main():
    print("Loading Data...")
    intensity_data_full = get_carbon_intensities_for_all_regions()
    curtailment_data = get_next_100_hours_curtailment()
    intensity_data_100 = {
        region: data[:100] for region, data in intensity_data_full.items()
    }

    combined_data = combine_client_data(curtailment_data, intensity_data_full, curtailment_threshold=430)
    plot_combined_heatmap(combined_data, output_name="combined_probing_heatmap")
    plot_region_intensity_lines(intensity_data_100, output_path="plots/intensity_lines.pdf")


if __name__ == "__main__":
    main()
