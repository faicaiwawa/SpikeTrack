import json
import os
import numpy as np
from collections import defaultdict


def load_all_spike_rates_and_calculate_overall_average(save_dir="spike_rate_data", output_name="overall_average"):
    """
    Load all average spike rate files from the folder and calculate overall average

    Args:
        save_dir: Directory storing average spike rate files
        output_name: Name of the output file

    Returns:
        overall_average_rates: Overall average spike rate dictionary
        file_count: Number of files processed
    """

    # Check if directory exists
    if not os.path.exists(save_dir):
        print(f"Directory does not exist: {save_dir}")
        return None, 0

    # Get all JSON files
    json_files = [f for f in os.listdir(save_dir) if f.endswith('_spike_rates.json')]

    if not json_files:
        print(f"No spike rate files found in directory {save_dir}")
        return None, 0

    print(f"Found {len(json_files)} spike rate files")

    # Store spike rate data for all layers
    all_layers_data = defaultdict(list)
    successful_files = []

    # Load files one by one
    for json_file in json_files:
        file_path = os.path.join(save_dir, json_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            average_rates = data['average_rates']
            video_name = data.get('video_name', json_file.replace('_spike_rates.json', ''))

            print(f"Successfully loaded: {video_name}")

            # Add data from each layer to the collection
            for layer_name, rates in average_rates.items():
                all_layers_data[layer_name].append(rates)

            successful_files.append(video_name)

        except Exception as e:
            print(f"Error loading file {json_file}: {e}")
            continue

    if not successful_files:
        print("No files loaded successfully")
        return None, 0

    print(f"Successfully loaded {len(successful_files)} files: {successful_files}")

    # Calculate overall average for each layer
    overall_average_rates = {}

    for layer_name, rates_list in all_layers_data.items():
        # Convert to numpy array: shape (num_videos, time_steps)
        rates_array = np.array(rates_list)

        # Calculate average across videos
        overall_avg = np.mean(rates_array, axis=0)
        overall_average_rates[layer_name] = overall_avg.tolist()

        print(f"Layer {layer_name}: {len(rates_list)} videos, time steps: {len(overall_avg)}")

    # Save overall average results
    save_overall_average(overall_average_rates, successful_files, output_name, save_dir)

    for item in overall_average_rates:
        print(f"{item}: {overall_average_rates[item]}")


    return overall_average_rates, len(successful_files)


def save_overall_average(overall_average_rates, video_list, output_name, save_dir):
    """
    Save overall average spike rates to JSON file

    Args:
        overall_average_rates: Overall average spike rates
        video_list: List of videos used in calculation
        output_name: Output file name
        save_dir: Save directory
    """
    import time

    output_data = {
        'overall_average_rates': overall_average_rates,
        'source_videos': video_list,
        'video_count': len(video_list),
        'timestamp': time.time(),
        'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': f'Overall average spike rates calculated from {len(video_list)} videos'
    }

    output_path = os.path.join(save_dir, f"{output_name}_overall_spike_rates.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Overall average spike rates saved to: {output_path}")


def load_overall_average_spike_rates(output_name="overall_average", save_dir="spike_rate_data"):
    """
    Load overall average spike rate file

    Args:
        output_name: File name
        save_dir: Save directory

    Returns:
        overall_average_rates: Overall average spike rates
        metadata: Metadata information
    """
    file_path = os.path.join(save_dir, f"{output_name}_overall_spike_rates.json")

    if not os.path.exists(file_path):
        print(f"Overall average file does not exist: {file_path}")
        return None, None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        overall_average_rates = data['overall_average_rates']
        metadata = {
            'source_videos': data.get('source_videos', []),
            'video_count': data.get('video_count', 0),
            'timestamp': data.get('timestamp'),
            'save_time': data.get('save_time'),
            'description': data.get('description')
        }

        print(f"Successfully loaded overall average spike rates: {file_path}")
        print(f"Calculated from {metadata['video_count']} videos")

        return overall_average_rates, metadata

    except Exception as e:
        print(f"Error loading overall average file: {e}")
        return None, None


# Main execution script
if __name__ == "__main__":

    '''calculate overall average spike rates in a folder and save it'''


    # Set parameters
    data_directory = "./tracking/spiketrack_b256_t1/search/"  # Modify to your data directory
    output_filename = "overall_average"  # Output file name

    print("Starting to process average spike rate files...")

    # Calculate overall average
    overall_rates, file_count = load_all_spike_rates_and_calculate_overall_average(
        save_dir=data_directory,
        output_name=output_filename
    )

    if overall_rates:
        print(f"\nOverall average calculation completed!")
        print(f"Processed {file_count} video files")
        print(f"Total of {len(overall_rates)} network layers")

        # Display statistics for each layer
        for layer_name, rates in overall_rates.items():
            print(f"  {layer_name}: {len(rates)} time steps, average spike rate range: {min(rates):.4f} - {max(rates):.4f}")

    # Test loading functionality
    print(f"\nTesting loading of overall average file...")
    loaded_rates, metadata = load_overall_average_spike_rates(output_filename, data_directory)

    if loaded_rates:
        print(f"Loading successful! Based on videos: {metadata['source_videos']}")
