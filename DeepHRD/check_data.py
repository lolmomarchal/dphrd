import torch
import os

# !!! IMPORTANT: Change this to your actual output/results path !!!
path_to_results = "/home/lolmomarchal/data/results"

data_file = os.path.join(path_to_results, "testData.pt")

if not os.path.exists(data_file):
    print(f"Error: Data file not found at {data_file}")
else:
    print(f"Loading data from: {data_file}")
    try:
        data = torch.load(data_file)

        num_slides = len(data['slides'])
        print(f"\nNumber of slides in the dataset: {num_slides}")

        if num_slides > 0:
            total_tiles = 0
            for i, slide in enumerate(data['slides']):
                num_tiles = len(data['tiles'][i])
                total_tiles += num_tiles
                print(f"  - Slide {i+1} ({os.path.basename(slide)}): {num_tiles} tiles")

            print(f"\nTotal number of tiles for the model: {total_tiles}")

            if total_tiles == 0:
                print("\n[CONCLUSION] The dataset is empty. No tiles were passed to the model, which caused the crash.")
            else:
                print("\n[CONCLUSION] The dataset seems valid. The issue might be with the model itself.")
        else:
            print("\n[CONCLUSION] The dataset contains no slides.")

    except Exception as e:
        print(f"An error occurred while loading or inspecting the file: {e}")