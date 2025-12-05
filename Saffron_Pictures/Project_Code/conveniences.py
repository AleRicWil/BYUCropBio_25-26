import numpy as np
from pathlib import Path
import pandas as pd
import os

CSVs_PATH = r"D:\CSVs"

def view_matrices():
    from pathlib import Path

    def load_and_display_homography(npy_path):
        """
        Loads a homography matrix from a .npy file, prints it to the console,
        and saves it as a human-readable .txt file in the same directory.
        """
        if not Path(npy_path).exists():
            print(f"File not found: {npy_path}")
            return None
        
        H = np.load(npy_path)
        print(f"\nHomography matrix from {npy_path}:")
        print(H)
        
        # Save to .txt file
        txt_path = Path(npy_path).with_suffix('.txt')
        np.savetxt(txt_path, H, fmt='%.8e', header=f"Homography matrix from {npy_path}")
        print(f"Saved to: {txt_path}")
        
        return H

    # List of homography files (adjust paths if needed)
    views = ['left', 'middle', 'right']
    homography_paths = [f'{view}_homography.npy' for view in views]

    # Load and process each
    for path in homography_paths:
        load_and_display_homography(path)

def save_masked_CSV():
    from pathlib import Path

    # Specify the full path to the input CSV file here
    # This can be altered directly in the IDE before running
    input_csv = CSVs_PATH + r'\fullgps.csv'

    # Extract input path and directory
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = input_path.parent
    output_path = output_dir / 'flowers_only.csv'

    # Load the large CSV file efficiently
    # Use low_memory=False to handle mixed types and large size without warnings
    # For very large files, chunking is enabled below to reduce memory usage
    chunksize = 10**6  # Process 1 million rows at a time; adjust based on available RAM
    filtered_chunks = []

    # Read and filter in chunks for efficiency on large datasets (230 MB)
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        # Apply mask: filter rows where 'flower?' == 1
        # Vectorized operation, efficient even on large chunks
        filtered_chunk = chunk[chunk['flower?'] == 1]
        filtered_chunks.append(filtered_chunk)

    # Concatenate all filtered chunks into a single DataFrame
    filtered_df = pd.concat(filtered_chunks, ignore_index=True)

    # Save the filtered DataFrame to the new CSV in the same folder
    # index=False avoids adding an extra index column
    filtered_df.to_csv(output_path, index=False)

    print(f"Filtered {len(filtered_df)} rows with flowers to {output_path}")

def save_CSV_by_day():
    from pathlib import Path
    import os

    # Specify the full path to the input CSV file here (flowers_only.csv from previous script)
    # This can be altered directly in the IDE before running
    # Example: "D:\\flowers_only.csv"
    input_csv = CSVs_PATH + r'\flowers_only.csv'

    # Extract input path and directory
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = input_path.parent

    # Function to extract date from timestamp
    # Assuming timestamp format: YYYYMMDD_HHMMSS_mmmZ (e.g., 20251028_042339_376551Z)
    # Extracts YYYYMMDD as string for grouping
    def extract_date(timestamp):
        if pd.isna(timestamp):
            return None  # Skip if NaN
        try:
            return str(timestamp)[:8]  # First 8 characters: YYYYMMDD
        except:
            return None  # Handle any parsing errors gracefully

    # Efficiently process large CSV in chunks to minimize memory usage
    # This is crucial for large datasets in robotic mapping applications where
    # sensor data (e.g., from color cameras) can generate massive logs
    chunksize = 10**6  # Process 1 million rows at a time; adjust based on available RAM

    # Dictionary to track which day files have been created (to handle headers on append)
    # Key: date_str, Value: output_path (we'll check existence instead for simplicity)
    created_files = set()

    # Read and process in chunks
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        # Extract date column efficiently
        chunk['date'] = chunk['timestamp'].apply(extract_date)
        
        # Drop rows with invalid dates (None)
        chunk = chunk[chunk['date'].notna()]
        
        # Group by date within the chunk
        grouped = chunk.groupby('date')
        
        # For each group (day), save or append to respective CSV
        for date_str, group in grouped:
            # Define output path: e.g., flowers_YYYYMMDD.csv in same directory
            output_path = output_dir / f'flowers_{date_str}.csv'
            
            # Drop the temporary 'date' column before saving
            group = group.drop(columns=['date'])
            
            # Check if file already exists
            file_exists = output_path.exists()
            
            # Append mode if exists (no header), else write with header
            group.to_csv(output_path, mode='a' if file_exists else 'w', 
                        header=not file_exists, index=False)
            
            # Track created files (though we use existence check)
            if not file_exists:
                created_files.add(date_str)

    # Print summary of created files
    print(f"Split data into {len(created_files)} daily CSVs in {output_dir}:")
    for date_str in sorted(created_files):
        print(f" - flowers_{date_str}.csv")

def save_CSV_by_session():
    """
    Splits each daily flowers_YYYYMMDD.csv file into separate CSVs based on the 'sourceFolder' column,
    which identifies the time of day or session (e.g., '10-25_10pm'). This function scans the directory
    for all daily CSVs, processes each one efficiently (using chunking for large files), groups by
    'sourceFolder', extracts the part after the day (after the underscore) as suffix, and saves outputs 
    as flowers_YYYYMMDD_suffix.csv (e.g., flowers_20251026_10pm.csv).

    This is designed for efficient execution on large datasets from robotic localization and mapping
    tasks using color camera sensors, where sensor logs can be voluminous. Chunking minimizes memory
    usage, and vectorized operations ensure speed.
    """
    # Specify the directory containing the daily CSVs (e.g., from save_CSV_by_day)
    # This can be altered directly in the IDE before running
    csv_dir = CSVs_PATH

    # Convert to Path for easy handling
    dir_path = Path(csv_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Find all daily CSV files matching 'flowers_YYYYMMDD.csv'
    daily_files = list(dir_path.glob('flowers_*.csv'))
    daily_files = [f for f in daily_files if len(f.stem) == 16 and f.stem[8:].isdigit()]  # Validate YYYYMMDD

    if not daily_files:
        print("No daily CSV files found in the directory.")
        return

    # Process each daily file
    for daily_path in daily_files:
        date_str = daily_path.stem[8:]  # Extract YYYYMMDD
        print(f"Processing daily file: {daily_path}")

        # Efficiently process in chunks to handle large daily files
        chunksize = 10**6  # Adjust based on RAM; suitable for large sensor datasets
        created_sessions = set()  # Track sessions for this day to handle appends

        # Read and process in chunks
        for chunk in pd.read_csv(daily_path, chunksize=chunksize, low_memory=False):
            # Drop rows with NaN in 'sourceFolder'
            chunk = chunk[chunk['sourceFolder'].notna()]

            # Group by 'sourceFolder' within the chunk
            grouped = chunk.groupby('sourceFolder')

            # For each group (session), save or append to respective CSV
            for session_str, group in grouped:
                # Extract the suffix after the day (part after underscore)
                parts = session_str.split('_')
                if len(parts) > 1:
                    suffix = '_'.join(parts[1:])  # Join in case there are multiple underscores
                else:
                    suffix = session_str  # Fallback if no underscore

                # Sanitize suffix for filename (replace invalid chars)
                safe_suffix = suffix.replace('/', '_').replace('\\', '_').replace(':', '_')

                # Define output path: e.g., flowers_YYYYMMDD_10pm.csv
                output_path = dir_path / f'flowers_{date_str}_{safe_suffix}.csv'

                # Check if file already exists
                file_exists = output_path.exists()

                # Append mode if exists (no header), else write with header
                group.to_csv(output_path, mode='a' if file_exists else 'w',
                             header=not file_exists, index=False)

                # Track created sessions (using original session_str for summary)
                if not file_exists:
                    created_sessions.add(session_str)

        # Print summary for this day
        print(f"Split {daily_path} into {len(created_sessions)} session CSVs:")
        for session_str in sorted(created_sessions):
            parts = session_str.split('_')
            if len(parts) > 1:
                suffix = '_'.join(parts[1:])
            else:
                suffix = session_str
            safe_suffix = suffix.replace('/', '_').replace('\\', '_').replace(':', '_')
            print(f" - flowers_{date_str}_{safe_suffix}.csv")

if __name__ == "__main__":
    # view_matrices()
    save_masked_CSV()
    save_CSV_by_day()
    save_CSV_by_session()