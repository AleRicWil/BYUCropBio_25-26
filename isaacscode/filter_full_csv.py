import pandas as pd
import numpy as np


def filter_csv(input_file='your_file.csv', output_file='filtered_output.csv', with_timestamp=True, convert_to_meters=False):
    # Read the CSV file
    import ast

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Filter rows where flower? column equals 1
    filtered_df = df[df['flower?'] == 1]

    # Select only the columns you want
    result_df = filtered_df[['timestamp', 'GPSpoint', 'horizontal_accuracy']].copy()

    # Expand rows with multiple GPS points
    expanded_rows = []

    for _, row in result_df.iterrows():
        timestamp = row['timestamp']
        horizontal_accuracy = row['horizontal_accuracy']
        gps_points = row['GPSpoint']
        
        # Parse the GPS points list
        try:
            gps_list = ast.literal_eval(gps_points)
            # If it's a list of GPS points, create a row for each
            for gps_point in gps_list:
                # Check if GPS point contains 'nan' (case insensitive)
                gps_str = str(gps_point).lower()
                if 'nan' not in gps_str:
                    expanded_rows.append({
                        'timestamp': timestamp,
                        'GPSpoint': gps_point,
                        'horizontal_accuracy': horizontal_accuracy
                    })
        except:
            # If parsing fails, check for nan and keep if valid
            gps_str = str(gps_points).lower()
            if 'nan' not in gps_str:
                expanded_rows.append({
                    'timestamp': timestamp,
                    'GPSpoint': gps_points,
                    'horizontal_accuracy': horizontal_accuracy
                })

    # Create new dataframe from expanded rows
    final_df = pd.DataFrame(expanded_rows)

    # Convert GPS coordinates to meters if requested
    if convert_to_meters:
        # Parse GPS coordinates
        coords = []
        for gps in final_df['GPSpoint']:
            try:
                # Extract numbers from string like '[  39.62705187 -111.63685472]'
                gps_str = str(gps).replace('[', '').replace(']', '').strip()
                lat, lon = map(float, gps_str.split())
                coords.append([lat, lon])
            except:
                coords.append([np.nan, np.nan])
        
        coords = np.array(coords)
        
        # Find min lat and lon (ignoring NaN values)
        min_lat = np.nanmin(coords[:, 0])
        min_lon = np.nanmin(coords[:, 1])
        
        # Convert to meters
        # Approximate: 1 degree latitude ≈ 111,000 meters
        # 1 degree longitude ≈ 111,000 * cos(latitude) meters
        avg_lat = np.nanmean(coords[:, 0])
        lat_to_m = 111000
        lon_to_m = 111000 * np.cos(np.radians(avg_lat))
        
        x_meters = (coords[:, 1] - min_lon) * lon_to_m + 1
        y_meters = (coords[:, 0] - min_lat) * lat_to_m + 1
        
        # Format as numpy array strings
        final_df['GPSparsed'] = [f"[{x:.8f} {y:.8f}]" for x, y in zip(x_meters, y_meters)]
        final_df = final_df.drop(columns=['GPSpoint'])
        
        print(f"\nConversion reference:")
        print(f"Min lat: {min_lat}, Min lon: {min_lon}")
        print(f"Origin (1,1) corresponds to GPS: ({min_lat}, {min_lon})")

    if not with_timestamp:
        final_df = final_df.drop(columns=['timestamp'])

    # Save to new CSV file
    final_df.to_csv(output_file, index=False)

    print(f"Filtered and expanded to {len(final_df)} rows")
    print("\nFirst few rows:")
    print(final_df.head())


if __name__ == "__main__":
    filter_csv('fullgpscsv.csv', 'filtered_flower_data.csv', with_timestamp=True, convert_to_meters=True)