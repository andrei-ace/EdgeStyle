import csv
import pandas as pd
import os
import youtube_dl


def download_videos(csv_file_path, download_directory):
    # Read the CSV file
    df = pd.read_csv(csv_file_path, keep_default_na=False)

    for index, row in df.iterrows():
        # Skip if file already downloaded
        if row["File"] is not None and os.path.isfile(row["File"]):
            print(f"Skipping {row['Name']}, already downloaded.")
            continue

        video_url = row["URL"]
        video_name = row["Name"]

        # Define youtube-dl options
        ydl_opts = {
            "outtmpl": os.path.join(download_directory, "%(title)s-%(id)s.%(ext)s"),
            "format": "bestvideo[ext=mp4]/best[ext=mp4]/best",
        }

        # Download the video
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_filename = ydl.prepare_filename(info_dict)
            # remove spaces from video filename
            os.rename(video_filename, video_filename.replace(" ", "_"))
            video_filename = video_filename.replace(" ", "_")
            # Update the DataFrame with the filename
            df.at[index, "File"] = video_filename

    # Save the updated DataFrame to CSV with specified formatting
    df.to_csv(csv_file_path, sep=",", quotechar='"', quoting=csv.QUOTE_ALL, index=False)
    print("Download complete and CSV updated.")


# Example usage
csv_file_path = "data/video/videos.csv"
download_directory = "data/video"
download_videos(csv_file_path, download_directory)
