from get_files import FILES
import audalign as ad
import os
import ffmpeg
import shutil

if __name__ == "__main__":
    for x, y in FILES:
        for dir_path in ["/tmp/a", "/tmp/b"]:
            if not os.path.exists(dir_path):
                # Create the directory
                os.makedirs(dir_path)
                print(f"Directory '{dir_path}' was created.")
        else:
            print(f"Directory '{dir_path}' already exists.")
        correlation_rec0 = ad.CorrelationRecognizer()
        correlation_rec1 = ad.CorrelationRecognizer()
        results = ad.align_files(
            x,
            y,
            destination_path="/tmp/a/",
            recognizer=correlation_rec0,
        )
        fine_results = ad.fine_align(
            results, destination_path="/tmp/b/", recognizer=correlation_rec1
        )
        for fi in [x, y]:
            ffmpeg.input(f"/tmp/b/{os.path.basename(fi)}").output(
                fi, format="wav", acodec="pcm_u16le"
            ).run(overwrite_output=True)
        for dir_path in ["/tmp/a", "/tmp/b"]:
            shutil.rmtree(dir_path)
