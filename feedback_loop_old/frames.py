import cv2
import os


def get_frames(video_path, n=10):
    """grab n frames spread out through the video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("cant open", video_path)
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        pos = int(total * (i + 1) / (n + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def save_frames(frames, outdir="frames_out"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    paths = []
    for i, f in enumerate(frames):
        p = os.path.join(outdir, "frame_%d.jpg" % i)
        cv2.imwrite(p, f)
        paths.append(p)
    return paths


if __name__ == "__main__":
    fs = get_frames("test.mp4")
    print("got", len(fs), "frames")
    print(save_frames(fs))
