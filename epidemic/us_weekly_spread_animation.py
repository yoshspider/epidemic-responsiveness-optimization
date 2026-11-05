"""Build a small looping GIF from the 12 weekly comparison plates."""

import argparse
from pathlib import Path

from PIL import Image

from .us_weekly_proxy import OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=OUT / "spread")
    parser.add_argument("--output", type=Path, default=OUT / "weekly_spread.gif")
    args = parser.parse_args()
    frames = []
    for week in range(1, 13):
        with Image.open(args.frames / f"week_{week:02d}.png") as source:
            frame = source.convert("RGB")
            frame.thumbnail((1100, 850), Image.Resampling.LANCZOS)
            frames.append(frame)
    # Derive one palette from all weeks so colors remain stable in playback.
    sample = Image.new("RGB", (300, 400))
    for idx, frame in enumerate(frames):
        thumb = frame.copy()
        thumb.thumbnail((100, 100))
        sample.paste(thumb, (100 * (idx % 3), 100 * (idx // 3)))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=[1100] + [750] * 10 + [2200], loop=0, optimize=False)
    with Image.open(args.output) as gif:
        assert gif.n_frames == 12 and gif.info["loop"] == 0
    print(args.output)


if __name__ == "__main__":
    main()
