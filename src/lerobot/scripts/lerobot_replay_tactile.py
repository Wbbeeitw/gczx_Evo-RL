#!/usr/bin/env python3
"""Replay a collected LeRobot dataset with tactile views."""

import argparse, time, cv2, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    p = argparse.ArgumentParser(description="Replay collected episodes")
    p.add_argument("--dataset", required=True, help="Dataset repo_id")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    p.add_argument("--fps", type=int, default=10, help="Replay speed")
    p.add_argument("--show-action", action="store_true", help="Print action values")
    p.add_argument("--show-left", action="store_true", help="Show left arm actions")
    p.add_argument("--show-right", action="store_true", help="Show right arm actions")
    return p.parse_args()


def main():
    args = parse_args()
    ds = LeRobotDataset(args.dataset)
    meta = ds.meta

    # Find episode boundaries
    ep_starts = []
    ep_ends = []
    current_ep = None
    for idx in range(len(ds)):
        ep_idx = ds[idx].get("episode_index", ds[idx].get("observation.episode_index",
                  ds.hf_dataset[idx].get("episode_index", 0)))
        ep_idx = int(ep_idx) if ep_idx is not None else 0
        if ep_idx != current_ep:
            if current_ep is not None:
                ep_ends.append(idx - 1)
            ep_starts.append(idx)
            current_ep = ep_idx
    ep_ends.append(len(ds) - 1)

    if args.episode >= len(ep_starts):
        print(f"Only {len(ep_starts)} episodes available (0-{len(ep_starts)-1})")
        return

    start, end = ep_starts[args.episode], ep_ends[args.episode]
    print(f"Episode {args.episode}: frames {start}-{end} ({end-start+1} frames)")
    print(f"Action feature: {meta.features['action']}")
    print(f"Image keys: {[k for k in meta.features if 'image' in k]}")

    # Get action names for display
    act_names = meta.features.get("action", {}).get("names", [f"dim_{i}" for i in range(14)])

    for idx in range(start, end + 1):
        frame = ds[idx]
        imgs = {}
        for k, v in frame.items():
            if "image" in k and hasattr(v, "shape") and len(v.shape) == 3:
                # Resize to display
                h, w = v.shape[:2]
                scale = min(400.0 / max(h, w), 1.0)
                nh, nw = int(h * scale), int(w * scale)
                imgs[k] = cv2.resize(v, (nw, nh))

        # Build display grid: camera row + tactile row
        cam_imgs = [v for k, v in imgs.items() if "tactile" not in k]
        tac_imgs = [v for k, v in imgs.items() if "tactile" in k]

        rows = []
        if cam_imgs:
            # Pad to same height
            max_h = max(i.shape[0] for i in cam_imgs)
            padded = []
            for i in cam_imgs:
                if i.shape[0] < max_h:
                    p = np.zeros((max_h, i.shape[1], 3), dtype=np.uint8)
                    p[:i.shape[0], :i.shape[1]] = i
                    padded.append(p)
                else:
                    padded.append(i)
            rows.append(np.hstack(padded))
        if tac_imgs:
            max_h = max(i.shape[0] for i in tac_imgs)
            padded = []
            for i in tac_imgs:
                if i.shape[0] < max_h:
                    p = np.zeros((max_h, i.shape[1], 3), dtype=np.uint8)
                    p[:i.shape[0], :i.shape[1]] = i
                    padded.append(p)
                else:
                    padded.append(i)
            rows.append(np.hstack(padded))

        if rows:
            display = np.vstack(rows)
        else:
            display = np.zeros((200, 400, 3), dtype=np.uint8)

        # Overlay frame info
        cv2.putText(display, f"Frame {idx}/{end}  Ep {args.episode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        action = frame.get("action")
        if action is not None and args.show_action:
            # Print first 14 meaningful dims
            a = np.array(action[:14])
            print(f"\n--- Frame {idx} ---")
            print("  Left arm  (dim 0-6):")
            for n in range(7):
                print(f"    {act_names[n]:25s}: {a[n]:10.4f}")
            print("  Right arm (dim 7-13):")
            for n in range(7, 14):
                print(f"    {act_names[n]:25s}: {a[n]:10.4f}")

        # Also show as small text on display
        if action is not None:
            a = np.array(action[:14])
            y = 55
            cv2.putText(display, f"L arm:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            y += 15
            for n in range(7):
                cv2.putText(display, f"  {act_names[n]:20s} {a[n]:7.2f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
                y += 12
            y += 5
            cv2.putText(display, f"R arm:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            y += 15
            for n in range(7, 14):
                cv2.putText(display, f"  {act_names[n]:20s} {a[n]:7.2f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
                y += 12

        cv2.imshow("Dataset Replay", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(int(1000 / args.fps)) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print("Paused — press any key to continue")
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
