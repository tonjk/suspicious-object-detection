import os
import pathlib
import argparse
from dataclasses import dataclass, field
from collections import deque

import cv2
import numpy as np

# Mac M1 safety knobs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ultralytics import YOLO  # noqa: E402
from boxmot import StrongSort  # noqa: E402

PERSON_CLASS = "person"
BAG_CLASSES = {"suitcase", "backpack", "handbag", "bag", "box"}

PERSON_CONF = 0.35
BAG_CONF_NEW = 0.30        # threshold for bag detections near no existing track
BAG_CONF_TRACKED = 0.15   # lower threshold for detections near already-tracked bags
MATCH_MAX_DIST = 110.0
TRACK_MAX_LOST = 75        # ~3.0s at 25fps; helps UNATTENDED bags survive short occlusions
ABANDONED_TRACK_MAX_LOST_MULT = 8  # abandoned bags survive 8× longer before deletion

ATTACH_DIST = 180.0
ABANDON_SECONDS = 3.0
UNOWNED_ABANDON_SECONDS = 8.0  # bag with no detected owner → abandoned after this
STABLE_PIXEL_DRIFT = 30.0
MIN_HISTORY_FOR_STABLE = 10

GHOST_TTL = 10.0        # seconds an abandoned ghost stays in the graveyard
GHOST_MATCH_DIST = 80.0  # pixels — new track within this radius inherits ghost state

SCENE_CUT_THRESHOLD = 42.0


@dataclass
class Track:
    track_id: int
    label: str
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    first_seen: float
    last_seen: float
    lost_frames: int = 0
    history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=40))

    # bag-owner relationship fields
    owner_id: int | None = None
    last_near_owner_time: float | None = None
    is_abandoned: bool = False  # latched True once ABANDONED, reset only when owner returns
    is_ghost_injected: bool = False  # True when this frame's position came from ghost


@dataclass
class AbandonedGhost:
    """Snapshot of a deleted abandoned track — used to re-ID resurrected detections."""
    last_bbox: tuple[int, int, int, int]
    last_centroid: tuple[int, int]
    first_seen: float
    last_near_owner_time: float | None
    owner_id: int | None
    deleted_at: float


def centroid_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def euclid(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def cross_class_nms(
    dets: list[tuple[str, tuple[int, int, int, int], float]],
    iou_thresh: float = 0.30,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Remove cross-class duplicate boxes; keep highest-confidence detection per overlap cluster.

    Threshold lowered to 0.30 (from 0.45) so that partially-overlapping
    detections of the same physical object under different class names
    (e.g. backpack vs handbag) are suppressed more aggressively.
    """
    sorted_dets = sorted(dets, key=lambda x: x[2], reverse=True)
    kept: list[tuple[str, tuple[int, int, int, int], float]] = []
    for label, bbox, conf in sorted_dets:
        if all(iou(bbox, k[1]) < iou_thresh for k in kept):
            kept.append((label, bbox, conf))
    return [(label, bbox) for label, bbox, _ in kept]


# IoU threshold for cross-class matching: if an unmatched detection overlaps
# heavily with an existing track of a *different* class, absorb it instead of
# spawning a duplicate track.
CROSS_CLASS_IOU_MERGE = 0.35
CROSS_CLASS_CENTROID_MERGE = 55.0  # also suppress cross-class duplicates when centroids are very close


def update_tracks(
    tracks: dict[int, Track],
    detections: list[tuple[str, tuple[int, int, int, int]]],
    now: float,
    next_id: int,
) -> tuple[dict[int, Track], int, set[int]]:
    """Greedy centroid matcher for stable IDs across frames.

    Matching priority:
      1. Same-label centroid match  (distance ≤ MATCH_MAX_DIST)
      2. Cross-label IoU match      (IoU ≥ CROSS_CLASS_IOU_MERGE)
      3. Create new track
    """
    current_ids: set[int] = set()

    if not detections:
        to_remove: list[int] = []
        for tid, tr in tracks.items():
            tr.lost_frames += 1
            max_lost = TRACK_MAX_LOST * ABANDONED_TRACK_MAX_LOST_MULT if tr.is_abandoned else TRACK_MAX_LOST
            if tr.lost_frames > max_lost:
                to_remove.append(tid)
        for tid in to_remove:
            tracks.pop(tid, None)
        return tracks, next_id, current_ids

    if not tracks:
        for label, bbox in detections:
            c = centroid_from_bbox(bbox)
            tr = Track(next_id, label, bbox, c, now, now)
            tr.history.append(c)
            tracks[next_id] = tr
            current_ids.add(next_id)
            next_id += 1
        return tracks, next_id, current_ids

    track_items = list(tracks.items())

    # --- Pass 1: same-label centroid matching (original behaviour) ---
    pairs: list[tuple[float, int, int]] = []
    for d_idx, (label, bbox) in enumerate(detections):
        c = centroid_from_bbox(bbox)
        for t_idx, (tid, tr) in enumerate(track_items):
            if tr.label != label:
                continue
            pairs.append((euclid(c, tr.centroid), d_idx, t_idx))

    pairs.sort(key=lambda x: x[0])
    used_det: set[int] = set()
    used_track_idx: set[int] = set()

    for dist, d_idx, t_idx in pairs:
        if dist > MATCH_MAX_DIST:
            break
        if d_idx in used_det or t_idx in used_track_idx:
            continue

        tid, tr = track_items[t_idx]
        label, bbox = detections[d_idx]
        c = centroid_from_bbox(bbox)

        tr.bbox = bbox
        tr.centroid = c
        tr.last_seen = now
        tr.lost_frames = 0
        tr.history.append(c)

        used_det.add(d_idx)
        used_track_idx.add(t_idx)
        current_ids.add(tid)

    # --- Pass 2: cross-class IoU / centroid matching ---
    # If a detection was NOT matched in pass 1 but overlaps heavily with an
    # existing track (regardless of label), or their centroids are very close,
    # absorb it into that track rather than creating a duplicate.
    cross_pairs: list[tuple[float, int, int]] = []   # (score, d_idx, t_idx)  lower = better
    for d_idx, (label, bbox) in enumerate(detections):
        if d_idx in used_det:
            continue
        c = centroid_from_bbox(bbox)
        for t_idx, (tid, tr) in enumerate(track_items):
            if t_idx in used_track_idx:
                continue
            overlap = iou(bbox, tr.bbox)
            cdist = euclid(c, tr.centroid)
            if overlap >= CROSS_CLASS_IOU_MERGE or cdist <= CROSS_CLASS_CENTROID_MERGE:
                score = -overlap if overlap >= CROSS_CLASS_IOU_MERGE else cdist
                cross_pairs.append((score, d_idx, t_idx))

    cross_pairs.sort()
    for score, d_idx, t_idx in cross_pairs:
        if d_idx in used_det or t_idx in used_track_idx:
            continue
        tid, tr = track_items[t_idx]
        label, bbox = detections[d_idx]
        c = centroid_from_bbox(bbox)

        # Keep the existing track's label (first-seen wins) but update position
        tr.bbox = bbox
        tr.centroid = c
        tr.last_seen = now
        tr.lost_frames = 0
        tr.history.append(c)

        used_det.add(d_idx)
        used_track_idx.add(t_idx)
        current_ids.add(tid)

    # Unmatched detections -> new tracks (suppress if overlapping an already-matched track)
    matched_track_bboxes = [
        track_items[t_idx][1].bbox for t_idx in used_track_idx
    ]
    for d_idx, (label, bbox) in enumerate(detections):
        if d_idx in used_det:
            continue
        c = centroid_from_bbox(bbox)
        # Suppress if this detection overlaps heavily or is very close to a
        # track that was already matched — prevents spawning a duplicate.
        if any(
            iou(bbox, mb) >= CROSS_CLASS_IOU_MERGE
            or euclid(c, centroid_from_bbox(mb)) <= CROSS_CLASS_CENTROID_MERGE
            for mb in matched_track_bboxes
        ):
            used_det.add(d_idx)
            continue
        tr = Track(next_id, label, bbox, c, now, now)
        tr.history.append(c)
        tracks[next_id] = tr
        current_ids.add(next_id)
        next_id += 1

    # Unmatched tracks -> lost aging
    to_remove: list[int] = []
    for t_idx, (tid, tr) in enumerate(track_items):
        if t_idx in used_track_idx:
            continue
        tr.lost_frames += 1
        max_lost = TRACK_MAX_LOST * ABANDONED_TRACK_MAX_LOST_MULT if tr.is_abandoned else TRACK_MAX_LOST
        if tr.lost_frames > max_lost:
            to_remove.append(tid)

    for tid in to_remove:
        tracks.pop(tid, None)

    return tracks, next_id, current_ids


def bag_is_stable(bag: Track) -> bool:
    if len(bag.history) < MIN_HISTORY_FOR_STABLE:
        return False
    # Check only the recent window — bag must not have moved recently.
    # Using history[0] would include frames when the owner was still carrying it.
    recent = list(bag.history)[-MIN_HISTORY_FOR_STABLE:]
    cx = sum(p[0] for p in recent) / len(recent)
    cy = sum(p[1] for p in recent) / len(recent)
    return all(euclid(p, (cx, cy)) <= STABLE_PIXEL_DRIFT for p in recent)


def merge_overlapping_bag_tracks(
    tracks: dict[int, Track],
    current_ids: set[int],
    iou_thresh: float = 0.35,
) -> set[int]:
    """Post-processing: if two active bag tracks overlap heavily, merge
    the younger one into the older one to prevent duplicate boxes."""
    ids = sorted(current_ids)  # deterministic order
    to_remove: set[int] = set()
    for i, id_a in enumerate(ids):
        if id_a in to_remove or id_a not in tracks:
            continue
        for id_b in ids[i + 1:]:
            if id_b in to_remove or id_b not in tracks:
                continue
            if iou(tracks[id_a].bbox, tracks[id_b].bbox) >= iou_thresh:
                # Keep the *older* track (lower first_seen); remove the newer one
                keep, drop = (id_a, id_b) if tracks[id_a].first_seen <= tracks[id_b].first_seen else (id_b, id_a)
                # Preserve owner info if the surviving track has none
                if tracks[keep].owner_id is None and tracks[drop].owner_id is not None:
                    tracks[keep].owner_id = tracks[drop].owner_id
                    tracks[keep].last_near_owner_time = tracks[drop].last_near_owner_time
                    tracks[keep].is_abandoned = tracks[drop].is_abandoned
                to_remove.add(drop)
    for tid in to_remove:
        tracks.pop(tid, None)
    return current_ids - to_remove


REID_WEIGHTS = pathlib.Path("osnet_x0_25_msmt17.pt")


def detect_abandoned(video_path: str, output_path: str = "video_output/output.mp4", preview: bool = True, max_frames: int = 0) -> str:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = YOLO("yolo11n.pt")
    model.to(device)

    # StrongSORT tracker for persons — uses appearance re-ID so the same person
    # keeps the same ID even after leaving and re-entering the frame.
    # Runs on CPU even when YOLO runs on MPS (MPS does not support all ops used by re-ID).
    person_tracker = StrongSort(
        reid_weights=REID_WEIGHTS,
        device=torch.device("cpu"),
        half=False,
    )
    # person_id -> Track (populated/updated from StrongSORT output)
    person_tracks: dict[int, Track] = {}
    # Remap StrongSORT's internal IDs -> sequential display IDs (P1, P2, ...)
    person_id_map: dict[int, int] = {}   # ss_id -> display_id
    next_display_pid: int = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    out_ext = os.path.splitext(output_path)[1].lower()
    fourcc_map = {
        ".mp4": "mp4v",
        ".mov": "mp4v",
        ".avi": "XVID",
        ".mkv": "mp4v",
        ".wmv": "WMV2",
        ".flv": "FLV1",
        ".webm": "VP80",
        ".m4v": "mp4v",
        ".ts":  "mp2v",
        ".mts": "mp2v",
    }
    fourcc_str = fourcc_map.get(out_ext, "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out: cv2.VideoWriter | None = None

    bag_tracks: dict[int, Track] = {}
    next_bag_id = 0
    abandoned_ghosts: dict[int, AbandonedGhost] = {}  # graveyard for deleted abandoned tracks

    frame_idx = 0
    prev_gray: np.ndarray | None = None

    if preview:
        cv2.namedWindow("Abandoned Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Abandoned Detection", 1280, 720)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            frame_idx += 1
            now = frame_idx / fps  # use video-time, not wall-clock
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Scene cut handling
            scene_cut = False
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                if float(diff.mean()) > SCENE_CUT_THRESHOLD:
                    scene_cut = True
                    person_tracks.clear()
                    bag_tracks.clear()
                    abandoned_ghosts.clear()
                    person_tracker.reset()
                    person_id_map.clear()
                    next_display_pid = 1
                    print(f"[INFO] Scene cut at frame {frame_idx}")
            prev_gray = gray

            if scene_cut:
                out.write(frame)
                if preview:
                    cv2.imshow("Abandoned Detection", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                continue

            # YOLO detection on each frame
            results = model(frame, verbose=False)[0]

            person_dets_raw: list[tuple[int, int, int, int, float]] = []  # x1,y1,x2,y2,conf
            bag_dets_raw: list[tuple[str, tuple[int, int, int, int], float]] = []

            # Pre-compute tracked bag centroids for adaptive confidence threshold
            tracked_bag_centroids = {bid: tr.centroid for bid, tr in bag_tracks.items()}

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = str(model.names[cls_id])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                if label == PERSON_CLASS and conf >= PERSON_CONF:
                    person_dets_raw.append((x1, y1, x2, y2, conf))
                elif label in BAG_CLASSES:
                    # Lower threshold for bags near existing tracks (prevents false drops)
                    c = centroid_from_bbox(bbox)
                    is_near_tracked = any(
                        euclid(c, tc) <= MATCH_MAX_DIST
                        for tc in tracked_bag_centroids.values()
                    )
                    threshold = BAG_CONF_TRACKED if is_near_tracked else BAG_CONF_NEW
                    if conf >= threshold:
                        bag_dets_raw.append((label, bbox, conf))

            # --- Person tracking via StrongSORT (appearance re-ID) ---
            # Feed detections as Nx6 array: [x1,y1,x2,y2,conf,cls=0]
            if person_dets_raw:
                ss_input = np.array(
                    [[x1, y1, x2, y2, conf, 0] for x1, y1, x2, y2, conf in person_dets_raw],
                    dtype=np.float32,
                )
            else:
                ss_input = np.empty((0, 6), dtype=np.float32)

            ss_out = person_tracker.update(ss_input, frame)
            # ss_out rows: [x1, y1, x2, y2, track_id, conf, cls, det_idx]

            current_person_ids: set[int] = set()
            seen_pids: set[int] = set()

            if len(ss_out) > 0:
                for row in ss_out:
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    ss_id = int(row[4])  # raw StrongSORT internal ID
                    bbox = (x1, y1, x2, y2)
                    c = centroid_from_bbox(bbox)

                    # Map to sequential display ID
                    if ss_id not in person_id_map:
                        person_id_map[ss_id] = next_display_pid
                        next_display_pid += 1
                    pid = person_id_map[ss_id]

                    if pid not in person_tracks:
                        person_tracks[pid] = Track(pid, PERSON_CLASS, bbox, c, now, now)
                    else:
                        tr = person_tracks[pid]
                        tr.bbox = bbox
                        tr.centroid = c
                        tr.last_seen = now
                        tr.lost_frames = 0

                    person_tracks[pid].history.append(c)
                    current_person_ids.add(pid)
                    seen_pids.add(pid)

            # Age out person tracks not seen this frame
            to_remove_p: list[int] = []
            for pid, tr in person_tracks.items():
                if pid not in seen_pids:
                    tr.lost_frames += 1
                    if tr.lost_frames > TRACK_MAX_LOST:
                        to_remove_p.append(pid)
            for pid in to_remove_p:
                person_tracks.pop(pid, None)

            # Ghost injection: re-insert last known bbox for stable bags so
            # update_tracks can match them even when YOLO misses a frame.
            # This now also covers UNATTENDED bags (owner known, not yet abandoned).
            yolo_bag_centroids = [centroid_from_bbox(b) for _, b, *_ in bag_dets_raw]
            for bid, bag in bag_tracks.items():
                if not bag_is_stable(bag):
                    continue
                should_inject = bag.is_abandoned or (bag.owner_id is not None)
                if not should_inject:
                    continue
                # Only inject if YOLO didn't already detect something nearby
                already_detected = any(
                    euclid(bag.centroid, yc) <= MATCH_MAX_DIST for yc in yolo_bag_centroids
                )
                if not already_detected:
                    bag_dets_raw.append((bag.label, bag.bbox, BAG_CONF_TRACKED))

            # Suppress overlapping boxes from different bag classes on the same object
            bag_dets = cross_class_nms(bag_dets_raw)

            # Snapshot existing track IDs before update to detect deletions
            pre_update_ids = set(bag_tracks.keys())
            pre_update_abandoned = {
                bid: bag_tracks[bid]
                for bid in pre_update_ids
                if bag_tracks[bid].is_abandoned
            }

            bag_tracks, next_bag_id, current_bag_ids = update_tracks(
                bag_tracks, bag_dets, now, next_bag_id
            )

            # Graveyard: save deleted abandoned tracks so new tracks can inherit their state
            deleted_ids = pre_update_ids - set(bag_tracks.keys())
            for bid in deleted_ids:
                if bid in pre_update_abandoned:
                    ghost_tr = pre_update_abandoned[bid]
                    abandoned_ghosts[bid] = AbandonedGhost(
                        last_bbox=ghost_tr.bbox,
                        last_centroid=ghost_tr.centroid,
                        first_seen=ghost_tr.first_seen,
                        last_near_owner_time=ghost_tr.last_near_owner_time,
                        owner_id=ghost_tr.owner_id,
                        deleted_at=now,
                    )

            # Expire stale ghosts
            for gid in [gid for gid, g in abandoned_ghosts.items() if now - g.deleted_at > GHOST_TTL]:
                del abandoned_ghosts[gid]

            # Re-ID: if a newly created track is close to a ghost, inherit abandoned state
            for bid in current_bag_ids:
                if bid not in bag_tracks:
                    continue
                bag = bag_tracks[bid]
                if bag.first_seen != now:  # not a newly created track
                    continue
                for gid, ghost in list(abandoned_ghosts.items()):
                    if euclid(bag.centroid, ghost.last_centroid) <= GHOST_MATCH_DIST:
                        bag.is_abandoned = True
                        bag.first_seen = ghost.first_seen
                        bag.last_near_owner_time = ghost.last_near_owner_time
                        bag.owner_id = ghost.owner_id
                        del abandoned_ghosts[gid]
                        print(f"[INFO] Re-ID: new track B{bid} inherited abandoned state from ghost {gid}")
                        break

            # Post-merge: collapse any remaining duplicate bag tracks
            current_bag_ids = merge_overlapping_bag_tracks(
                bag_tracks, current_bag_ids
            )

            visible_people = {pid for pid in current_person_ids if pid in person_tracks}

            abandoned_count = 0

            # Evaluate bag-owner relationship
            for bid in current_bag_ids:
                if bid not in bag_tracks:
                    continue
                bag = bag_tracks[bid]
                bx1, by1, bx2, by2 = bag.bbox

                nearest_pid: int | None = None
                nearest_dist = float("inf")

                for pid in visible_people:
                    person = person_tracks[pid]
                    d = euclid(bag.centroid, person.centroid)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_pid = pid

                # True only when the ACTUAL owner (or first-seen person) is nearby.
                # An already-abandoned bag must not be claimable by a passing stranger.
                owner_is_near = (
                    nearest_pid is not None
                    and nearest_dist <= ATTACH_DIST
                    and (
                        # Known owner returning
                        (bag.owner_id is not None and nearest_pid == bag.owner_id)
                        # Unclaimed bag not yet abandoned — first nearby person becomes owner
                        or (bag.owner_id is None and not bag.is_abandoned)
                    )
                )

                if owner_is_near:
                    if bag.owner_id is None:
                        bag.owner_id = nearest_pid
                    bag.last_near_owner_time = now
                    bag.is_abandoned = False  # owner returned — reset latch
                    status = "WITH OWNER"
                    color = (0, 255, 255)
                else:
                    owner_visible = bag.owner_id in visible_people if bag.owner_id is not None else False
                    last_near = bag.last_near_owner_time if bag.last_near_owner_time is not None else bag.first_seen
                    away_for = now - last_near

                    # Latch ABANDONED: once triggered, stay ABANDONED even if a passerby
                    # occludes the bag and bag_is_stable() temporarily returns False.
                    if bag.is_abandoned:
                        status = f"ABANDONED {away_for:.1f}s"
                        color = (0, 0, 255)
                        abandoned_count += 1
                    elif bag.owner_id is not None and not owner_visible and bag_is_stable(bag) and away_for >= ABANDON_SECONDS:
                        bag.is_abandoned = True
                        status = f"ABANDONED {away_for:.1f}s"
                        color = (0, 0, 255)
                        abandoned_count += 1
                    elif bag.owner_id is not None and owner_visible:
                        # Owner is still in scene but just stepped away briefly
                        status = f"UNATTENDED {away_for:.1f}s"
                        color = (0, 165, 255)
                    elif bag.owner_id is None:
                        away_for_unowned = now - bag.first_seen
                        if bag_is_stable(bag) and away_for_unowned >= UNOWNED_ABANDON_SECONDS:
                            bag.is_abandoned = True
                            status = f"ABANDONED {away_for_unowned:.1f}s"
                            color = (0, 0, 255)
                            abandoned_count += 1
                        else:
                            status = f"UNOWNED {away_for_unowned:.1f}s"
                            color = (128, 128, 128)
                    else:
                        status = f"UNATTENDED {away_for:.1f}s"
                        color = (0, 165, 255)

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(
                    frame,
                    f"B{bid} {bag.label} {status}",
                    (bx1, max(by1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            hud = f"Frame: {frame_idx}  Bags: {len(current_bag_ids)}  Abandoned: {abandoned_count}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)

            if preview:
                cv2.imshow("Abandoned Detection", frame)
                delay = max(1, int(1000 / fps))
                if (cv2.waitKey(delay) & 0xFF) == ord("q"):
                    print("[INFO] User quit.")
                    break

            if max_frames > 0 and frame_idx >= max_frames:
                print(f"[INFO] Stop at max_frames={max_frames}")
                break

    finally:
        cap.release()
        if out is not None:
            out.release()
        if preview:
            cv2.destroyAllWindows()

    print(f"[INFO] Done. Saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abandoned object detection (Mac M1)")
    parser.add_argument("--input", default="video/Abandoned Object Detection.mp4", help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path (default: <input_name>_output.mp4)")
    parser.add_argument("--no-preview", action="store_true", help="Disable realtime preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit frames (0=all)")
    args = parser.parse_args()

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output if args.output else f"video_output/{input_stem}_output.mp4"

    supported_exts = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts"}
    input_ext = os.path.splitext(args.input)[1].lower()
    if input_ext not in supported_exts:
        print(f"[WARNING] Unsupported input format '{input_ext}'. Supported: {', '.join(sorted(supported_exts))}")
        raise SystemExit(1)

    detect_abandoned(
        video_path=args.input,
        output_path=output_path,
        preview=not args.no_preview,
        max_frames=args.max_frames,
    )
