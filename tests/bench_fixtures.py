"""Synthetic GSR labels: the shape of Labels-GameState.json v1.3, none of its content."""
from footlytics.bench.adapters.gsr import CATEGORY_ID, image_id

#: (track_id, role, team, jersey, x, y): left team defends x < 0
PEOPLE = [
    (1, "goalkeeper", "left", None, -50.0, 0.0),
    (2, "player", "left", "4", -20.0, -10.0),
    (3, "player", "left", "8", -15.0, 12.0),
    (4, "goalkeeper", "right", None, 49.0, 2.0),
    (5, "player", "right", "9", 10.0, -5.0),
    (6, "player", "right", None, 18.0, 20.0),
    (7, "referee", None, None, 0.0, 30.0),
    (8, "ball", None, None, 1.0, 1.0),
    (9, "other", None, None, 5.0, 36.0),
]


def make_labels(seq="SNGS-990", split="valid", n_frames=3, version="1.3"):
    images, annotations = [], []
    for f in range(n_frames):
        iid = image_id(split, seq, f)
        images.append({"is_labeled": True, "image_id": iid, "file_name": f"{f + 1:06d}.jpg",
                       "height": 1080, "width": 1920, "has_labeled_person": True,
                       "has_labeled_pitch": True, "has_labeled_camera": True})
        for tid, role, team, jersey, x, y in PEOPLE:
            x += 0.1 * f
            annotations.append({
                "id": f"{iid}{tid:02d}", "image_id": iid, "track_id": tid,
                "supercategory": "object", "category_id": CATEGORY_ID[role],
                "attributes": {"role": role, "jersey": jersey, "team": team},
                "bbox_image": {"x": 100 + 10 * tid, "y": 500, "w": 20, "h": 60},
                "bbox_pitch": {"x_bottom_left": x - 0.3, "y_bottom_left": y,
                               "x_bottom_middle": x, "y_bottom_middle": y,
                               "x_bottom_right": x + 0.3, "y_bottom_right": y}})
        annotations.append({"id": f"{iid}99", "image_id": iid, "supercategory": "pitch",
                            "category_id": 5, "lines": {}})
    # an unlabelled frame (the evaluator skips it) holding a person not placed on the pitch:
    # real GT has `bbox_pitch: null` only in such frames
    blank = image_id(split, seq, n_frames)
    images.append({"is_labeled": False, "image_id": blank, "file_name": f"{n_frames + 1:06d}.jpg",
                   "height": 1080, "width": 1920, "has_labeled_person": True,
                   "has_labeled_pitch": False, "has_labeled_camera": False})
    annotations.append({"id": "x", "image_id": blank, "track_id": 10,
                        "supercategory": "object", "category_id": 1,
                        "attributes": {"role": "player", "jersey": None, "team": "left"},
                        "bbox_image": {"x": 1, "y": 1, "w": 1, "h": 1}, "bbox_pitch": None})
    info = {"version": version, "name": seq, "id": seq.split("-")[-1], "frame_rate": 25,
            "seq_length": n_frames, "im_dir": "img1", "im_ext": ".jpg"}
    return {"info": info, "images": images, "annotations": annotations, "categories": []}
