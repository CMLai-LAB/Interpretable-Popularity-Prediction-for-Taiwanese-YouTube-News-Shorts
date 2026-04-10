import json
from datetime import datetime
from pathlib import Path

records = []

ocr_files = [
    "data/newsebc_thumbnail_ocr.json",
    "data/setnews_thumbnail_ocr.json",
    "data/TTV_NEWS_thumbnail_ocr.json",
    "data/TVBSNEWS01_thumbnail_ocr.json",
    "data/中天新聞CtiNews_thumbnail_ocr.json"
]

metadata_files = [
    "data/newsebc_shorts_full_metadata.json",
    "data/setnews_shorts_full_metadata.json",
    "data/TTV_NEWS_shorts_full_metadata.json",
    "data/TVBSNEWS01_shorts_full_metadata.json",
    "data/中天新聞CtiNews_shorts_full_metadata.json"
]


def metadata_path_to_channel_slug(path: str) -> str:
    stem = Path(path).stem
    suffix = "_shorts_full_metadata"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem

def url_to_topic_name(url: str) -> str:
    if not url:
        return ""
    name = url.rstrip("/").split("/")[-1]
    name = name.replace("_", " ")
    return name

def load_data():
    records = []
    for ocr_file, meta_file in zip(ocr_files, metadata_files):
        channel_slug = metadata_path_to_channel_slug(meta_file)
        with open(ocr_file, "r", encoding="utf-8") as f:
            ocr_json = json.load(f)

        ocr_lookup = {
            item["image"].replace(".jpg", ""): item.get("joined_text", "")
            for item in ocr_json
        }

        with open(meta_file, "r", encoding="utf-8") as f:
            video_data = json.load(f)

        for video in video_data:

            vid = video["video_id"]
            view_count = int(video.get("view_count", 0))
            comment_count = int(video.get("comment_count", 0))
            pub = video.get("published_at")
            topicCategories = video.get("topicCategories", [])
            if view_count == 0 or pub is None:
                continue

            date = datetime.fromisoformat(pub.replace("Z",""))
            records.append({
                "video_id": vid,
                "date": date,
                "duration_seconds": video.get("duration_seconds", 0),
                "channel_id": video.get("channel_id", ""),
                "channel": video.get("channel_title", meta_file),
                "channel_slug": channel_slug,
                "thumbnail_path": str(Path("data/thumbnails") / channel_slug / f"{vid}.jpg"),
                "title_text": video["title"],
                "title_len": len(video["title"]),
                "exclaim": video["title"].count("!") + video["title"].count("！"),
                "question": video["title"].count("?") + video["title"].count("？"),
                "digit_ratio": sum(c.isdigit() for c in video["title"]) / max(len(video["title"]), 1),
                "ocr_text": ocr_lookup.get(vid, ""),
                "ocr_len": len(ocr_lookup.get(vid, "")),
                "ocr_exclaim": ocr_lookup.get(vid, "").count("!") + ocr_lookup.get(vid, "").count("！"),
                "ocr_question": ocr_lookup.get(vid, "").count("?") + ocr_lookup.get(vid, "").count("？"),
                "ocr_digit_ratio": sum(c.isdigit() for c in ocr_lookup.get(vid, "")) / max(len(ocr_lookup.get(vid, "")), 1),
                "framing_text": video["title"] + " " + ocr_lookup.get(vid, ""),
                "comment_rate": comment_count / view_count,
                "view_count": view_count,
                "comment_count": comment_count,
                "topicCategories": [url_to_topic_name(url) for url in topicCategories]
            })

    print("Total samples:", len(records))
    return records
