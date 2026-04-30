"""
author: Jayesh Pandey
summary: Simple API test client to verify /score and /compare endpoints.
"""

import argparse
import json
import urllib.request


def _post_json(url: str, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple API test client.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", type=str)
    parser.add_argument("--prompt", default="a photo of a dog", type=str)
    parser.add_argument("--image", default="output/compare_image_a.png", type=str)
    parser.add_argument("--image-a", default="output/compare_image_a.png", type=str)
    parser.add_argument("--image-b", default="output/compare_image_a.png", type=str)
    args = parser.parse_args()

    score = _post_json(args.base_url.rstrip("/") + "/score", {"prompt": args.prompt, "image": args.image})
    print("POST /score =>")
    print(json.dumps(score, indent=2))

    comp = _post_json(
        args.base_url.rstrip("/") + "/compare",
        {"prompt": args.prompt, "imageA": args.image_a, "imageB": args.image_b},
    )
    print("\nPOST /compare =>")
    print(json.dumps(comp, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

