"""Command-line prediction: `python predict.py path/to/fruit.jpg`"""

import sys

from PIL import Image

from inference import FreshnessClassifier


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image = Image.open(sys.argv[1])
    clf = FreshnessClassifier()
    verdict, scores = clf.predict(image)

    print(verdict)
    print("\nScores:")
    for cls, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:10s} {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
