
## Preprocess Data Pipeline

### Overview

Suppose you have `input.json` with following format:
```
[
    {
        "img_path": "/path/to/image1.jpg",
        "caption" : ["caption1", "caption2", ...]
    },
    ...
]
```

The segmentation maps will have the following format
```
OUTPUT_DIR/
    seg/
        image1/
            mask_image1_noun1.png
            mask_image1_noun2.png
            ...
        image2/
            mask_image2_noun1.png
            mask_image2_noun2.png
            ...
        ...
```

