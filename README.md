# Video Stabilization

There were 2 goals that this project accomplishes. For better understanding, lookup ***{Assignment2,Assignment3}.pdf***.

To run code corresponding to Assignment2.pdf, change to *src/ directory* and type linux shell:

```
python final_lucas.py <source image path> <target image path> <epsilon> <output_path>
```

To run code corresponding to Assignment2.pdf for video stabilization, change to *src/ directory* and type linux shell:

```
python stable_without_temp.py <input_video_path> <output_video_path>
```

To run code corresponding to Assignment2.pdf for video stabilization, change to *Corrections/ directory* and type linux shell:

```
python affine_metric.py <input_video_path>
```


## Notes on running the code

1. The code has been tested and developed in ***python2*** using ***Ubuntu 16.04***.
2. For part 2 of Assignment 2, there are 2 codes. The one mentioned above focus on ***stabilizing complete frame***. The other in directory stabilizes a ***template***. In either case you are asked to select a template.

## Notes on directories

* **CorrectedImage**: Contains some sample images that were affine and metric corrected.
* **Corrections**: Contains source code for performing corrections(Part 3).
* **Image**: Minor collection of images used for testing purpose.
* **Latex Works**: Used for preparing report.
* **Stabilized_Video**: Contain some video stabilized by my heuristic.
* **Video**: Sample videos.
* **WarpedImage**: Output of warping algorithm using non-linear optimization.
* **src**: Source code for Part 2.
