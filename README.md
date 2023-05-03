# Graph Cut Segmentation

Playing with graph cut algorithms for unsupervised image segmentation on black
and white images. The idea is to separate foreground from background by first
analyzing the histogram of color intensities and fitting a two-component Gaussian Mixture
Model (GMM).

<img src="https://user-images.githubusercontent.com/12631256/235843320-7314424d-4a1b-450e-8a8d-62ed8c8c0968.png" width="100" height="100">

The two distributions are then used to prime an augmenting path
method (default: Boykov-Kolmogorov) which finds the min-cut separating the two
distributions.

![test1_before_after](https://user-images.githubusercontent.com/12631256/235843181-ac57ccf7-7aac-400a-b207-18e2027f7a7b.png | width=640)

Future improvements:
- Currently the method assumes the brighter distribution is the foreground (as
  in a "white hot" infrared thermal camera image). A better way might be to
analyze the average of the center pixels vs the border pixels to figure out
which distribution is the foreground.
-  Segmenting multiple targets based on multi-component GMM.

# Install locally

```
python -m pip install -e .
```

# Run an example

```
python -m graph_cut_segmentation images/test1.png
```
