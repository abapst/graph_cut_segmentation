# Graph Cut Segmentation

Playing with graph cut algorithms for unsupervised image segmentation on black
and white images. The idea is to separate foreground from background by first
analyzing the histogram of color intensities and fitting a two-component Gaussian Mixture
Model (GMM). The two distributions are then used to prime an augmenting path
method (default: Boykov-Kolmogorov) which finds the min-cut separating the two
distributions.

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
