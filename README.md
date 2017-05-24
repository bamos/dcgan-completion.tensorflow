# Image Completion with Deep Learning in TensorFlow

![](/completion.compressed.gif)

+ [See my blog post for more details and usage instructions](http://bamos.github.io/2016/08/09/deep-completion/).
+ This repository implements Raymond Yeh and Chen Chen et al.'s paper
  [Semantic Image Inpainting with Perceptual and Contextual Losses](https://arxiv.org/abs/1607.07539).
+ Most of the code in this repository was written by modifying a
  duplicate of [Taehoon Kim's](http://carpedm20.github.io/)
  [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) project,
  which is MIT-licensed.
  My modifications are also [MIT-licensed](./LICENSE).
+ The [./checkpoint](./checkpoint) directory contains a pre-trained
  model for faces, trained on the CelebA dataset for 20 epochs.

# Citations

Please consider citing this project in your
publications if it helps your research.
The following is a [BibTeX](http://www.bibtex.org/)
and plaintext reference.
The BibTeX entry requires the `url` LaTeX package.

```
@misc{amos2016image,
    title        = {{Image Completion with Deep Learning in TensorFlow}},
    author       = {Amos, Brandon},
    howpublished = {\url{http://bamos.github.io/2016/08/09/deep-completion}},
    note         = {Accessed: [Insert date here]}
}

Brandon Amos. Image Completion with Deep Learning in TensorFlow.
http://bamos.github.io/2016/08/09/deep-completion.
Accessed: [Insert date here]
```
