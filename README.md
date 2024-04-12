# DistNet2D: Leveraging long-range temporal information for efficient segmentation and tracking

This repository contains python code for training the neural network.

[Link to preprint](https://arxiv.org/abs/2310.19641)

[Link to tutorial](https://github.com/jeanollion/bacmman/wiki/DistNet2D)

Jean Ollion, Martin Maliet, Caroline Giuglaris, Elise Vacher, Maxime Deforet

Extracting long tracks and lineages from videomicroscopy requires an extremely low error rate, which is challenging on complex datasets of dense or deforming cells. Leveraging temporal context is key to overcoming this challenge. We propose DistNet2D, a new deep neural network (DNN) architecture for 2D cell segmentation and tracking that leverages both mid- and long-term temporal information. DistNet2D considers seven frames at the input and uses a post-processing procedure that exploits information from the entire video to correct segmentation errors. DistNet2D outperforms two recent methods on two experimental datasets, one containing densely packed bacterial cells and the other containing eukaryotic cells. It is integrated into an ImageJ-based graphical user interface for 2D data visualization, curation, and training. Finally, we demonstrate the performance of DistNet2D on correlating the size and shape of cells with their transport properties over large statistics, for both bacterial and eukaryotic cells.
