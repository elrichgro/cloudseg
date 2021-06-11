# Full-Sky Cloud Segmentation with Deep Learning

![comparison-cirrus](https://user-images.githubusercontent.com/10598816/116904443-fa76bc00-ac3d-11eb-9a11-52c97a67d95e.png)

Code for our research project at ETH Zürich in collaboration with PMOD/WRC in Davos, Switzerland. We developed a deep learning based approach for continuous cloud monitoring using an all-sky infrared camera.

## Abstract

Cloud coverage is an important metric in weather prediction but is still most commonly determined by human observation. Automatic measurement using an RGB all-sky camera is unfortunately limited to daytime. To alleviate this problem the team at PMOD/WRC developed a prototype thermal infrared camera (IRCCAM). Their previous work utilized fixed thresholding which had problems with consistently detecting thin, high-altitude cirrus clouds. We utilized RGB images taken at the same location to create a labelled dataset on which we trained a deep learning semantic segmentation model. The resulting algorithm matches the previous approach in detecting thicker clouds and qualitatively outperforms it in detecting thinner cloud. We believe that coupled with the IRCCAM our model is comparable to human observation and can be used for continuous cloud coverage monitoring anywhere.

## Report

Read the full report [here](report.pdf).

## Usage

### Prediction

To make predictions:

```shell
cloudseg input_file [--limit LIMIT] [--model MODEL]
```

e.g.

```shell
cloudseg example_input.mat
cloudseg example_input.mat --limit 100 --model model1 --output_file example_preds.mat
```

#### Arguments

- `input_file`: Path to the input file.
- `output_dir`: Directory to save the output file.
- `output_file`: Name for output file.
- `limit` (optional): Limit the number of predictions to make. The first input images are used up to the specified limit.
- `model`: Which model to use for predictions. Currently there is only one option: `model_1`.

#### Input file

The prediction script was designed for use with MATLAB files created by the
preprocessing steps at PMOD/WRC. The script expects an input file with the following fields:

- `BT` (n x 640 x 640): Calibrated temperature images from the IRCCAM.
- `TB` (n x 640 x 640): Clear-sky reference values calculated per timestamp.
- `mask` (640 x 640): Binary background mask for the IRCCAM location.

An example input file (`example_input.mat`) is provided.
