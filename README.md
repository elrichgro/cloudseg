# Full-sky Cloud Segmentation with the Infrared Cloud Camera at PMOD/WRC in Davos, Switzerland

![comparison-cirrus](https://user-images.githubusercontent.com/10598816/116904443-fa76bc00-ac3d-11eb-9a11-52c97a67d95e.png)

Code for our research project at ETH Zürich in collaboration with PMOD/WRC in Davos, Switzerland. We developed a deep learning based approach for continuous cloud monitoring using an all-sky infrared camera. 

## Abstract 
Cloud coverage is an important metric in weather prediction but is still most commonly determined by human observation. Automatic measurement using an RGB all-sky camera is unfortunately limited to daytime. To alleviate this problem the team at PMOD/WRC developed a prototype thermal infrared camera (IRCCAM). Their previous work utilized fixed thresholding which had problems with consistently detecting thin, high-altitude cirrus clouds. We utilized RGB images taken at the same location to create a labelled dataset on which we trained a deep learning semantic segmentation model. The resulting algorithm matches the previous approach in detecting thicker clouds and qualitatively outperforms it in detecting thinner cloud. We believe that coupled with the IRCCAM our model is comparable to human observation and can be used for continuous cloud coverage monitoring anywhere.

## Report
Read the full report [here](report.pdf).
