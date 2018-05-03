# licensePlateDatasetGenerator
Python package for generating a dataset composed of random Chilean license plates, with both styles (pre-2007 and current). This dataset is meant for Deep Learning algorithm training for OCR and text recognition purposes.

License plates can be generated as a CSV file and as separate JPG images. Text in images are randomly distorted and transformed for data augmentation.

Fonts utilized for LPs are Helvetica (pre-2007) and FE-Schrift (current font utilized in Chilean LPs).

## Usage
Run from command line.

The following arguments are accepted:

```
-s, --style   Selects license plate style (current or old) [default: current]
-r, --reps    Number of license plates to generate [default: 1]
```
