
# Anantadi Pipeline

A pipeline of the work done by the Christ Team on the Anantadi complete pipeline.For further information contact [@keegan](https://github.com/KeeganFernandesWork)



## Authors

- [@keegan](https://github.com/KeeganFernandesWork)


## Installation

Install 

```bash
  git clone https://github.com/dheekshi10/POC-1-VAPP.git
  cd POC-1-VAPP
  pip install -r requirements.txt
```
    
## Usage/Examples

To run the default parameters run
```bash
  python app.py
```

To run with your custom detector ,video and png image run
```bash
  python3 app.py --model "Models/cup_detector.pt" --smoothness 1 --video "Input_videos/Podcast.mp4" --advertisement "Input_ads/nescafe_cup.png"
```

The results should be saved in Output_videos folder




