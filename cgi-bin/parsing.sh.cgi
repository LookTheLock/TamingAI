#!/bin/bash
set -euo pipefail

magick - \
    -trim +repage \
    -resize 20x20 \
    -gravity center -background white -extent 28x28 \
    -negate \
    png:-  | /var/www/src/NeuralNetwork/src/NeuralNetwork
