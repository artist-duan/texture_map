# Texture map

This repository contains vertex and uv texture map.

# Progress

- [x] vertex map

- [x] uv map
  - [x] Max Projection Area
  - [x] Graph Optimization
  - [ ] Color Adjustment
  - [ ] Photo-Consistency Check(occlude)

# Installation

- ```pip install -r requirements.txt ```

- ```git clone https://github.com/js-duan/texture_map.git```

# Data

- mesh(triangle)
- images
- poses (w2c)
- intrinsic

# RUN

## vertex texture map: Fast

- include ``min_depth``, ``best view``, ``mean view``, ``weight-mean view`` and ``optimization``(implement use open3d) vertex texture map

- ```python vertex_map_example.py --path PATH_TO_DATA --mesh NAME_OF_MESH [--depth PATH_TO_DEPTH] [--display]```

- if u don't have depth, the code will render depth use mesh

## uv map: Slow but performance is great

- include ``best view``, ``max projection area`` and ``graph optimization``(optim-algorithm implemented by [graph-optimzation](https://github.com/DIYer22/graph_optimization)) uv map

- ```python uv_map_example.py --path PATH_TO_DATA --mesh NAME_OF_MESH [--depth PATH_TO_DEPTH] [--display] [--label]```

- use [--label] will save save image label of each face in mesh

# Examples

## original mesh

-  <img src="./statics/original.png" width = "300"/>

## vertex map
- weight-mean
  
  <img src="./statics/weight-mean.png" width = "300"/>

- optimization(open3d)

  <img src="./statics/optim.png" width = "300"/>

## uv map
- Max Projection Area(The same color in the right image indicates projection onto the same image)
  
  <img src="./statics/max-area.png" width = "300"/> <img src="./statics/max-area-label.png" width = "265"/>

- Graph-Opimization
  
  <img src="./statics/graph-optim.png" width = "300"/> <img src="./statics/graph-optim-label.png" width = "260"/>