# Collaborative-Filtering
Recommendation Engine using Collaborative Filtering in Spark

Considering the latent factor modeling of utility matrix M (e.g., a rating matrix where rows represent users and columns products such as movies), the goal is to decompose M into lower-rank matrices U and V such that the difference between M and UV is minimized.

This involves 2 parts:

1. Implentation of incremental UV decomposition algorithm.
The learning starts with learning elements in U row by row, i.e., U[1,1], U[1,2], …, U[2, 1], …
It then moves on to learn elements in V column by column, i.e., V[1,1], V[2,1], …, V[1, 2], …
When learning an element, it uses the latest value learned for all other elements. It should compute the optimal value for the element to minimize the current RMSE as described in class.

2. Modified ALS to give RMSE outputs over a set of iterations.

## Input:
Utility matrix  M where rows represent users and columns represent movies.

## Execution format for UV decomposition:
python uv.py input-matrix n m f k

## Execution format for ALS :
bin/spark-submit als.py input-matrix n m f k p output-file

## Output:
RMSE values as console output.


