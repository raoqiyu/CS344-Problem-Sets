### Problem Set 2
Tile Algorithm 

```c
#define O_TILE_WIDTH 8
#define BLOCK_WIDTH (O_TILE_WIDTH + 8)
...
const dim3 blockSize(BLOCK_WIDTH,BLOCK_WIDTH);
const dim3 gridSize(numCols/O_TILE_WIDTH+1, numRows/O_TILE_WIDTH+1, 1);
```

Only ***O_TILE_WIDTH*** \* ***O_TILE_WIDTH*** threads participate in calculating outputs,
and only safe threads participate in writing output.
```c
if( threadIdx.y < O_TILE_WIDTH && threadIdx.x < O_TILE_WIDTH && (row < numRows) && (col < numCols) ){
...
}
```
In ***gaussian_blur*** kernel function, I used shared memeory do get fast memeory access time.
