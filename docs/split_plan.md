# Split plan

## Why not the official split
The official RDD2022 test folders have no labels, so evaluation uses splits of the labelled train folders (India, Japan, Czech, United_States).

## The leakage problem
Image IDs are not in capture order, and the same road stretch appears in many non-adjacent images (frames from one drive, or several crops around one Street View position). A random split would put near-copies in both train and validation.

## Rule (per country, seed 0)
1. Each image becomes a 32x32 grayscale thumbnail, z-scored; similarity = correlation.
2. Link image pairs with correlation >= 0.97 and merge linked images into groups (union-find).
   Largest group at 0.97: US 6, Japan 7, Czech 96, India 95.
   Grouping at 0.92 was rejected: it merged roughly half of India (3,711 images) and Czech (1,439) into one group.
3. Assign whole groups: test ~20%, val ~10%, rest train.
4. Purge: train/val images with correlation >= 0.95 to any test image (and train images to val) are marked "purged" and unused.

## Result (images)
| Country | train | val | test | purged |
|---|---|---|---|---|
| United_States | 3,323 | 476 | 961 | 45 |
| Japan | 7,257 | 1,040 | 2,102 | 107 |
| Czech | 1,588 | 262 | 566 | 413 |
| India | 4,457 | 678 | 1,542 | 1,029 |

## Known limitations
- Similarity is a crude 32x32 pixel measure. The 0.92 / 0.95 / 0.97 cut-offs were judged by eye from 4 sampled pairs per band per country.
- Near-copies with correlation between 0.92 and 0.95 are NOT removed. Before purging, with a 20% val split, 56.6% (Czech) and 57.6% (India) of val images still had a train image >= 0.92.
- Czech images share a fixed windscreen pattern that inflates correlations.
- Exact duplicates exist (e.g. Japan_005550, _012402, _008969 at correlation 1.000).
- Rare classes: test boxes US D40 = 29, India D10 = 18; val has 8 and 10. No per-class claims from val.
- A pale vertical bar appears in several Japan images (unexplained; may be a camera artefact).