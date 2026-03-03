
# Project TODO List

## Project Overview
DSLR (Data Science from Scratch) - A Hogwarts-themed data analysis project implementing statistical functions from scratch.

## Current Status
✅ Completed:
- `maths.py` - Statistical functions (mean, std, quartile, min_max)
- `CsvManip.py` - CSV loading and numeric feature extraction
- `describe.py` - FeatureMetrics and DescribeReport classes

⚠️ In Progress:
- `histogram.py` - Started but incomplete

---

## TODO Items

### Phase 1: Complete Histogram Module
- [ ] Finish implementing histogram.py
- [ ] Add bin calculation logic
- [ ] Add frequency distribution calculation

### Phase 2: Visualisation
- [ ] Add histogram plotting functionality
- [ ] Add scatter plot (pair plot)
- [ ] Add house-based comparison plots

### Phase 3: Model Training
- [ ] Implement Trainer class
- [ ] Implement Predict class
- [ ] Add model evaluation metrics

### Phase 4: Package Organization
- [ ] Create proper Python package structure
- [ ] Move math functions to `dslr/math.py`
- [ ] Move data manipulation to `dslr/dataManipulation.py`
- [ ] Consolidate shared utilities
- [ ] Optimize to avoid redundant sorting (single sort for all metrics)
