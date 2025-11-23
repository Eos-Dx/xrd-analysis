# Hankel Transform: Input vs Output Explained

## Your Setup
```
Input:  2D array (N × M) in polar coordinates
        N = angles (0° to 360°)
        M = radial bins (q-values)

Output: 2D array (N × M) in frequency domain
        Same shape!
```

## Concrete Example

### INPUT: Polar Data
```
Imagine a 4×3 polar image:

       Radial (q-values): q₁, q₂, q₃
Angles        │   │   │
      0°      │ 10│ 20│ 15│
     90°      │ 12│ 22│ 18│
    180°      │ 11│ 21│ 17│
    270°      │ 10│ 23│ 16│

Shape: (4 angles) × (3 q-bins)
```

**What this represents:**
- Each row = intensity measured at a specific angle
- Each column = intensity at a specific q-value (radius)
- Data in SPATIAL domain


### OUTPUT: Hankel Transformed Data
```
After QDHT (Order=0):

       Frequency domain: k₁, k₂, k₃
Angles        │   │   │
      0°      │ 2.5│-0.8│ 0.1│
     90°      │ 2.3│-0.9│ 0.2│
    180°      │ 2.4│-0.7│ 0.1│
    270°      │ 2.6│-0.8│ 0.0│

Shape: (4 angles) × (3 frequencies)
```

**What this represents:**
- Each row = frequency spectrum at a specific angle
- Each column = amplitude of that frequency component
- Data in FREQUENCY domain


## What Does Each Value Mean?

### INPUT[angle, q-bin] = Intensity
```
Input[0°, q₂] = 20
↓
"At angle 0° and q-value q₂, the X-ray intensity is 20"
```

### OUTPUT[angle, frequency] = Amplitude of frequency component
```
Output[0°, k₂] = -0.8
↓
"At angle 0°, the frequency component k₂ has amplitude -0.8"
```


## Physical Interpretation

### INPUT (Spatial Domain)
```
Raw intensities:
- Shows peaks and valleys at different q-values
- Direct measurement from detector
- Hard to identify periodic structures
```

### OUTPUT (Frequency Domain)
```
Frequency amplitudes:
- Low frequencies (k≈0): Long-range patterns
- High frequencies (k→max): Fine structures, noise
- Amplitude tells you: "How strong is this frequency?"

Example:
- If Output[θ, low_k] is large  → Strong long-range periodicity at angle θ
- If Output[θ, high_k] is small → Less high-frequency noise at angle θ
```


## Step-by-Step: What Happens to One Row

**Input row at θ=0°:**
```
[10, 20, 15]  ← Spatial intensities at different q-values
```

**QDHT applies this transformation:**
```
For each frequency k_j:
  Output[0°, j] = ∫₀^R J₀(k_j × r) × Input_radial(r) × r dr

Where:
  J₀ = Bessel function
  k_j = frequency component j
  r = radius (q-value)
  Input_radial = your intensity values [10, 20, 15]
```

**Output row at θ=0°:**
```
[2.5, -0.8, 0.1]  ← Frequency amplitudes
```


## Key Points

1. **Shape stays the same** (N × M) but meaning changes
2. **Each row is transformed independently** (for Order=0)
3. **Column axis changes:**
   - Input columns = q-values (spatial radii)
   - Output columns = frequencies (k-values)
4. **Values change meaning:**
   - Input = raw intensity
   - Output = frequency amplitude (can be positive or negative)


## Why Use It?

### Use Case 1: Detect Periodic Structures
```
Input:  [100, 105, 100, 105, 100, 105, ...]  (periodic)
Output: [0.1, 50.2, 0.1, 0.05, 0.02, ...]   (spike at frequency k₂)

"Aha! There's strong periodicity at frequency k₂"
```

### Use Case 2: Noise Analysis
```
Input:  [100, 102, 99, 101, 98, 103, ...]  (with noise)
Output: [45.3, 2.1, 0.8, 0.3, 0.1, 0.05, ...] (energy spectrum)

High-frequency components are small → mostly noise
Low-frequency component is large → true signal
```

### Use Case 3: Feature Extraction for ML
```
Input:  Raw polar data (hard to compare)
Output: Frequency spectrum (easy to compare with ML models)

Two samples with same structure → similar Output patterns
Two samples with different structure → different Output patterns
```


## Summary Table

| Aspect | INPUT | OUTPUT |
|--------|-------|--------|
| **Domain** | Spatial (q-space) | Frequency (k-space) |
| **Row represents** | Intensity at different q-values | Frequency spectrum |
| **Column represents** | Radial position (q-bin) | Frequency component |
| **Values mean** | Raw detector intensity | Frequency amplitude |
| **Uses** | See raw data | Analyze periodicity, noise |


## In Your Code

```python
polar_img = row[self.column].copy()[:, self.start_radius:]  # Shape: N × M (spatial)
r = row["q_range"][self.start_radius:]                      # Radial coordinates

transformer = HankelTransform(order=0, max_radius=R, n_points=n_radial)
H = transformer.qdht(polar_img, axis=1)                     # Apply transform along axis=1

X_copy.at[i, "hankel"] = H                                  # Store: Shape N × M (frequency)
```

The `axis=1` means: **Transform each row independently along the radial direction**
- Each angular profile becomes a frequency spectrum
- Shape preserved: (N angles) × (M frequencies)
