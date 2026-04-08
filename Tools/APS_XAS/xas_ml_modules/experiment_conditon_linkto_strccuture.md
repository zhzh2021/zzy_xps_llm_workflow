# XANES Feature → Fe Structure → Experimental Condition Linkage Guide

## Overview

This guide connects XANES spectroscopic features to Fe electronic/coordination structure and experimental conditions, enabling rational electrolyte design.

---

## Top 3 Structure-Sensitive Features (from ANOVA Analysis)

### 1. **Edge Shape/Curvature** (`second_derivative_zero`)

- **XANES Feature**: Energy where 2nd derivative = 0 (eV)
- **Fe Structure**: Shape of absorption edge curvature
  - Sharp edge → Well-defined oxidation state
  - Broad edge → Mixed valence or disordered environment
- **ANOVA Effect Size**: η² = 0.383 (Large - **38.3% of variance explained**)
- **Experimental Control**:
  - **No significant direct correlations** detected
  - This feature changes significantly between chemical groups but not linearly with individual conditions
  - Likely controlled by complex interactions between anion, ligand, pH
- **Scientific Meaning**:
  - Edge curvature reflects the **transition from Fe 1s to 4p states**
  - Inflection point energy shift indicates changes in effective nuclear charge seen by Fe
  - Most sensitive "fingerprint" for distinguishing different electrolyte formulations

---

### 2. **Coordination Order/Disorder** (`edge_slope`)

- **XANES Feature**: Slope of absorption edge (normalized units/eV)
- **Fe Structure**: Structural order in Fe coordination environment
  - Steep slope → Crystalline, ordered coordination (e.g., octahedral FeO₆)
  - Gentle slope → Amorphous, disordered, or multiple coordination geometries
- **ANOVA Effect Size**: η² = 0.333 (Large - **33.3% of variance explained**)
- **Experimental Control**:
  - **Ligand Type** (r = +0.41, p < 0.05)
    - Malic acid (0) → Tartaric acid (1) **increases** edge_slope
    - Interpretation: Tartaric acid creates more ordered Fe coordination
  - **XANES Centroid** (r = -0.39, p < 0.05)
    - Higher average energy → lower edge slope
- **Scientific Meaning**:
  - Edge slope reflects the **density of final states** available for electron transition
  - Steeper slope = more uniform local structure (better defined coordination geometry)
  - **Tartaric acid** (with two COOH groups) may form more regular Fe-carboxylate coordination than malic acid

---

### 3. **Fe-Ligand Covalency** (`white_line_intensity`)

- **XANES Feature**: Height of first major peak after edge (normalized)
- **Fe Structure**: Degree of covalent character in Fe-ligand bonds
  - High intensity → More covalent Fe-ligand bonding (electron density transfer to Fe 3d orbitals)
  - Low intensity → More ionic bonding (less mixing of Fe-ligand orbitals)
  - Also reflects **density of unoccupied Fe 3d states**
- **ANOVA Effect Size**: η² = 0.286 (Large - **28.6% of variance explained**)
- **Experimental Control**:
  - **No significant direct correlations** detected
  - Changes significantly between chemical groups but not linearly with individual conditions
- **Scientific Meaning**:
  - White line arises from **Fe 1s → 3d quadrupole transitions**
  - Intensity depends on:
    1. **Oxidation state**: Fe³⁺ has MORE unoccupied 3d orbitals → higher intensity
    2. **Ligand field strength**: Stronger field → greater orbital mixing → higher intensity
    3. **Coordination geometry**: Affects d-orbital splitting and occupancy
  - Critical for understanding Fe redox state and bonding in electrolyte

---

## Additional Structure Parameters

### 4. **Fe Oxidation State** (`e0` - Edge Energy)

- **XANES Feature**: Energy at absorption edge inflection point (eV)
- **Fe Structure**: Oxidation state of Fe (Fe²⁺ vs Fe³⁺)
  - Higher E₀ → Higher oxidation state (Fe³⁺)
  - Lower E₀ → Lower oxidation state (Fe²⁺)
  - ~2-3 eV shift from Fe²⁺ to Fe³⁺
- **Experimental Influence**:
  - pH: Lower pH can stabilize Fe²⁺
  - Ligand complexation: Affects effective charge on Fe
  - Oxygen exposure: Can oxidize Fe²⁺ → Fe³⁺
- **Why It Matters**: Determines electrochemical activity and solubility

---

### 5. **Coordination Geometry** (`pre_edge_area`)

- **XANES Feature**: Integrated area of pre-edge peak
- **Fe Structure**: Degree of inversion symmetry breaking
  - Large pre-edge → Tetrahedral coordination (Td symmetry)
  - Small pre-edge → Octahedral coordination (Oh symmetry)
  - 3d-4p orbital mixing allows 1s → 3d transitions
- **Experimental Influence**:
  - Ligand sterics: Bulky ligands favor tetrahedral
  - Concentration: High concentration can distort geometry
- **Why It Matters**: Coordination geometry affects reactivity and stability

---

## Experimental Condition → Structure Linkage

### Categorical Variables

#### **Ligand Type** (Malic vs Tartaric Acid)

**Effect on Structure:**

- ✓ **Coordination Order** (edge_slope, r = +0.41)
  - Tartaric acid → More ordered Fe coordination
  - Mechanism: Two symmetric COOH groups in tartaric acid create regular chelate rings

**Chemical Formulas:**

- Malic acid: HOOC-CH₂-CH(OH)-COOH (1 COOH, 1 OH)
- Tartaric acid: HOOC-CH(OH)-CH(OH)-COOH (2 COOH, 2 OH, symmetric)

**Design Implication**: Use tartaric acid for more homogeneous electrolytes

---

#### **Anion Type** (Cl⁻ vs SO₄²⁻)

**Effect on Structure:**

- ✗ **No significant correlations** detected (p > 0.05)

**Possible Reasons:**

1. Anions may be outer-sphere (not directly coordinated to Fe)
2. Organic ligands dominate inner coordination sphere
3. Sample size (N=27) may be insufficient to detect subtle effects

**Note**: ANOVA still shows anion type affects overall spectral fingerprint (via second_derivative_zero)

---

### Continuous Variables

#### **pH** (2-5)

**Expected Effects** (from literature):

- Lower pH → Favors Fe²⁺ (prevents oxidation)
- Higher pH → Fe(OH)₃ precipitation risk for Fe³⁺

**Observed Correlations**: Not statistically significant in this dataset

- Possible pH buffering by organic acids

---

#### **Ligand Concentration** (0.05-1.0 M)

**Effect on Structure:**

- Strongest correlations with multiple features (from condition impact analysis)
- Higher concentration → More Fe-ligand complexation
- Can shift coordination number and geometry

---

#### **Anion Concentration** (0.1-1.0 M)

**Effect on Structure:**

- Secondary importance (after ligand concentration)
- Affects ionic strength and activity coefficients

---

## Practical Application

### For Electrolyte Design:

1. **To control Fe oxidation state** (E₀):
   - Adjust pH and oxygen exposure
   - Use reducing agents if Fe²⁺ needed

2. **To create ordered structures** (edge_slope):
   - Prefer **tartaric acid** over malic acid
   - Optimize ligand concentration

3. **To maximize Fe-ligand covalency** (white_line_intensity):
   - Use strong-field ligands (carboxylates, phosphates)
   - Control oxidation state (Fe³⁺ typically higher)

4. **To distinguish formulations**:
   - Monitor **second_derivative_zero** (most sensitive)
   - Use XANES as quality control metric

---

## How to Use These Visualizations

### 1. **feature_structure_linkage.png**

- Shows XANES features ranked by ANOVA effect size
- Bars → How much chemistry affects each structural property
- Larger bars = better quality control metrics

### 2. **condition_structure_impact.png** (2-panel)

- **Left panel**: Categorical conditions (anion/ligand type)
  - Green → Positive correlation (increase)
  - Red → Negative correlation (decrease)
- **Right panel**: Continuous conditions (pH, concentrations)
  - Top 10 strongest correlations
  - Shows which experimental knobs control which structures

### 3. **feature_structure_condition_linkage.csv**

- Tabular summary linking all three levels
- Use for quantitative analysis and reporting

---

## Statistical Notes

- **Effect Size** (η²): Proportion of variance explained
  - Large: η² > 0.14 (14%)
  - Medium: 0.06 < η² < 0.14
  - Small: η² < 0.06

- **Correlation** (r): Pearson or point-biserial
  - |r| > 0.4: Moderate to strong
  - |r| > 0.3: Weak to moderate
  - p < 0.05: Statistically significant

- **Sample Size**: N = 27 Fe K-edge XANES spectra
  - May limit detection of weak correlations
  - Results most robust for large effect sizes

---

## Conclusion

This analysis successfully links:

1. **Experimental Design** (anion type, ligand type, pH, concentrations)
2. **Fe Electronic/Coordination Structure** (oxidation state, coordination geometry, bonding)
3. **XANES Spectroscopic Features** (quantitative fingerprints)

The three most important structural properties affected by chemistry are:

1. Edge shape/curvature (38% variance)
2. Coordination order/disorder (33% variance)
3. Fe-ligand covalent bonding (29% variance)

Use these insights to:

- Design electrolytes with targeted Fe structure
- Predict XANES spectra from formulation
- Optimize synthesis conditions for desired properties
- Implement quality control using XAS fingerprinting

---

**Generated**: 2026-03-04  
**Dataset**: 27 Fe K-edge XANES samples (FeCl₂/FeSO₄ + malic/tartaric acid electrolytes)  
**Analysis**: PCA, K-means, ANOVA, Random Forest, t-SNE  
**Workflow**: test_prj_ml_workflow.py
