# Feature Proposal: Narwhals Integration for DataFrame-Agnostic Functionality

## Motivation

PanelSplit currently requires pandas, limiting users to a single dataframe backend. The scientific Python ecosystem is rapidly evolving with high-performance alternatives like Polars, and researchers increasingly need flexibility in their tool choices.

## Proposed Enhancement

Integrate **Narwhals** - a lightweight compatibility layer that enables seamless support for pandas, polars, and other dataframe libraries with **zero breaking changes** to existing code.

## Technical Benefits

### 🔧 **Zero-Overhead Abstraction**

```python
# Current: pandas-only
df = pd.DataFrame({'period': [1, 2, 3]})
ps = PanelSplit(periods=df['period'])  # Works only with pandas

# Proposed: universal compatibility  
df = pl.DataFrame({'period': [1, 2, 3]})  # or pandas, or any backend
ps = PanelSplit(periods=df['period'])     # Identical API, any backend
```

### 📊 **Performance & Choice**

- **Polars integration** - 5-30x faster operations for large datasets
- **Future-proof design** - automatic support for emerging dataframe libraries
- **Algorithmic consistency** - identical results across all backends
- **Memory efficiency** - users choose optimal backend for their data size

### 🎯 **Research Impact**

- **Tool flexibility** - researchers use preferred dataframe library
- **Reproducibility** - consistent behavior across computational environments
- **Scalability** - access to high-performance backends when needed
- **Interoperability** - seamless integration with diverse data processing pipelines

## Implementation Approach

**Core Pattern:**

```python
import narwhals as nw

# Universal dataframe operations
data_nw = nw.from_native(data, pass_through=True)
result = nw.concat(processed_data)
return nw.to_native(result)
```

**Strategic Design:**

- **Narwhals for dataframe operations** - zero-overhead passthrough to native implementations
- **NumPy for heavy computation** - maintain optimal performance for numerical operations
- **Robust fallbacks** - graceful handling of edge cases

## Compatibility Guarantee

**100% Backward Compatible** - All existing pandas code continues to work unchanged:

```python
# ✅ Existing workflows - no migration required
import pandas as pd
from panelsplit import PanelSplit

df = pd.DataFrame(...)  # Still works identically
ps = PanelSplit(periods=df['period'], n_splits=2)
```

## Dependencies

**Minimal Addition:** `narwhals>=1.43.1` (~50KB)

- **No breaking changes** to existing dependencies
- **Optional adoption** - users can choose when/if to use alternative backends
- **Active maintenance** - Narwhals is backed by major library authors

## Example Use Cases

### **High-Performance Research**

```python
import polars as pl
# 5-30x faster for large datasets
large_df = pl.read_parquet("large_timeseries.parquet")
ps = PanelSplit(periods=large_df['period'], n_splits=5)
```

### **Cross-Platform Reproducibility**

```python
# Same analysis, different environments
# Researcher A (pandas user)
df_pandas = pd.read_csv("data.csv")

# Researcher B (polars user)  
df_polars = pl.read_csv("data.csv")

# Identical PanelSplit API and results
ps = PanelSplit(periods=df['period'], n_splits=3)  # Works with both
```

## Request for Feedback

Would the maintainers be open to this enhancement? Key questions:

1. **Timeline compatibility** - Would this fit the project roadmap?
2. **Testing requirements** - Any specific backends or edge cases to prioritize?
3. **Documentation needs** - What level of migration guidance would be helpful?

I prepared a fully working proof of concept [here](https://github.com/m9o8/panelsplit) which also resolves the currently two outstanding GitHub issues ([#54](https://github.com/4Freye/panelsplit/issues/54) and [#59](https://github.com/4Freye/panelsplit/issues/59)).

I'd be happy about you're feedback and what you think of the idea.

---

**Benefits Summary:**
✅ Zero breaking changes - existing code unaffected  
✅ Performance options - users choose optimal backend  
✅ Future-proof - automatic support for new dataframe libraries  
✅ Research flexibility - tool choice freedom for diverse workflows  
✅ Minimal overhead - ~50KB dependency with zero runtime cost
