# PRMS-Python
A Python module to assist with calibration, data processing and visualization for the Precipitation Runoff Modeling System (PRMS) computer program.

## Parameter File Tools

Currently you can do the following, starting from the root directory.

```python
from prms_python import Parameters
p = Parameters('test/data/parameter')

# select PRMS parameter by name, raising KeyError if DNE
snow_adj = p['snow_adj']
assert snow_adj.shape == (12, 16)

# assign values to PRMS parameter
import numpy as np
z = np.zeros(snow_adj.shape)
p['snow_adj'] = z  # now p['snow_adj'] is 12x16 matrix of zeros

# write modified parameters to file
p.write('newparameters')
```

## Notes

See the `models` directory for data and the Windows PRMS executable. To run PRMS just run

```
prms myrun.control
```

There are more built distributions for Linux and Windows in the `dists` directory.

* load_PRMSstatvar in ipynb in `scripts/statvarfile.ipynb`
* modifying params in `scripts/change_param.py`
* input and output paths defined in control file
