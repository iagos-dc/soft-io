# SOFT-IO v2.0
This is a version 2.0 of an implementation of the methodology of source attribution using FLEXPART and carbon monoxide emission inventories described in the paper

Sauvage, B., Fontaine, A., Eckhardt, S., Auby, A., Boulanger, D., Petetin, H., Paugam, R., Athier, G., Cousin, J.-M., Darras, S., Nédélec, P., Stohl, A., Turquety, S., Cammas, J.-P., and Thouret, V.: Source attribution using FLEXPART and carbon monoxide emission inventories: SOFT-IO version 1.0, Atmos. Chem. Phys., 17, 15271–15292, https://doi.org/10.5194/acp-17-15271-2017, 2017.


## Source code structure
The main part of the code source is in `softio` package. The main routine of that package is `get_co_contrib`.


## Deployment
This software needs some external data to be run (emission inventories data, FLEXPART simulation results, etc.),
and for the moment (due to several hard-coded paths to data files) is intended to be run on
OMP's computational cluster NUWA.
