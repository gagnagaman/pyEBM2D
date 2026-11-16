FC = gfortran
NAME = EBM
FFLAGS = -O2 -fallow-argument-mismatch $(shell nf-config --fflags)
FLIBS = $(shell nf-config --flibs)

SRCS = EBM.f90 grid.f90 app.f90 geography_input.f90 albedo_input.f90 A_value.f90 orbital_params.f90 monthly_output.f90 timesteps_output.f90

all: $(SRCS)
	$(FC) $(FFLAGS) $(SRCS) -o $(NAME) $(FLIBS)

