PROGRAM print_orbit
IMPLICIT NONE
REAL:: ecc, ob, per
CALL orbital_params(1950,ecc,ob,per)
PRINT *, "ecc=", ecc, "ob=", ob, "per=", per
END PROGRAM
