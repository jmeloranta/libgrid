/*
 * Constants to convert from/to atomic units.
 *
 */

#ifndef __ATOMIC_UNITS__
#define __ATOMIC_UNITS__
#define GRID_AUTOANG  0.52917725             /* Bohr to Angstrom  */
#define GRID_AUTOM    (GRID_AUTOANG * 1E-10) /* Bohr to meter */
#define GRID_AUTOK    3.15773213e5           /* Hartree to Kelvin */
#define GIRD_AUTOJ    4.359748E-18           /* Hartree to Joule */
#define GRID_AUTOCM1  (3.15773213e5/1.439)   /* Hartree to wavenumbers */
#define GRID_HZTOCM1  3.335641E-11           /* Hz to wavenumbers */
#define GRID_AUTOAMU  (1.0/1822.88853006)    /* Me (mass of electron) to atomic mass unit */
#define GRID_AUTOFS   0.02418884             /* Atomic time unit to femtosecond */
#define GRID_AUTOS    (GRID_AUTOFS * 1E-15)  /* Atomic time unit to second */
#define GRID_AUTOBAR  2.9421912E8            /* Atomic pressure unit (Hartree/bohr**2) to bar */
#define GRID_AUTOPA   2.9421912E13           /* Atomic pressure unit to Pa */
#define GRID_AUTOATM  (GRID_AUTOBAR * 9.869E-1) /* Atomic pressure unit to atm */
#define GRID_AUTOMPS  (GRID_AUTOM / GRID_AUTOS) /* Atomic velocity unit to m/s */
#define GRID_AUTON    8.2387225E-8           /* Atomic force unit to N */
#define GRID_AUTOVPM  5.14220652E11          /* Atomic electric field strength to V/m */
#define GRID_AUTOPAS  (GRID_AUTOPA * GRID_AUTOS) /* Atomic viscosity unit to Pa s */
#define GRID_AUKB     3.1668773658e-06       /* Boltzmann constant in a.u. (k_B) */
#define GRID_AVOGADRO 6.022137E23            /* Avogardro's number: convert from # of particles to moles */
#endif

