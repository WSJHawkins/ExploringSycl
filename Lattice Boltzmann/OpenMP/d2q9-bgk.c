/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/* struct to hold the 'speed' values */
typedef struct
{
  float* restrict s0;
  float* restrict s1;
  float* restrict s2;
  float* restrict s3;
  float* restrict s4;
  float* restrict s5;
  float* restrict s6;
  float* restrict s7;
  float* restrict s8;
} t_speeds;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_scells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speeds speeds, t_speeds tmp_speeds, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_scells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_scells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_scells, &obstacles, &av_vels);


  t_speeds speeds;
  speeds.s0 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s1 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s2 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s3 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s4 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s5 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s6 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s7 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  speeds.s8 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  t_speeds tmp_speeds;
  tmp_speeds.s0 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s1 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s2 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s3 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s4 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s5 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s6 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s7 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
  tmp_speeds.s8 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);

  #pragma omp parallel for
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      speeds.s0[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[0];
      speeds.s1[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[1];
      speeds.s2[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[2];
      speeds.s3[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[3];
      speeds.s4[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[4];
      speeds.s5[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[5];
      speeds.s6[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[6];
      speeds.s7[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[7];
      speeds.s8[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[8];
      tmp_speeds.s0[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[0];
      tmp_speeds.s1[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[1];
      tmp_speeds.s2[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[2];
      tmp_speeds.s3[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[3];
      tmp_speeds.s4[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[4];
      tmp_speeds.s5[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[5];
      tmp_speeds.s6[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[6];
      tmp_speeds.s7[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[7];
      tmp_speeds.s8[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[8];
    }
  }
  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int tt = 0; tt < params.maxIters/2; tt++)
  {
    av_vels[2*tt] = timestep(params, speeds, tmp_speeds, obstacles);
    av_vels[2*tt+1] = timestep(params, tmp_speeds, speeds, obstacles);

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);


  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
    cells[ii + jj*params.nx].speeds[0] = speeds.s0[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[1] = speeds.s1[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[2] = speeds.s2[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[3] = speeds.s3[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[4] = speeds.s4[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[5] = speeds.s5[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[6] = speeds.s6[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[7] = speeds.s7[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[8] = speeds.s8[ii + jj*params.nx];
    }
  }


  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_scells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speeds speeds, t_speeds tmp_speeds,int* restrict obstacles)
{

  float* restrict s0 = speeds.s0;
  float* restrict s1 = speeds.s1;
  float* restrict s2 = speeds.s2;
  float* restrict s3 = speeds.s3;
  float* restrict s4 = speeds.s4;
  float* restrict s5 = speeds.s5;
  float* restrict s6 = speeds.s6;
  float* restrict s7 = speeds.s7;
  float* restrict s8 = speeds.s8;
  float* restrict tmp_s0 = tmp_speeds.s0;
  float* restrict tmp_s1 = tmp_speeds.s1;
  float* restrict tmp_s2 = tmp_speeds.s2;
  float* restrict tmp_s3 = tmp_speeds.s3;
  float* restrict tmp_s4 = tmp_speeds.s4;
  float* restrict tmp_s5 = tmp_speeds.s5;
  float* restrict tmp_s6 = tmp_speeds.s6;
  float* restrict tmp_s7 = tmp_speeds.s7;
  float* restrict tmp_s8 = tmp_speeds.s8;
  //ACCELERATE FLOW
  /* compute weighting factors */
  const float w11 = params.density * params.accel / 9.f;
  const float w21 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  #pragma vector aligned
  #pragma ivdep
  __assume_aligned(s0,64);
  __assume_aligned(s1,64);
  __assume_aligned(s2,64);
  __assume_aligned(s3,64);
  __assume_aligned(s4,64);
  __assume_aligned(s5,64);
  __assume_aligned(s6,64);
  __assume_aligned(s7,64);
  __assume_aligned(s8,64);
  __assume(params.nx%128==0);
  __assume(params.nx%64==0);
  __assume(params.nx%32==0);
  __assume(params.nx%16==0);
  __assume(params.nx%8==0);
  __assume(params.nx%4==0);
  __assume(params.nx%2==0);
  #pragma omp parallel for simd default(none) shared(s0,s1,s2,s3,s4,s5,s6,s7,s8,obstacles,jj)
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (s3[ii + jj*params.nx] - w11) > 0.f
        && (s6[ii + jj*params.nx] - w21) > 0.f
        && (s7[ii + jj*params.nx] - w21) > 0.f)
    {
      /* increase 'east-side' densities */
      s1[ii + jj*params.nx] += w11;
      s5[ii + jj*params.nx] += w21;
      s8[ii + jj*params.nx] += w21;
      /* decrease 'west-side' densities */
      s3[ii + jj*params.nx] -= w11;
      s6[ii + jj*params.nx] -= w21;
      s7[ii + jj*params.nx] -= w21;
    }
  }

  //PROPOGATE
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float c_sq_inv = 3.f;
  const float temp1 = 4.5f;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */


  //AVERAGE VELOCITY VARS
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */


  #pragma vector aligned
  #pragma ivdep
  __assume_aligned(s0,64);
  __assume_aligned(s1,64);
  __assume_aligned(s2,64);
  __assume_aligned(s3,64);
  __assume_aligned(s4,64);
  __assume_aligned(s5,64);
  __assume_aligned(s6,64);
  __assume_aligned(s7,64);
  __assume_aligned(s8,64);
  __assume_aligned(tmp_s0,64);
  __assume_aligned(tmp_s1,64);
  __assume_aligned(tmp_s2,64);
  __assume_aligned(tmp_s3,64);
  __assume_aligned(tmp_s4,64);
  __assume_aligned(tmp_s5,64);
  __assume_aligned(tmp_s6,64);
  __assume_aligned(tmp_s7,64);
  __assume_aligned(tmp_s8,64);
  __assume(params.nx%128==0);
  __assume(params.nx%64==0);
  __assume(params.nx%32==0);
  __assume(params.nx%16==0);
  __assume(params.nx%8==0);
  __assume(params.nx%4==0);
  __assume(params.nx%2==0);
  __assume(params.ny%128==0);
  __assume(params.ny%64==0);
  __assume(params.ny%32==0);
  __assume(params.ny%16==0);
  __assume(params.ny%8==0);
  __assume(params.ny%4==0);
  __assume(params.ny%2==0);
  #pragma omp parallel for default(none) shared(s0,s1,s2,s3,s4,s5,s6,s7,s8,tmp_s0,tmp_s1,tmp_s2,tmp_s3,tmp_s4,tmp_s5,tmp_s6,tmp_s7,tmp_s8,obstacles) reduction(+:tot_cells) reduction(+:tot_u)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index = ii + jj*params.nx;
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_s0[index] = s0[index]; /* central cell, no movement */
      tmp_s1[index] = s1[x_w + jj*params.nx]; /* east */
      tmp_s2[index] = s2[ii + y_s*params.nx]; /* north */
      tmp_s3[index] = s3[x_e + jj*params.nx]; /* west */
      tmp_s4[index] = s4[ii + y_n*params.nx]; /* south */
      tmp_s5[index] = s5[x_w + y_s*params.nx]; /* north-east */
      tmp_s6[index] = s6[x_e + y_s*params.nx]; /* north-west */
      tmp_s7[index] = s7[x_e + y_n*params.nx]; /* south-west */
      tmp_s8[index] = s8[x_w + y_n*params.nx]; /* south-east */


        /* compute local density total */
        float local_density = 0.f;
        local_density += tmp_s0[index];
        local_density += tmp_s1[index];
        local_density += tmp_s2[index];
        local_density += tmp_s3[index];
        local_density += tmp_s4[index];
        local_density += tmp_s5[index];
        local_density += tmp_s6[index];
        local_density += tmp_s7[index];
        local_density += tmp_s8[index];

        /* compute x velocity component */
        float u_x = (tmp_s1[index]
                      + tmp_s5[index]
                      + tmp_s8[index]
                      - tmp_s3[index]
                      - tmp_s6[index]
                      - tmp_s7[index])
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_s2[index]
                      + tmp_s5[index]
                      + tmp_s6[index]
                      - tmp_s4[index]
                      - tmp_s7[index]
                      - tmp_s8[index])
                     / local_density;

        /* velocity squared */
        float temp2 = - (u_x * u_x + u_y * u_y)/ (2.f * c_sq);

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f + temp2);
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u_x * c_sq_inv
                                         + (u_x * u_x) * temp1
                                         + temp2);
        d_equ[2] = w1 * local_density * (1.f + u_y * c_sq_inv
                                         + (u_y * u_y) * temp1
                                         + temp2);
        d_equ[3] = w1 * local_density * (1.f - u_x * c_sq_inv
                                         + (u_x * u_x) * temp1
                                         + temp2);
        d_equ[4] = w1 * local_density * (1.f - u_y * c_sq_inv
                                         + (u_y * u_y) * temp1
                                         + temp2);
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + (u_x + u_y) * c_sq_inv
                                         + ((u_x + u_y) * (u_x + u_y)) * temp1
                                         + temp2);
        d_equ[6] = w2 * local_density * (1.f + (-u_x + u_y) * c_sq_inv
                                         + ((-u_x + u_y) * (-u_x + u_y)) * temp1
                                         + temp2);
        d_equ[7] = w2 * local_density * (1.f + (-u_x - u_y) * c_sq_inv
                                         + ((-u_x - u_y) * (-u_x - u_y)) * temp1
                                         + temp2);
        d_equ[8] = w2 * local_density * (1.f + (u_x - u_y) * c_sq_inv
                                         + ((u_x - u_y) * (u_x - u_y)) * temp1
                                         + temp2);
        float tmp;

        tmp_s0[index] = (obstacles[index]) ? tmp_s0[index] : tmp_s0[index] + params.omega * (d_equ[0] - tmp_s0[index]);
        tmp = tmp_s1[index];
        tmp_s1[index] = (obstacles[index]) ? tmp_s3[index] : tmp_s1[index] + params.omega * (d_equ[1] - tmp_s1[index]);
        tmp_s3[index] = (obstacles[index]) ? tmp : tmp_s3[index] + params.omega * (d_equ[3] - tmp_s3[index]);
        tmp = tmp_s2[index];
        tmp_s2[index] = (obstacles[index]) ? tmp_s4[index] : tmp_s2[index] + params.omega * (d_equ[2] - tmp_s2[index]);
        tmp_s4[index] = (obstacles[index]) ? tmp : tmp_s4[index] + params.omega * (d_equ[4] - tmp_s4[index]);
        tmp = tmp_s5[index];
        tmp_s5[index] = (obstacles[index]) ? tmp_s7[index] : tmp_s5[index] + params.omega * (d_equ[5] - tmp_s5[index]);
        tmp_s7[index] = (obstacles[index]) ? tmp : tmp_s7[index] + params.omega * (d_equ[7] - tmp_s7[index]);
        tmp = tmp_s6[index];
        tmp_s6[index] = (obstacles[index]) ? tmp_s8[index] : tmp_s6[index] + params.omega * (d_equ[6] - tmp_s6[index]);
        tmp_s8[index] = (obstacles[index]) ? tmp : tmp_s8[index] + params.omega * (d_equ[8] - tmp_s8[index]);

        //AVERAGE VELOCITY CODE
        /* local density total */
        local_density = 0.f;
        local_density += tmp_s0[index];
        local_density += tmp_s1[index];
        local_density += tmp_s2[index];
        local_density += tmp_s3[index];
        local_density += tmp_s4[index];
        local_density += tmp_s5[index];
        local_density += tmp_s6[index];
        local_density += tmp_s7[index];
        local_density += tmp_s8[index];
        local_density = 1/local_density;

        /* x-component of velocity */
        u_x = (tmp_s1[index]
                      + tmp_s5[index]
                      + tmp_s8[index]
                      - tmp_s3[index]
                      - tmp_s6[index]
                      - tmp_s7[index])
                     * local_density;
        /* compute y velocity component */
        u_y = (tmp_s2[index]
                      + tmp_s5[index]
                      + tmp_s6[index]
                      - tmp_s4[index]
                      - tmp_s7[index]
                      - tmp_s8[index])
                     * local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += (obstacles[index]) ? 0 : sqrtf((u_x * u_x) + (u_y * u_y)) ;
        /* increase counter of inspected cells */
        tot_cells += (obstacles[index]) ? 0 : 1 ;


    }
  }

  return tot_u / (float)tot_cells;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_scells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_scells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_scells_ptr == NULL) die("cannot allocate memory for tmp_scells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_scells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_scells_ptr);
  *tmp_scells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
