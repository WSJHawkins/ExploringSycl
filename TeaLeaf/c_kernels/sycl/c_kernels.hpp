#include "../../settings.h"

/* STORE ENERGY */
// Copies energy0 into energy1.
void store_energy(
  const int x, const int y, SyclBuffer& energyBuff, SyclBuffer& energy0Buff,
  queue& device_queue);

/* SET CHUNK DATA */
// Initialises the vertices
void set_chunk_data_vertices(
  const int x, const int y, const int halo_depth, SyclBuffer& vertex_xBuff,
  SyclBuffer& vertex_yBuff, const double x_min, const double y_min,
  const double dx, const double dy, queue& device_queue);

// Sets all of the cell data for a chunk
void set_chunk_data(
  const int x, const int y, const int halo_depth, SyclBuffer& vertex_xBuff,
  SyclBuffer& vertex_yBuff, SyclBuffer& cell_xBuff, SyclBuffer& cell_yBuff,
	SyclBuffer& volumeBuff, SyclBuffer& x_areaBuff, SyclBuffer& y_areaBuff,
	const double x_min, const double y_min, const double dx, const double dy,
	queue& device_queue);

/* SET CHUNK STATE */
// Sets the initial state for the chunk
void set_chunk_initial_state(
  const int x, const int y, double default_energy, double default_density,
  SyclBuffer& energy0Buff, SyclBuffer& densityBuff, queue& device_queue);

// Sets all of the additional states in order
void set_chunk_state(
  const int x, const int y, const int halo_depth, State state,
  SyclBuffer& energy0Buff, SyclBuffer& densityBuff, SyclBuffer& uBuff,
  SyclBuffer& cell_xBuff, SyclBuffer& cell_yBuff,  SyclBuffer& vertex_xBuff,
  SyclBuffer& vertex_yBuff, queue& device_queue);

/* SOLVER METHODS */
// Copies the inner u into u0.
void copy_u(
  const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
  SyclBuffer& u0Buff, queue& device_queue);

// Calculates the residual r.
void calculate_residual(
  const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
  SyclBuffer& u0Buff, SyclBuffer& rBuff, SyclBuffer& kxBuff, SyclBuffer& kyBuff,
  queue& device_queue);

// Calculates the 2 norm of the provided buffer.
void calculate_2norm(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  double* norm, queue& device_queue);

// Finalises the energy field.
void finalise(
  const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
  SyclBuffer& densityBuff, SyclBuffer& energyBuff, queue& device_queue);

/* PPCG SOLVER */
// Initialises Sd
void ppcg_init(
  const int x, const int y, const int halo_depth, const double theta,
  SyclBuffer& sdBuff, SyclBuffer& rBuff, queue& device_queue);

// Calculates U and R
void ppcg_calc_ur(
  const int x, const int y, const int halo_depth, SyclBuffer& sdBuff,
  SyclBuffer& rBuff, SyclBuffer& uBuff, SyclBuffer& kxBuff, SyclBuffer& kyBuff,
  queue& device_queue);

// Calculates Sd
void ppcg_calc_sd(
  const int x, const int y, const int halo_depth, const double theta,
  const double alpha, const double beta, SyclBuffer& sdBuff, SyclBuffer& rBuff,
  queue& device_queue);

/* CHEBY SOLVER */
// Initialises the Chebyshev solver
void cheby_init(
  const int x, const int y, const int halo_depth, const double theta,
  SyclBuffer& pBuff, SyclBuffer& rBuff, SyclBuffer& uBuff, SyclBuffer& u0Buff,
  SyclBuffer& wBuff, SyclBuffer& kxBuff, SyclBuffer& kyBuff, queue& device_queue);

// Calculates U
void cheby_calc_u(
  const int x, const int y, const int halo_depth, SyclBuffer& pBuff,
  SyclBuffer& uBuff, queue& device_queue);

// The main Cheby iteration step
void cheby_iterate(
  const int x, const int y, const int halo_depth,
  const double alpha, const double beta, SyclBuffer& pBuff, SyclBuffer& rBuff,
  SyclBuffer& uBuff, SyclBuffer& u0Buff,  SyclBuffer& wBuff, SyclBuffer& kxBuff,
  SyclBuffer& kyBuff, queue& device_queue);

/* CG SOLVER */
// Calculates a value for p
void cg_calc_p(
  const int x, const int y, const int halo_depth, const double beta,
  SyclBuffer& pBuff, SyclBuffer& rBuff, queue& device_queue);

// Calculates the value of u and r
void cg_calc_ur(
  const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
  SyclBuffer& rBuff, SyclBuffer& pBuff, SyclBuffer& wBuff, const double alpha,
  double* rrn, queue& device_queue);

// Calculates the value for w
void cg_calc_w(
  const int x, const int y, const int halo_depth, SyclBuffer& wBuff,
  SyclBuffer& pBff, SyclBuffer& kxBuff, SyclBuffer& kyBuff, double* pw,
  queue& device_queue);

// Initialises kx,ky
void cg_init_k(
  const int x, const int y, const int halo_depth, SyclBuffer& wBuff,
  SyclBuffer& kxBuff, SyclBuffer& kyBuff, const double rx, const double ry,
  queue& device_queue);

// Initialises w,r,p and calculates rro
void cg_init_others(
  const int x, const int y, const int halo_depth, SyclBuffer& kxBuff,
  SyclBuffer& kyBuff, SyclBuffer& pBuff, SyclBuffer& rBuff, SyclBuffer& uBuff,
  SyclBuffer& wBuff, double* rro, queue& device_queue);

// Initialises p,r,u,w
void cg_init_u(
  const int x, const int y, const int coefficient,
  SyclBuffer& pBuff, SyclBuffer& rBuff, SyclBuffer& uBuff, SyclBuffer& wBuff,
  SyclBuffer& densityBuff, SyclBuffer& energyBuff, queue& device_queue);

/* JACOBI SOLVER */
// Initialises the Jacobi solver
void jacobi_init(
  const int x, const int y, const int halo_depth,
  const int coefficient, const double rx, const double ry, SyclBuffer& uBuff,
  SyclBuffer& u0Buff, SyclBuffer& densityBuff, SyclBuffer& energyBuff,
	SyclBuffer& kxBuff, SyclBuffer& kyBuff, queue& device_queue);

// Copies u into r
void jacobi_copy_u(
	const int x, const int y, SyclBuffer& rBuff, SyclBuffer& uBuff,
  queue& device_queue);

// Main Jacobi solver method.
void jacobi_iterate(
  const int x, const int y, const int halo_depth, SyclBuffer& u, SyclBuffer& u0,
  SyclBuffer& r, SyclBuffer& kx, SyclBuffer& ky, double* error,
  queue& device_queue);

/* UPDATING LOCAL HALOS */
// Updates the local left halo region(s)
void update_left(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue);

// Updates the local right halo region(s)
void update_right(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue);

// Updates the local top halo region(s)
void update_top(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue);

// Updates the local bottom halo region(s)
void update_bottom(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue);

/* PACK HALOS */
// Packs the top halo buffer(s)
void pack_top(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

// Packs the bottom halo buffer(s)
void pack_bottom(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

// Unpacks the top halo buffer(s)
void unpack_top(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

// Unpacks the bottom halo buffer(s)
void unpack_bottom(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

// Packs the left halo buffer(s)
void pack_left(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

// Packs the right halo buffer(s)
void pack_right(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

// Unpacks the left halo buffer(s)
void unpack_left(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

// Unpacks the right halo buffer(s)
void unpack_right(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  SyclBuffer& fieldBuff, const int depth, queue& device_queue);

//Manages field summary
void field_summary_func(
	const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
	SyclBuffer& densityBuff, SyclBuffer& energy0Buff, SyclBuffer& volumeBuff,
	double* vol, double* mass, double* ie, double* temp, queue& device_queue);
